
import os
import json
import difflib
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st


#           SETTINGS

APP_TITLE = "🧠 MLP Next-Word/Line Generator"
APP_VERSION = "1.4.1"
DEFAULT_CKPT_DIR = "/content/drive/MyDrive/Model_Checkpoints/weights"   # your checkpoints
DEFAULT_JSON_DIR  = "/content/drive/MyDrive/es335-assignment-3"          # optional JSONs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#          UTILITIES

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def strip_compile_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        return {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    return sd

def id2tok_from_inv(inv_vocab: Dict) -> List[str]:
    # inv_vocab keys can be str or int
    keys = [int(k) for k in inv_vocab.keys()] if inv_vocab and isinstance(next(iter(inv_vocab.keys())), str) else list(inv_vocab.keys())
    max_id = max(keys) if keys else 0
    arr = [""] * (max_id + 1)
    for k, v in inv_vocab.items():
        idx = int(k) if isinstance(k, str) else k
        if idx < 0: 
            continue
        if idx >= len(arr):
            arr.extend([""] * (idx - len(arr) + 1))
        arr[idx] = v
    for i, t in enumerate(arr):
        if t == "":
            arr[i] = "<UNK>"
    return arr

def best_fuzzy_match(tok: str, vocab: Dict[str, int], cutoff: float = 0.8) -> Optional[str]:
    matches = difflib.get_close_matches(tok, vocab.keys(), n=1, cutoff=cutoff)
    return matches[0] if matches else None

def normalize_token(tok: str) -> str:
    return tok.strip()

def map_oov_token(tok: str, vocab: Dict[str, int], id2tok: List[str], strategy: str, dataset_hint: str) -> str:
    if tok in vocab: 
        return tok
    if strategy == "fuzzy":
        cand = best_fuzzy_match(tok, vocab, cutoff=0.75)
        if cand: 
            return cand
    if strategy == "skip":
        return ""
    # fallback
    if dataset_hint == "code" and "<NL>" in vocab:
        return "<NL>"
    if "." in vocab:
        return "."
    return id2tok[0] if id2tok else next(iter(vocab.keys()))

def encode_context(words: List[str], vocab: Dict[str, int], id2tok: List[str], context_len: int,
                   oov_strategy: str, dataset_hint: str) -> torch.Tensor:
    pad = "." if "." in vocab else "<NL>" if "<NL>" in vocab else (id2tok[0] if id2tok else next(iter(vocab.keys())))
    toks = []
    for w in words:
        w = normalize_token(w)
        mapped = map_oov_token(w, vocab, id2tok, oov_strategy, dataset_hint)
        if mapped != "":
            toks.append(mapped)
    toks = toks[-context_len:]
    toks = ([pad] * (context_len - len(toks))) + toks
    idx = [vocab[t] for t in toks]
    return torch.tensor([idx], dtype=torch.long, device=DEVICE)


#            MODEL

class MLPTextGen(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_len: int,
                 emb_dim: int = 64,
                 hidden_layers: int = 1,
                 hidden_dim: int = 1024,
                 activation: str = "relu",
                 dropout: float = 0.2,
                 use_adaptive_softmax: bool = False,
                 adaptive_cutoffs: Optional[List[int]] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.use_adaptive = use_adaptive_softmax

        self.emb = nn.Embedding(vocab_size, emb_dim)
        act = nn.ReLU if activation.lower() == "relu" else nn.Tanh
        layers = [nn.Linear(context_len * emb_dim, hidden_dim), act(), nn.Dropout(dropout)]
        if hidden_layers == 2:
            layers += [nn.Linear(hidden_dim, hidden_dim), act(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, emb_dim, bias=False)

        if self.use_adaptive:
            if adaptive_cutoffs is None:
                c1 = min(20000, vocab_size // 10)
                c2 = min(60000, vocab_size // 2)
                adaptive_cutoffs = [c for c in [c1, c2] if c < vocab_size]
            self.adaptive = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=emb_dim, n_classes=vocab_size, cutoffs=adaptive_cutoffs
            )
            self.decoder = None
        else:
            self.decoder = nn.Linear(emb_dim, vocab_size, bias=False)
            self.decoder.weight = self.emb.weight  # weight tying

    @torch.no_grad()
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        e = self.emb(x)
        h = self.mlp(e.reshape(e.size(0), -1))
        z = self.proj(h)
        if self.use_adaptive:
            return self.adaptive.log_prob(z)
        else:
            return F.log_softmax(self.decoder(z), dim=-1)


#       CHECKPOINT LOADING

def safe_load_checkpoint(path: Path):
    if not path.exists() or not path.is_file():
        raise ValueError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=DEVICE)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")

    needed = ["model_state", "config", "vocab", "inv_vocab"]
    if not isinstance(ckpt, dict) or any(k not in ckpt for k in needed):
        raise ValueError("Checkpoint must contain keys: model_state, config, vocab, inv_vocab")

    sd = strip_compile_prefix(ckpt["model_state"])
    cfg = ckpt["config"]; vocab = ckpt["vocab"]; inv_vocab = ckpt["inv_vocab"]

    for k in ["vocab_size","context_len","emb_dim","hidden_layers","hidden_dim","activation","dropout"]:
        if k not in cfg:
            raise ValueError(f"Config missing key: {k}")

    model = MLPTextGen(
        vocab_size=cfg["vocab_size"],
        context_len=cfg["context_len"],
        emb_dim=cfg["emb_dim"],
        hidden_layers=cfg["hidden_layers"],
        hidden_dim=cfg["hidden_dim"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
        use_adaptive_softmax=cfg.get("use_adaptive_softmax", False),
    ).to(DEVICE)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Non-fatal: warn in console
        print("State dict warnings:", "missing:", missing, "unexpected:", unexpected)

    model.eval()
    return model, cfg, vocab, inv_vocab

@st.cache_resource(show_spinner=False)
def cached_load(path_str: str):
    return safe_load_checkpoint(Path(path_str))

def list_checkpoints(folder: str) -> List[Path]:
    if not folder or not Path(folder).exists():
        return []
    return sorted([p for p in Path(folder).glob("*.pt")])


#         GENERATION

@torch.no_grad()
def generate(model: MLPTextGen,
             start_words: List[str],
             vocab: Dict[str, int],
             inv_vocab: Dict[str, str],
             k_steps: int = 20,
             temperature: float = 1.0,
             topk: Optional[int] = 50,
             context_len_override: Optional[int] = None,
             oov_strategy: str = "fallback",
             dataset_hint: str = "natural",
             seed: int = 1337) -> str:
    set_seed(seed)
    id2tok = id2tok_from_inv(inv_vocab)
    context_len = min(int(context_len_override or model.context_len), model.context_len)
    out = list(start_words)

    for _ in range(k_steps):
        x = encode_context(out, vocab, id2tok, context_len, oov_strategy, dataset_hint)
        logp = model.log_prob(x) / max(temperature, 1e-8)
        if topk is not None and topk > 0 and topk < logp.size(-1):
            v, ix = torch.topk(logp, topk)
            mask = torch.full_like(logp, float("-inf"))
            logp = mask.scatter(1, ix, v)
        probs = torch.softmax(logp, dim=-1).squeeze(0)
        nxt_id = torch.multinomial(probs, num_samples=1).item()
        nxt_tok = inv_vocab[str(nxt_id)] if isinstance(inv_vocab, dict) else id2tok[nxt_id]
        out.append(nxt_tok)
    return " ".join(out)


#             UI

import requests, zipfile, io

# URL of GitHub release asset
WEIGHTS_URL = "https://github.com/ClairvoyantCoder/es335-assignment3/releases/download/v1.0/models_release.zip"
CACHE_DIR = Path("weights")
CACHE_DIR.mkdir(exist_ok=True)

# Check if already downloaded
if not any(CACHE_DIR.glob("*.pt")):
    st.sidebar.info("Downloading model weights from GitHub release...")
    try:
        r = requests.get(WEIGHTS_URL)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(CACHE_DIR)
        st.sidebar.success("Weights downloaded successfully ✅")
    except Exception as e:
        st.sidebar.error(f"Failed to download weights: {e}")
        st.stop()

# After this, ckpt_dir will point to local 'weights'
DEFAULT_CKPT_DIR = CACHE_DIR.as_posix()


st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")
st.title(f"{APP_TITLE} · {APP_VERSION}")
st.caption("Load a trained variant, set temperature/top-k/seed/context, and generate next k tokens. OOV-safe.")

# --- Sidebar: where are the models? ---
st.sidebar.header("Model Variants")
ckpt_dir = st.sidebar.text_input("Checkpoint folder", value=DEFAULT_CKPT_DIR)
refresh = st.sidebar.button("🔄 Refresh list")

# Discover checkpoints
ckpts = list_checkpoints(ckpt_dir)
if not ckpts:
    st.warning("No checkpoints found in the provided folder. Make sure Drive is mounted and the path is correct.")
    st.info(f"Expected path example: {DEFAULT_CKPT_DIR}")
    st.stop()

# Let user choose a variant (string path so it persists across reruns)
variant_path_str = st.sidebar.selectbox(
    "Choose variant (.pt)", options=[p.as_posix() for p in ckpts], index=0
)
if not variant_path_str:
    st.warning("Please select a model variant to continue.")
    st.stop()

# --- Load model safely (cached) ---
try:
    with st.spinner("Loading model…"):
        model, cfg, vocab, inv_vocab = cached_load(variant_path_str)
except Exception as e:
    st.error(f"Failed to load checkpoint:\n\n{e}")
    st.stop()

# Only now is it safe to reference cfg/vocab
dataset_hint = "code" if "<NL>" in vocab else "natural"
st.success(
    f"Loaded ✅  |  Variant: **{Path(variant_path_str).name}**  |  Vocab: **{len(vocab)}**  |  "
    f"Context len: **{cfg['context_len']}**  |  Adaptive: **{cfg.get('use_adaptive_softmax', False)}**"
)

# --- Sidebar: generation controls (after cfg exists) ---
st.sidebar.header("Generation Controls")
context_len_override = st.sidebar.number_input(
    "Context length (≤ model)", min_value=1, max_value=cfg["context_len"], value=min(5, cfg["context_len"])
)
temperature = st.sidebar.slider("Temperature", 0.2, 2.0, 1.0, 0.05)
topk = st.sidebar.slider("Top-k (0=disabled)", 0, 200, 50, 5)
topk = None if topk == 0 else topk
k_steps = st.sidebar.number_input("Predict next k tokens", 1, 200, 30)
seed = st.sidebar.number_input("Random seed", 1, 999999, 1337)
oov_strategy = st.sidebar.selectbox("OOV handling", ["fallback", "fuzzy", "skip"], index=0)

# --- Sidebar: model info ---
st.sidebar.header("Model Config")
st.sidebar.write(f"Embedding dim: **{cfg['emb_dim']}**")
st.sidebar.write(f"Hidden layers: **{cfg['hidden_layers']}**")
st.sidebar.write(f"Hidden size: **{cfg['hidden_dim']}**")
st.sidebar.write(f"Activation: **{cfg['activation']}**")
st.sidebar.write(f"Dropout: **{cfg['dropout']}**")

# --- Main: input & generation ---
st.subheader("Input")
default_prompt = "to sherlock holmes she is" if dataset_hint == "natural" else "for ( i = 0;"
user_text = st.text_input("Enter context (space-separated tokens):", value=default_prompt)

c1, c2, c3 = st.columns([1,1,1])
with c1:
    do_generate = st.button("🚀 Generate")
with c2:
    show_vocab = st.checkbox("Show vocab size", value=False)
with c3:
    show_oov = st.checkbox("Show OOV mapping example", value=False)

if do_generate:
    tokens = [t for t in user_text.strip().split() if t]
    t0 = time.time()
    try:
        out = generate(
            model=model,
            start_words=tokens,
            vocab=vocab,
            inv_vocab=inv_vocab,
            k_steps=int(k_steps),
            temperature=float(temperature),
            topk=topk,
            context_len_override=int(context_len_override),
            oov_strategy=oov_strategy,
            dataset_hint=dataset_hint,
            seed=int(seed),
        )
        dt = time.time() - t0
        st.subheader("Output")
        st.code(out)
        st.caption(f"Generated in {dt:.2f}s on {DEVICE.type}")
    except Exception as e:
        st.error(f"Generation failed: {e}")

if show_vocab:
    st.info(f"Vocabulary size: **{len(vocab)}**  ·  Dataset: **{dataset_hint}**")

if show_oov:
    sample_oov = "Sherloc" if dataset_hint == "natural" else "memcoy"
    mapped = map_oov_token(sample_oov, vocab, id2tok_from_inv(inv_vocab), oov_strategy, dataset_hint)
    st.write(f"OOV example: `{sample_oov}` → `{mapped}` (strategy: **{oov_strategy}**)")

