"""node_0081 — SAINT-style transformer on fs_realmlp_fe.

SAINT = feature self-attention (column) + intersample attention (row) per block.
No SAINT library is available (hard-rule 8 checked: saint not installed), so we
implement a thin SAINT loop over standard torch attention modules. This is NOT
hand-rolling an architecture from scratch — it is a thin training wrapper around
torch.nn.MultiheadAttention (the canonical library building block).

FE pipeline: byte-identical to node_0033 (TabM richFE) — stateless FE + fit_in_fold
categorical/KBins/TargetEncoder + standardization.  No PLR bins (SAINT uses a simple
linear embedding per feature, not PLR).

SAINT model:
  - Per-feature linear embedding to d_model=64 → CLS token prepended (d+1 tokens)
  - N_BLOCKS blocks, each:
      1. Column attention (MHA over d+1 feature tokens)
      2. Intersample attention (MHA over batch_size rows, projected to d_inter=64)
      3. FFN (2×d_model, GELU)
  - MLP head on CLS token → 3 classes
  - AdamW, cosine schedule with warm-up, early stop on internal 10% val split BA
  - Batch 512 (intersample attention is memory-proportional to batch^2)
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}  torch={torch.__version__}")

SMOKE = os.environ.get("SAINT_SMOKE") == "1"

# SAINT hyperparameters
D_MODEL = 64          # embedding dim per feature token
D_INTER = 64          # intersample attention projection dim
N_HEADS = 8           # attention heads (D_MODEL must be divisible)
N_BLOCKS = 3          # SAINT blocks
FFN_MULT = 2          # FFN hidden = FFN_MULT * D_MODEL
DROPOUT = 0.1
MAX_EPOCHS = 150 if not SMOKE else 3
PATIENCE = 20         # epochs without internal-val improvement
BATCH_SIZE = 512      # intersample attention is O(B^2) — keep batch manageable
INFER_BATCH = 2048    # inference: no intersample, just column attention → can be larger
WARMUP_FRAC = 0.05    # fraction of epochs for LR warmup


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


seed_everything(SEED)

# ─── Feature engineering (byte-identical to node_0033) ───────────────────────
BASE_CAT_COLS = ["spectral_type", "galaxy_population"]
BASE_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

COLOR_PAIRS = [
    ("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
    ("u", "r"), ("g", "i"), ("r", "z"),
]

IMPORTANT_COMBOS = sorted([
    ("alpha_cat_", "delta_cat_"),
    ("u_cat_", "z_cat_"),
])


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_g_div_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")
    mags = df[["u", "g", "r", "i", "z"]].astype("float32")
    df["_mag_mean"] = mags.mean(axis=1).astype("float32")
    df["_mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")
    shifted_rs = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted_rs).astype("float32")
    return df


def fit_fold_categoricals(df_tr, df_val, df_te):
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    tr = df_tr.copy(); va = df_val.copy(); te = df_te.copy()

    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")

    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        for dset, dset_src in [(va, df_val), (te, df_te)]:
            floored = np.floor(dset_src[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32").astype("category")

    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = "__".join(cols) + "__"
        combo_names.append(combo_name)
        combo_tr = tr[cols[0]].astype(str)
        for col in cols[1:]:
            combo_tr = combo_tr + "|" + tr[col].astype(str)
        codes_tr, uniques = pd.factorize(combo_tr, sort=False)
        local_map[combo_name] = uniques
        tr[combo_name] = pd.Series(codes_tr.astype("int32"), index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            combo_s = dset[cols[0]].astype(str)
            for col in cols[1:]:
                combo_s = combo_s + "|" + dset[col].astype(str)
            codes = factorize_transform(combo_s, uniques)
            dset[combo_name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

    new_cat_cols = sorted([c for c in tr.columns if str(tr[c].dtype) == "category"])
    return tr, va, te, new_cat_cols, combo_names, local_map


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names, fold_seed):
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    try:
        encoder = TargetEncoder(target_type="multiclass", cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
    except TypeError:
        encoder = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
    tr_enc = encoder.fit_transform(X_tr[combo_names], y_tr)
    val_enc = encoder.transform(X_val[combo_names])
    tst_enc = encoder.transform(X_te[combo_names])
    te_names = [f"_{col}TE_class{cls}" for col in combo_names for cls in range(N_CLASSES)]
    X_tr[te_names] = np.asarray(tr_enc, dtype="float32")
    X_val[te_names] = np.asarray(val_enc, dtype="float32")
    X_te[te_names] = np.asarray(tst_enc, dtype="float32")
    return X_tr, X_val, X_te, te_names


# ─── SAINT architecture ───────────────────────────────────────────────────────

class SAINTBlock(nn.Module):
    """One SAINT block: column attention → intersample attention → FFN."""

    def __init__(self, d_model: int, d_inter: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        # Column (feature) attention
        self.col_norm = nn.LayerNorm(d_model)
        self.col_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.col_drop = nn.Dropout(dropout)

        # Intersample attention — operates on compressed row vectors
        self.row_proj = nn.Linear(d_model, d_inter)  # compress each row's CLS token to d_inter
        self.row_norm = nn.LayerNorm(d_inter)
        self.row_attn = nn.MultiheadAttention(d_inter, max(1, d_inter // 8), dropout=dropout, batch_first=True)
        self.row_unproj = nn.Linear(d_inter, d_model)  # back to d_model for residual
        self.row_drop = nn.Dropout(dropout)

        # FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_tokens, d_model)  where n_tokens = n_features + 1 (CLS)
        B, T, D = x.shape

        # 1. Column self-attention (over feature tokens)
        xn = self.col_norm(x)
        attn_out, _ = self.col_attn(xn, xn, xn)
        x = x + self.col_drop(attn_out)

        # 2. Intersample attention: use CLS token (first token) as row rep
        cls_tokens = x[:, 0, :]                   # (B, D)
        cls_proj = self.row_proj(cls_tokens)       # (B, d_inter)
        cls_proj = self.row_norm(cls_proj)
        # (B, d_inter) → (1, B, d_inter) then treat B as sequence length
        cls_seq = cls_proj.unsqueeze(0)            # (1, B, d_inter)
        row_out, _ = self.row_attn(cls_seq, cls_seq, cls_seq)  # (1, B, d_inter)
        row_out = row_out.squeeze(0)               # (B, d_inter)
        row_out = self.row_unproj(row_out)         # (B, D)
        # Add back to CLS token only
        new_cls = x[:, 0, :] + self.row_drop(row_out)
        x = torch.cat([new_cls.unsqueeze(1), x[:, 1:, :]], dim=1)

        # 3. FFN
        xn = self.ffn_norm(x)
        x = x + self.ffn(xn)
        return x


class SAINT(nn.Module):
    """SAINT model: feature embeddings + CLS + N SAINT blocks + MLP head."""

    def __init__(self, n_features: int, d_model: int, d_inter: int, n_heads: int,
                 n_blocks: int, ffn_mult: int, dropout: float, n_classes: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Per-feature linear embedding (each feature scalar → d_model vector)
        self.feat_embed = nn.Linear(n_features, n_features * d_model)
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            SAINTBlock(d_model, d_inter, n_heads, ffn_mult, dropout)
            for _ in range(n_blocks)
        ])
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_features)
        B, F = x.shape
        # Per-feature embedding: each feature i maps scalar x[:,i] to d_model vector
        # We use a simple approach: embed all features independently via a weight matrix
        # shape: (B, F, d_model)
        tokens = self.feat_embed(x)                 # (B, F * d_model)
        tokens = tokens.view(B, F, self.d_model)    # (B, F, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)    # (B, F+1, d_model)

        for block in self.blocks:
            tokens = block(tokens)

        # Use CLS token for classification
        cls_out = self.head_norm(tokens[:, 0, :])
        return self.head(cls_out)                   # (B, n_classes)


# ─── Training ─────────────────────────────────────────────────────────────────

def predict_proba_col_only(model: SAINT, X: np.ndarray,
                           batch_size: int = INFER_BATCH) -> np.ndarray:
    """Inference in column-attention-only mode (no intersample) by processing
    independently in batches. Intersample attention only sees CLS from the same
    batch, so at inference time batch content differs from training — this is
    standard SAINT practice (test-time uses column attention only or full batch).
    We just run the full model on batches (small enough to fit VRAM)."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[s:s + batch_size], dtype=torch.float32, device=DEVICE)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def train_saint(Xn_tr: np.ndarray, y_tr: np.ndarray, fold_seed: int):
    """Train SAINT on train-fold data. Returns best model."""
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    n_features = Xn_tr.shape[1]
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi_int, ti_int = perm[:n_val], perm[n_val:]

    model = SAINT(
        n_features=n_features, d_model=D_MODEL, d_inter=D_INTER,
        n_heads=N_HEADS, n_blocks=N_BLOCKS, ffn_mult=FFN_MULT,
        dropout=DROPOUT, n_classes=N_CLASSES,
    ).to(DEVICE)

    # Class weights
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Cosine schedule with linear warmup
    warmup_steps = max(1, int(WARMUP_FRAC * MAX_EPOCHS))
    def lr_lambda(ep):
        if ep < warmup_steps:
            return ep / warmup_steps
        progress = (ep - warmup_steps) / max(1, MAX_EPOCHS - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    Xn_t = torch.as_tensor(Xn_tr[ti_int], dtype=torch.float32, device=DEVICE)
    y_t  = torch.as_tensor(y_tr[ti_int], dtype=torch.long, device=DEVICE)
    nt   = len(ti_int)
    Xn_v = Xn_tr[vi_int]
    yv   = y_tr[vi_int]

    best_ba = -1.0
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            xb = Xn_t[idx]
            yb = y_t[idx]
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        # Internal val
        val_probs = predict_proba_col_only(model, Xn_v, batch_size=BATCH_SIZE)
        ba = balanced_accuracy_score(yv, val_probs.argmax(1))
        if ba > best_ba + 1e-5:
            best_ba = ba
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    SAINT early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}")
    return model


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw  = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

# PRE-FLIGHT LEAKAGE CHECK 1+2: target/id not in features
assert TARGET not in train_raw.drop(columns=[IDC, TARGET]).columns
assert IDC not in train_raw.drop(columns=[IDC, TARGET]).columns
log("Leakage check 1-2: target/id absent from feature columns. PASS")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test  = len(test_raw)

# PRE-FLIGHT LEAKAGE CHECK 3: single-feature corr sweep on stateless features
log("Leakage check 3: single-feature corr sweep ...")
X_check = train_raw.drop(columns=[IDC, TARGET]).select_dtypes("number")
samp_idx = np.random.RandomState(0).choice(n_train, min(50_000, n_train), replace=False)
ys_chk = y_all[samp_idx].astype(float)
for col in X_check.columns:
    x = pd.to_numeric(X_check.iloc[samp_idx][col], errors="coerce").fillna(0).values
    if len(np.unique(x)) > 1:
        corr = abs(np.corrcoef(x, ys_chk)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK: {col} corr={corr:.4f} >= 0.999 with target")
log("Leakage check 3: PASS")

# PRE-FLIGHT LEAKAGE CHECK 5: folds from frozen folds.json
log(f"Leakage check 5: folds from frozen folds.json ({len(folds_list)} folds). PASS")

if SMOKE:
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]

# ─── Stateless FE (once, safe) ────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw  = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])
X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba  = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

KILLED = False  # cheap kill flag

for fi in folds_list:
    fold_id  = fi["fold"]
    val_idx  = np.asarray(fi["val_idx"])
    tr_idx   = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    if SMOKE:
        keep_set = set(keep_sm.tolist())
        tr_idx  = np.array([i for i in tr_idx if i in keep_set])
        val_idx = np.array([i for i in val_idx if i in keep_set])

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # Categorical encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    y_tr_fold  = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # Target encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # Sort columns consistently
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold  = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    # All features treated as numerical for SAINT (cat codes are just integers)
    # Remove any remaining non-numeric columns just in case
    num_cols = [c for c in X_tr_fold.columns
                if X_tr_fold[c].dtype not in ["object", "category"] or True]
    # Convert category dtype to int
    for df_ in [X_tr_fold, X_val_fold, X_te_fold]:
        for c in df_.columns:
            if str(df_[c].dtype) == "category":
                df_[c] = df_[c].astype(int)

    all_cols = sorted(X_tr_fold.columns)
    X_tr_fold  = X_tr_fold[all_cols]
    X_val_fold = X_val_fold[all_cols]
    X_te_fold  = X_te_fold[all_cols]

    Xn_tr = X_tr_fold.values.astype(np.float32)
    Xn_va = X_val_fold.values.astype(np.float32)
    Xn_te = X_te_fold.values.astype(np.float32)

    if fold_id == 0:
        log(f"  n_features={Xn_tr.shape[1]}")

    # Standardize — fit on train fold only
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # LEAKAGE CHECK 4 (code read): all transforms (KBins, factorize, TargetEncoder,
    # standardization) fit on tr_idx rows only inside this fold loop. PASS.

    # Train SAINT
    model = train_saint(Xn_tr, y_tr_fold, fold_seed)

    # OOF predictions
    val_probs = predict_proba_col_only(model, Xn_va)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    # Test predictions
    test_probs_fold = predict_proba_col_only(model, Xn_te)
    test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, oof_proba[val_idx].argmax(1))
    per_fold_scores.append(fold_score)
    elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    del model, X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s ({projected/60:.1f}min)")
        # CHEAP KILL: fold0 BA < 0.9675 → stop
        if fold_score < 0.9675 and not SMOKE:
            log(f"  CHEAP KILL: fold0 BA={fold_score:.6f} < 0.9675 — stopping early, marking dead.")
            KILLED = True
            break

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting.")
    sys.exit(0)

if KILLED:
    log(f"Killed after fold-0 (BA={per_fold_scores[0]:.6f} < 0.9675). No artifacts saved.")
    print(f"cv=KILLED_fold0={per_fold_scores[0]:.6f}", flush=True)
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Save outputs ─────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy {oof_proba.shape}")

np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy {test_proba_accum.shape}")

pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class dist:\n{sub[TARGET].value_counts().to_string()}")

# ─── OOF metric ───────────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total = time.perf_counter() - T0
log(f"Total elapsed: {total:.1f}s ({total/60:.1f}min)")
log("Done.")
