"""node_0055 — DCN/CrossNet NN base on fs_realmlp_fe features.

THE ONE ATOMIC CHANGE vs node_0033:
  Replace TabM (k=32 internal ensemble + PLR embeddings) with a DCN (Deep & Cross Network).
  Architecture: DenseInput (per-categorical embeddings) → CrossNet (explicit feature-cross
  layers, 7 cross levels) in parallel with a deep MLP block (hidden=768, SiLU, dropout=0.10),
  concat → linear head.
  Training: per-class-balanced CE + label_smoothing=0.002, EMA (decay=0.995), AdamW lr=6e-4,
  cosine schedule, 84 epochs / patience=16, micro-batch=1024 with grad-accum=4.
  qbin embeddings (levels [16,32,64]) and artifact-count features — all FIT-IN-FOLD.
  NO SDSS17 external data (append_original=False).

FE pipeline (same fs_realmlp_fe as node_0033):
  - Stateless FE: redshift ratios, 7 color pairs, mag_mean, mag_range, log1p_redshift,
    integer-floor categorical views of every base numeric — computed once on full df (safe).
  - fit_in_fold KBinsDiscretizer (delta 100/500 quantile bins) on train fold only.
  - fit_in_fold TargetEncoder (on combo cats) on train fold only.
  - fit_in_fold standardization (mean/std from train fold numerical features only).
  - fit_in_fold qbin embeddings [16, 32, 64 levels] on top numeric cols.
  - fit_in_fold artifact-count features (floor + rare count).

DCN model:
  - DenseInput: per-cat embeddings (dim = min(32, max(2, round(card^0.25 * 3.5)))), emb_dropout=0.05.
  - CrossNet: 7 cross layers (explicit multiplicative feature-cross).
  - MLP: 4 blocks of (Linear(hidden,hidden), SiLU, Dropout(0.10)).
  - Head: LayerNorm(input_dim + hidden), Linear -> 3.
  - EMA model for inference.

Leakage discipline:
  - Stateless FE: no target, no cross-row stats, no fitting — safe to compute once.
  - KBinsDiscretizer, factorize maps, TargetEncoder: fit on train-fold rows only.
  - Standardization (mean/std): fit on train-fold numerical features only.
  - qbin edges: fit on train-fold rows only.
  - artifact floor stats: fit on train-fold rows only (cat_maps, count frequencies from train fold).
  - Frozen folds.json used throughout; no refitting of folds.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
Also runs CORE15 + n55 re-stack A/B.
"""
from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, TargetEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ─── GPU check ───────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    GPU_GB = props.total_memory / 1024**3
    # verify real GPU matmul
    _a = torch.ones(512, 512, device=DEVICE)
    _b = torch.ones(512, 512, device=DEVICE)
    _c = (_a @ _b).sum().item()
    del _a, _b, _c
    log(f"GPU: {props.name} | memory={GPU_GB:.2f} GiB | matmul OK")
else:
    GPU_GB = 0.0
    log("WARNING: no CUDA — running on CPU")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ─── Constants ────────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

SMOKE = os.environ.get("DCN_SMOKE") == "1"

# ─── Hyper-parameters (from nn-v2-for-s6e6.py ref) ───────────────────────────
CFG = SimpleNamespace(
    seed=20260604,
    epochs=84 if not SMOKE else 3,
    patience=16,
    batch_size=4096,
    micro_batch_size=1024,
    grad_accum_steps=4,
    eval_batch_size=8192,
    hidden=768,
    blocks=4,
    cross_layers=7,
    dropout=0.10,
    emb_dropout=0.05,
    max_emb_dim=32,
    lr=6e-4,
    weight_decay=1e-4,
    label_smoothing=0.002,
    ema_decay=0.995,
    grad_clip=5.0,
    qbin_levels=[16, 32, 64],
    qbin_cols=28,
    rare_min_count=10,
    amp=torch.cuda.is_available(),
    pin_memory=torch.cuda.is_available(),
    report_every=5,
    threads=12,
)

torch.set_num_threads(CFG.threads)
try:
    torch.set_num_interop_threads(max(1, min(4, CFG.threads // 4)))
except RuntimeError:
    pass


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── Feature engineering globals (same as node_0033 / fs_realmlp_fe) ─────────
BASE_CAT_COLS = ["spectral_type", "galaxy_population"]
BASE_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
BANDS = ["u", "g", "r", "i", "z"]
MISSING = "__MISSING__"

COLOR_PAIRS = [
    ("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
    ("u", "r"), ("g", "i"), ("r", "z"),
]

IMPORTANT_COMBOS = sorted([
    ("alpha_cat_", "delta_cat_"),
    ("u_cat_", "z_cat_"),
])


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Pure row-wise stateless FE — safe to apply once on full df."""
    df = df.copy()
    # Redshift ratios
    df["_g_div_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    # Color pairs
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")
    # Magnitude aggregates
    mags = df[["u", "g", "r", "i", "z"]].astype("float32")
    df["_mag_mean"] = mags.mean(axis=1).astype("float32")
    df["_mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")
    # Log1p of shifted redshift
    shifted_rs = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted_rs).astype("float32")
    return df


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Fit categorical encodings on train-fold only, transform val and test.
    fit_in_fold — called INSIDE the fold loop.
    Returns (df_tr, df_val, df_te, cat_cols, combo_names, local_map).
    """
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    tr = df_tr.copy(); va = df_val.copy(); te = df_te.copy()

    # Original categorical columns
    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")

    # Integer-floor categorical views of every base numeric
    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        for dset, dset_src in [(va, df_val), (te, df_te)]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

    # Delta quantile bins — fit_in_fold via KBinsDiscretizer
    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32").astype("category")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32").astype("category")

    # Interaction cross-combos
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


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names: list, fold_seed: int):
    """TargetEncoder fit on train fold only (fit_in_fold)."""
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    try:
        encoder = TargetEncoder(target_type="multiclass", cv=5, smooth="auto",
                                shuffle=True, random_state=fold_seed)
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


# ─── qbin helpers (fit_in_fold) ───────────────────────────────────────────────
QBIN_PREFERRED = [
    "redshift", "_log1p_redshift", "_g_div_redshift", "_i_div_redshift",
    "u", "g", "r", "i", "z",
    "_u-g", "_g-r", "_r-i", "_i-z", "_u-r", "_g-i", "_r-z",
    "_mag_mean", "_mag_range",
    "alpha", "delta",
]


def choose_qbin_cols(num_cols: list, n: int) -> list:
    out = [c for c in QBIN_PREFERRED if c in num_cols]
    out += [c for c in num_cols if c not in out]
    return out[:n]


def fit_qbins(df: pd.DataFrame, qcols: list, n_bins: int):
    qs = np.linspace(0, 1, n_bins + 1, dtype="float32")[1:-1]
    edges_list = []
    cards = []
    for c in qcols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float32")
        v = v[np.isfinite(v)]
        e = np.unique(np.nanquantile(v, qs).astype("float32")) if len(v) else np.array([], dtype="float32")
        edges_list.append(e)
        cards.append(len(e) + 2)
    return edges_list, cards


def transform_qbins(df: pd.DataFrame, qcols: list, edges_list: list) -> np.ndarray:
    if not qcols:
        return np.zeros((len(df), 0), dtype="int32")
    out = np.zeros((len(df), len(qcols)), dtype="int32")
    for j, (c, e) in enumerate(zip(qcols, edges_list)):
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype="float32")
        good = np.isfinite(v)
        ids = np.zeros(len(v), dtype="int32")
        ids[good] = np.searchsorted(e, v[good], side="right").astype("int32") + 1
        out[:, j] = ids
    return out


# ─── Artifact-count features (fit_in_fold) ───────────────────────────────────
def _floor_cat(series):
    return np.floor(pd.to_numeric(series, errors="coerce")).astype("Int64").astype("string").fillna(MISSING).astype(str)


def add_artifact_features_fold(tr_df: pd.DataFrame, va_df: pd.DataFrame, te_df: pd.DataFrame):
    """
    Low-freq artifact features + count-frequency features, fit on train fold only.
    Returns (tr_df, va_df, te_df, new_num_cols).
    """
    new_num_cols = []

    # Floor cats for artifact detection
    art_cat_vals = {}  # name -> (tr_vals, va_vals, te_vals) as str Series
    for c in BASE_NUM_COLS:
        art_cat_vals[f"art_{c}_floor"] = (
            _floor_cat(tr_df[c]), _floor_cat(va_df[c]), _floor_cat(te_df[c])
        )
    if "alpha" in tr_df.columns and "delta" in tr_df.columns:
        art_cat_vals["art_alpha_floor_x_delta_floor"] = (
            _floor_cat(tr_df["alpha"]) + "__" + _floor_cat(tr_df["delta"]),
            _floor_cat(va_df["alpha"]) + "__" + _floor_cat(va_df["delta"]),
            _floor_cat(te_df["alpha"]) + "__" + _floor_cat(te_df["delta"]),
        )
    if "u" in tr_df.columns and "z" in tr_df.columns:
        art_cat_vals["art_u_floor_x_z_floor"] = (
            _floor_cat(tr_df["u"]) + "__" + _floor_cat(tr_df["z"]),
            _floor_cat(va_df["u"]) + "__" + _floor_cat(va_df["z"]),
            _floor_cat(te_df["u"]) + "__" + _floor_cat(te_df["z"]),
        )

    # Count frequency features — fit on train fold only
    for name, (tr_s, va_s, te_s) in art_cat_vals.items():
        # Count from train fold only
        vc = tr_s.value_counts()
        total = float(len(tr_s))
        for split_name, s_vals, df_ref in [("tr", tr_s, tr_df), ("va", va_s, va_df), ("te", te_s, te_df)]:
            pass  # we'll write below

        def make_count_features(s, vc, total, out_df):
            cnt = s.map(vc).fillna(0).astype("float32").to_numpy()
            out_df[f"art_count_log_{name}"] = np.log1p(cnt).astype("float32")
            out_df[f"art_freq_{name}"] = (cnt / total).astype("float32")

        make_count_features(tr_s, vc, total, tr_df)
        make_count_features(va_s, vc, total, va_df)
        make_count_features(te_s, vc, total, te_df)
        new_num_cols.extend([f"art_count_log_{name}", f"art_freq_{name}"])

    return tr_df, va_df, te_df, new_num_cols


# ─── Cat cardinality helper for DCN embeddings ────────────────────────────────
def fit_cat_maps(df: pd.DataFrame, cat_cols: list, min_count: int):
    """Fit category-to-int maps from df, used for DCN DenseInput embeddings."""
    maps = []
    cards = []
    for c in cat_cols:
        s = df[c].astype(str).fillna(MISSING)
        vc = s.value_counts()
        keep = vc[vc >= min_count].index
        mapping = {v: i + 1 for i, v in enumerate(keep)}  # 0 = unknown
        maps.append(mapping)
        cards.append(len(mapping) + 1)
    return maps, cards


def encode_cats(df: pd.DataFrame, cat_cols: list, maps: list) -> np.ndarray:
    if not cat_cols:
        return np.zeros((len(df), 0), dtype="int32")
    out = np.zeros((len(df), len(cat_cols)), dtype="int32")
    for j, (c, mapping) in enumerate(zip(cat_cols, maps)):
        out[:, j] = df[c].astype(str).fillna(MISSING).map(mapping).fillna(0).astype("int32").to_numpy()
    return out


# ─── DCN model architecture ───────────────────────────────────────────────────
def emb_dim(card: int, max_dim: int) -> int:
    return min(max_dim, max(2, int(round(card ** 0.25 * 3.5))))


class DenseInput(nn.Module):
    def __init__(self, n_num: int, cards: list, max_emb_dim: int, emb_dropout: float):
        super().__init__()
        self.embs = nn.ModuleList()
        emb_out = 0
        for card in cards:
            dim = emb_dim(card, max_emb_dim)
            emb = nn.Embedding(card, dim)
            nn.init.normal_(emb.weight, 0, 0.02)
            self.embs.append(emb)
            emb_out += dim
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.out_dim = n_num + emb_out

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if not self.embs:
            return x_num
        emb = torch.cat([e(x_cat[:, j]) for j, e in enumerate(self.embs)], dim=1)
        return torch.cat([x_num, self.emb_dropout(emb)], dim=1)


class CrossNet(nn.Module):
    def __init__(self, dim: int, layers: int):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(dim) * 0.01) for _ in range(layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for w, b in zip(self.weights, self.biases):
            xw = torch.sum(x * w, dim=1, keepdim=True)
            x = x0 * xw + b + x
        return x


class DCN(nn.Module):
    def __init__(self, n_num: int, cards: list, hidden: int, blocks: int,
                 cross_layers: int, dropout: float, emb_dropout: float,
                 max_emb_dim: int, n_classes: int = 3):
        super().__init__()
        self.input = DenseInput(n_num, cards, max_emb_dim, emb_dropout)
        self.norm = nn.LayerNorm(self.input.out_dim)
        self.cross = CrossNet(self.input.out_dim, cross_layers)
        layers = [nn.Linear(self.input.out_dim, hidden), nn.SiLU(), nn.Dropout(dropout)]
        for _ in range(blocks - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout)]
        self.deep = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(self.input.out_dim + hidden),
            nn.Linear(self.input.out_dim + hidden, n_classes)
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x0 = self.norm(self.input(x_num, x_cat))
        return self.head(torch.cat([self.cross(x0), self.deep(x0)], dim=1))


def build_model(n_num: int, cards: list) -> DCN:
    return DCN(
        n_num=n_num, cards=cards,
        hidden=CFG.hidden, blocks=CFG.blocks, cross_layers=CFG.cross_layers,
        dropout=CFG.dropout, emb_dropout=CFG.emb_dropout, max_emb_dim=CFG.max_emb_dim,
        n_classes=N_CLASSES,
    )


# ─── Training helpers ─────────────────────────────────────────────────────────
class TabDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.from_numpy(x_num.astype("float32", copy=False))
        self.x_cat = torch.from_numpy(x_cat.astype("int64", copy=False))
        self.y = torch.from_numpy(y.astype("int64", copy=False)) if y is not None else None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx], self.x_cat[idx]
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


def compute_loss(logits, yb, class_weights):
    """Per-class balanced CE with label smoothing."""
    loss = F.cross_entropy(logits, yb, label_smoothing=CFG.label_smoothing, reduction="none")
    parts = []
    for cls in range(N_CLASSES):
        mask = yb == cls
        if torch.any(mask):
            parts.append(loss[mask].mean())
    return torch.stack(parts).mean()


@torch.no_grad()
def predict_probs(model: DCN, loader: DataLoader) -> np.ndarray:
    model.eval()
    chunks = []
    amp_enabled = CFG.amp and DEVICE.type == "cuda"
    for batch in loader:
        xb_num = batch[0].to(DEVICE, non_blocking=True)
        xb_cat = batch[1].to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            logits = model(xb_num, xb_cat)
        chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.vstack(chunks).astype("float32")


def update_ema(model: DCN, ema_model: DCN, decay: float):
    m_state = model.state_dict()
    e_state = ema_model.state_dict()
    with torch.no_grad():
        for key, ev in e_state.items():
            mv = m_state[key].detach()
            if torch.is_floating_point(ev):
                ev.mul_(decay).add_(mv, alpha=1.0 - decay)
            else:
                ev.copy_(mv)


def train_fold(fold_id: int, Xn_tr: np.ndarray, Xc_tr: np.ndarray,
               y_tr: np.ndarray, Xn_va: np.ndarray, Xc_va: np.ndarray,
               Xn_te: np.ndarray, Xc_te: np.ndarray,
               cards: list, y_va: np.ndarray):
    seed_everything(CFG.seed + fold_id)

    ds_tr = TabDataset(Xn_tr, Xc_tr, y_tr)
    ds_va = TabDataset(Xn_va, Xc_va)
    ds_te = TabDataset(Xn_te, Xc_te)

    dl_tr = DataLoader(ds_tr, batch_size=CFG.micro_batch_size, shuffle=True,
                       drop_last=True, num_workers=0, pin_memory=CFG.pin_memory)
    dl_va = DataLoader(ds_va, batch_size=CFG.eval_batch_size, shuffle=False,
                       num_workers=0, pin_memory=CFG.pin_memory)
    dl_te = DataLoader(ds_te, batch_size=CFG.eval_batch_size, shuffle=False,
                       num_workers=0, pin_memory=CFG.pin_memory)

    model = build_model(Xn_tr.shape[1], cards).to(DEVICE)
    ema_model = build_model(Xn_tr.shape[1], cards).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Per-class balanced weights from train labels
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float32)
    cw = torch.tensor(counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.epochs, eta_min=CFG.lr * 0.02)
    gscaler = torch.amp.GradScaler("cuda", enabled=CFG.amp and DEVICE.type == "cuda")

    best_bac = -1.0
    best_state = None
    stale = 0
    grad_accum_steps = max(1, int(CFG.grad_accum_steps))
    amp_enabled = CFG.amp and DEVICE.type == "cuda"
    n_batches = len(dl_tr)

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl_tr, start=1):
            xb_num = batch[0].to(DEVICE, non_blocking=True)
            xb_cat = batch[1].to(DEVICE, non_blocking=True)
            yb = batch[2].to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                loss = compute_loss(model(xb_num, xb_cat), yb, cw) / grad_accum_steps
            gscaler.scale(loss).backward()
            if step % grad_accum_steps == 0 or step == n_batches:
                gscaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                gscaler.step(opt)
                gscaler.update()
                opt.zero_grad(set_to_none=True)
                update_ema(model, ema_model, CFG.ema_decay)
        scheduler.step()

        # Evaluate on val using EMA
        va_prob = predict_probs(ema_model, dl_va)
        bac = float(balanced_accuracy_score(y_va, va_prob.argmax(axis=1)))
        if bac > best_bac + 1e-7:
            best_bac = bac
            best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= CFG.patience:
                log(f"  fold={fold_id} early_stop epoch={epoch} best_BAC={best_bac:.8f}")
                break

        if epoch == 1 or epoch % CFG.report_every == 0:
            log(f"  fold={fold_id} epoch={epoch:03d} BAC={bac:.8f} best={best_bac:.8f}")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in ema_model.state_dict().items()}

    ema_model.load_state_dict(best_state)
    ema_model.to(DEVICE)
    va_prob = predict_probs(ema_model, dl_va)
    te_prob = predict_probs(ema_model, dl_te)
    fold_bac = float(balanced_accuracy_score(y_va, va_prob.argmax(axis=1)))

    del model, ema_model, opt, scheduler, gscaler, ds_tr, ds_va, ds_te, dl_tr, dl_va, dl_te, best_state
    cleanup_cuda()
    return va_prob, te_prob, fold_bac


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

if SMOKE:
    log("SMOKE MODE: 1 fold, 30k rows")
    rng_sm = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]

# ─── Stateless FE ─────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])
X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test={X_test_stateless.shape}")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype="float32")
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype="float32")
per_fold_scores = []
features_saved = False

log("Starting OOF fold loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    if SMOKE:
        keep_set = set(keep_sm.tolist())
        tr_idx = np.array([i for i in tr_idx if i in keep_set])
        val_idx = np.array([i for i in val_idx if i in keep_set])

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # --- FE fit_in_fold ---
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # Artifact count features — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, art_num_cols = add_artifact_features_fold(
        X_tr_fold, X_val_fold, X_te_fold
    )

    # Sort columns consistently
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    # Separate num and cat columns
    cat_cols_sorted = sorted([c for c in X_tr_fold.columns if str(X_tr_fold[c].dtype) == "category"])
    num_cols_sorted = [c for c in sorted(X_tr_fold.columns) if c not in cat_cols_sorted]

    # For DCN: all cat columns get embeddings; all num columns are standardized
    Xn_tr = X_tr_fold[num_cols_sorted].values.astype("float32")
    Xn_va = X_val_fold[num_cols_sorted].values.astype("float32")
    Xn_te = X_te_fold[num_cols_sorted].values.astype("float32")

    # Standardize numerical features — fit on train fold only
    scaler = StandardScaler()
    Xn_tr = scaler.fit_transform(Xn_tr).astype("float32")
    Xn_va = scaler.transform(Xn_va).astype("float32")
    Xn_te = scaler.transform(Xn_te).astype("float32")

    # Fit cat cardinality maps on train fold — for DCN DenseInput embeddings
    cat_maps, cat_cards = fit_cat_maps(X_tr_fold, cat_cols_sorted, CFG.rare_min_count)
    Xc_tr = encode_cats(X_tr_fold, cat_cols_sorted, cat_maps)
    Xc_va = encode_cats(X_val_fold, cat_cols_sorted, cat_maps)
    Xc_te = encode_cats(X_te_fold, cat_cols_sorted, cat_maps)

    # qbin embeddings — fit on train fold only
    qcols = choose_qbin_cols(num_cols_sorted, CFG.qbin_cols)
    # NOTE: qbin uses unstandardized data (from X_tr_fold before scaler), so re-extract
    qn_tr_raw = X_tr_fold[qcols].copy()
    qn_va_raw = X_val_fold[qcols].copy()
    qn_te_raw = X_te_fold[qcols].copy()
    qbin_cards_all = []
    qc_tr_parts = []
    qc_va_parts = []
    qc_te_parts = []
    for n_bins in CFG.qbin_levels:
        edges_list, level_cards = fit_qbins(qn_tr_raw, qcols, n_bins)
        qbin_cards_all.extend(level_cards)
        qc_tr_parts.append(transform_qbins(qn_tr_raw, qcols, edges_list))
        qc_va_parts.append(transform_qbins(qn_va_raw, qcols, edges_list))
        qc_te_parts.append(transform_qbins(qn_te_raw, qcols, edges_list))

    if qc_tr_parts:
        Xc_tr = np.hstack([Xc_tr] + qc_tr_parts).astype("int32")
        Xc_va = np.hstack([Xc_va] + qc_va_parts).astype("int32")
        Xc_te = np.hstack([Xc_te] + qc_te_parts).astype("int32")
        all_cards = cat_cards + qbin_cards_all
    else:
        all_cards = cat_cards

    if not features_saved:
        log(f"  n_num={len(num_cols_sorted)}  n_cat_emb={len(cat_cols_sorted)}  "
            f"n_qbin_emb={sum(len(p[0]) for p in [(qcols, qc_tr_parts)])}  total_cards={len(all_cards)}")
        features_saved = True

    # Timing one fold probe
    if fold_id == 0:
        t_fold_start = time.perf_counter()

    va_prob, te_prob, fold_bac = train_fold(
        fold_id, Xn_tr, Xc_tr, y_tr_fold, Xn_va, Xc_va, Xn_te, Xc_te, all_cards, y_val_fold
    )

    if fold_id == 0:
        fold0_time = time.perf_counter() - t_fold_start
        projected = fold0_time * len(folds_list)
        log(f"TIMING: fold0={fold0_time:.1f}s  projected_5fold={projected:.1f}s ({projected/60:.1f}min)")

    oof_proba[val_idx] = va_prob.astype("float32")
    test_proba_accum += te_prob.astype("float32") / len(folds_list)

    per_fold_scores.append(fold_bac)
    log(f"Fold {fold_id}: balanced_accuracy={fold_bac:.8f}")
    print(f"fold{fold_id}_score={fold_bac:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM: {vram_gb:.2f} GB")

    del X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te, Xc_tr, Xc_va, Xc_te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── OOF full metric ──────────────────────────────────────────────────────────
oof_metric = float(balanced_accuracy_score(y_all, oof_proba.argmax(1)))
log(f"OOF full balanced_accuracy={oof_metric:.8f}")

# ─── Save artifacts ───────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")

# ─── Write features.txt (all columns fed to DCN) ─────────────────────────────
# Re-compute on fold 0 for reference (stateless — same col names every fold)
_fi = folds_list[0]
_val_idx = np.asarray(_fi["val_idx"])
_tr_idx = np.setdiff1d(np.arange(n_train), _val_idx)
_X_tr, _X_va, _X_te, _cat_cols, _combo_names, _ = fit_fold_categoricals(
    X_stateless.iloc[_tr_idx].reset_index(drop=True),
    X_stateless.iloc[_val_idx].reset_index(drop=True),
    X_test_stateless.copy(),
)
_X_tr, _X_va, _X_te, _te_names = add_target_encoding(
    _X_tr, y_all[_tr_idx], _X_va, _X_te, _combo_names, SEED + 100
)
_X_tr, _X_va, _X_te, _art_cols = add_artifact_features_fold(_X_tr, _X_va, _X_te)
_all_feat_cols = sorted(_X_tr.columns)
(NODE_SRC / "features.txt").write_text("\n".join(_all_feat_cols) + "\n")
log(f"Wrote features.txt ({len(_all_feat_cols)} features)")

# ─── Standalone DE threshold scoring (fold-honest) ────────────────────────────
log("Computing standalone DE threshold balanced accuracy ...")

NC = N_CLASSES


def score_fn(y_true, y_pred) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))


def best_thr_de(probs, labels) -> np.ndarray:
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)],
                                maxiter=40, tol=1e-7, seed=0, polish=False, workers=1)
    return np.array([r.x[0], r.x[1], 1.0])


fval = [np.asarray(f["val_idx"]) for f in folds_list]
standalone_scores = []
for i, vi in enumerate(fval):
    other = np.setdiff1d(np.arange(n_train), vi)
    w = best_thr_de(oof_proba[other], y_all[other])
    pred = np.argmax(oof_proba[vi] * w, axis=1)
    s = score_fn(y_all[vi], pred)
    standalone_scores.append(s)
    print(f"standalone_fold{i}: score={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]", flush=True)

standalone_mean = float(np.mean(standalone_scores))
standalone_sem = float(np.std(standalone_scores, ddof=1) / np.sqrt(len(standalone_scores)))
log(f"STANDALONE DE-threshold BAC: {standalone_mean:.6f} +/- {standalone_sem:.6f}")
log(f"  (TabM node_0033 reference: 0.968053)")
print(f"standalone_cv={standalone_mean:.6f}", flush=True)

# ─── Re-stack A/B: CORE15 + n55 ──────────────────────────────────────────────
log("Re-stack A/B: CORE15 + node_0055 ...")

BASES = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030",
    "node_0039",
]

nodes_dir = COMP_DIR / "nodes"


def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))


def fit_meta(Xtr, ytr) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m


try:
    OOF_CORE = np.concatenate(
        [logp(np.load(nodes_dir / b / "oof.npy")) for b in BASES], axis=1
    )
    TEST_CORE = np.concatenate(
        [logp(np.load(nodes_dir / b / "test_probs.npy")) for b in BASES], axis=1
    )
    log(f"  CORE15 OOF={OOF_CORE.shape}  TEST={TEST_CORE.shape}")

    # Add n55 (this node) log-probs as 16th base
    n55_oof_logp = logp(oof_proba)    # (N_train, 3)
    n55_test_logp = logp(test_proba_accum)  # (N_test, 3)

    OOF_STACK = np.concatenate([OOF_CORE, n55_oof_logp], axis=1)
    TEST_STACK = np.concatenate([TEST_CORE, n55_test_logp], axis=1)
    log(f"  stacked OOF={OOF_STACK.shape}  TEST={TEST_STACK.shape}  (16 bases)")

    # Fold-honest stacked OOF
    stack_oof = np.zeros((n_train, NC))
    for vi in fval:
        tr_idx_s = np.setdiff1d(np.arange(n_train), vi)
        m = fit_meta(OOF_STACK[tr_idx_s], y_all[tr_idx_s])
        stack_oof[vi] = m.predict_proba(OOF_STACK[vi])

    restack_scores = []
    for i, vi in enumerate(fval):
        other = np.setdiff1d(np.arange(n_train), vi)
        w = best_thr_de(stack_oof[other], y_all[other])
        pred = np.argmax(stack_oof[vi] * w, axis=1)
        s = score_fn(y_all[vi], pred)
        restack_scores.append(s)
        print(f"restack_fold{i}: score={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]", flush=True)

    restack_mean = float(np.mean(restack_scores))
    restack_sem = float(np.std(restack_scores, ddof=1) / np.sqrt(len(restack_scores)))
    log(f"RE-STACK A/B (CORE15+n55): {restack_mean:.6f} +/- {restack_sem:.6f}")
    log(f"  (champion node_0041: 0.969808)")
    log(f"  delta vs champion: {restack_mean - 0.969808:+.6f}  (lift threshold ~+0.0003)")
    print(f"restack_cv={restack_mean:.6f}", flush=True)

except Exception as exc:
    log(f"WARNING: re-stack failed: {exc}")
    restack_mean = None

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
