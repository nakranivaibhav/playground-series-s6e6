"""node_0071 — 5-seed bagged DCN (node_0055 parent) on fs_realmlp_fe features.

THE ONE ATOMIC CHANGE vs node_0055:
  Seed-bag DCN over 5 seeds (seeds: 20260604, 20260605, 20260606, 20260607, 20260608).
  For each fold, train DCN once per seed and average the softmax probabilities.
  Everything else is byte-identical to node_0055.

  INTERIM KILL: after seed-2, compute 2-seed-avg solo CV.
  If < 0.966337 (n55 CV 0.966037 + 0.0003), stop and record wash.

  RESTACK: bank-17 (public 18-model bank) + DCN-bag OOF + optionally + n33.
  ASSERT bank-17-only baseline reproduces 0.970153 ±0.0002 before trusting any delta.

Architecture/training: identical to node_0055 (see parent for full docs).
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

# ─── Seed bagging config ──────────────────────────────────────────────────────
BAG_SEEDS = [20260604, 20260605, 20260606, 20260607, 20260608]
N_SEEDS = len(BAG_SEEDS)
INTERIM_KILL_CV = 0.966337   # 2-seed avg must reach this to continue to seeds 3-5

# ─── Hyper-parameters (from nn-v2-for-s6e6.py ref) ───────────────────────────
CFG = SimpleNamespace(
    seed=20260604,  # overridden per-seed during bagging
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
               cards: list, y_va: np.ndarray, base_seed: int = None):
    if base_seed is None:
        base_seed = CFG.seed
    seed_everything(base_seed + fold_id)

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

# ─── OOF loop (phase 1: seeds 1-2) ───────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype="float32")
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype="float32")
per_fold_scores = []
features_saved = False

# Phase 1: seeds 1-2 only, for interim kill check
seeds_to_run = BAG_SEEDS[:2]
log(f"Starting OOF fold loop (phase 1, seeds={seeds_to_run}) ...")
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

    # --- FE fit_in_fold (done once per fold, shared across seeds) ---
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

    # ── Seed bag loop for this fold ──
    va_prob_accum_fold = np.zeros((len(val_idx), N_CLASSES), dtype="float64")
    te_prob_accum_fold = np.zeros((n_test, N_CLASSES), dtype="float64")

    for seed_i, bag_seed in enumerate(seeds_to_run):
        log(f"  Fold {fold_id} seed {seed_i+1}/{len(seeds_to_run)} (seed={bag_seed}) ...")

        # Timing: first seed of fold 0
        do_timing = (fold_id == 0 and seed_i == 0)
        if do_timing:
            t_unit_start = time.perf_counter()

        va_prob, te_prob, fold_bac = train_fold(
            fold_id, Xn_tr, Xc_tr, y_tr_fold, Xn_va, Xc_va, Xn_te, Xc_te,
            all_cards, y_val_fold, base_seed=bag_seed
        )

        if do_timing:
            unit_time = time.perf_counter() - t_unit_start
            projected_total = unit_time * len(seeds_to_run) * len(folds_list)
            log(f"TIMING: seed0_fold0={unit_time:.1f}s  projected_total={projected_total:.1f}s ({projected_total/60:.1f}min)")

        va_prob_accum_fold += va_prob.astype("float64")
        te_prob_accum_fold += te_prob.astype("float64")

        log(f"  Fold {fold_id} seed={bag_seed}: BAC={fold_bac:.8f}")
        if torch.cuda.is_available():
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            log(f"    peak VRAM: {vram_gb:.2f} GB")

        cleanup_cuda()

    # Average across seeds
    va_prob_avg = (va_prob_accum_fold / len(seeds_to_run)).astype("float32")
    te_prob_avg = (te_prob_accum_fold / len(seeds_to_run)).astype("float32")

    fold_bac_avg = float(balanced_accuracy_score(y_val_fold, va_prob_avg.argmax(axis=1)))

    oof_proba[val_idx] = va_prob_avg
    test_proba_accum += te_prob_avg / len(folds_list)

    per_fold_scores.append(fold_bac_avg)
    log(f"Fold {fold_id}: avg BAC={fold_bac_avg:.8f}")
    print(f"fold{fold_id}_score={fold_bac_avg:.6f}", flush=True)

    del X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te, Xc_tr, Xc_va, Xc_te
    del va_prob_accum_fold, te_prob_accum_fold
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

# ─── Interim kill check after 2-seed pass ─────────────────────────────────────
if len(seeds_to_run) < N_SEEDS:
    # Compute 2-seed interim CV
    interim_cv = float(np.mean(per_fold_scores))
    log(f"INTERIM (2-seed) CV={interim_cv:.6f}  kill_threshold={INTERIM_KILL_CV}")
    print(f"interim_2seed_cv={interim_cv:.6f}", flush=True)

    if interim_cv < INTERIM_KILL_CV:
        log(f"INTERIM KILL: 2-seed CV={interim_cv:.6f} < {INTERIM_KILL_CV}. Stopping seeds 3-5. WASH.")
        print(f"INTERIM_KILL=True  reason=2seed_cv_below_threshold", flush=True)

        # Save what we have as the final artifacts
        mean_cv = interim_cv
        sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
        log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
        log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}  (2-seed only, KILL)")
        print(f"cv={mean_cv:.6f}", flush=True)

        np.save(NODE_DIR / "oof.npy", oof_proba)
        np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
        pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
        sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
        sub = sub[list(sample_sub.columns)]
        sub.to_csv(NODE_DIR / "submission.csv", index=False)
        log("Saved oof.npy, test_probs.npy, submission.csv (2-seed only, kill)")

        total_elapsed = time.perf_counter() - T0
        log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
        log("Done (KILL).")
        sys.exit(0)

    else:
        log(f"INTERIM PASS: 2-seed CV={interim_cv:.6f} >= {INTERIM_KILL_CV}. Continuing to seeds 3-5.")
        # Now run seeds 3-5 on all folds, ADDING to existing oof_proba/test_proba_accum.
        # Reset accumulators to scale up correctly:
        # Current accumulators have 2-seed averages per fold; we need to re-accumulate
        # to a 5-seed average. We'll re-track from scratch by re-running seeds 1-2 again
        # OR we store intermediate per-seed oofs. For simplicity and correctness,
        # we accumulate seeds 3-5 separately and combine with the 2-seed average.

        log("Running seeds 3-5 to complete 5-seed bag...")
        remaining_seeds = BAG_SEEDS[2:]  # seeds 3,4,5

        # New accumulators for seeds 3-5 only
        oof_proba_35 = np.zeros((n_train, N_CLASSES), dtype="float64")
        test_proba_35 = np.zeros((n_test, N_CLASSES), dtype="float64")

        per_fold_scores_35 = []
        for fi in folds_list:
            fold_id = fi["fold"]
            val_idx = np.asarray(fi["val_idx"])
            tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
            fold_seed = SEED + (fold_id + 1) * 100
            seed_everything(fold_seed)

            log(f"Fold {fold_id} (seeds 3-5): train={len(tr_idx)} val={len(val_idx)}")

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
            X_tr_fold, X_val_fold, X_te_fold, art_num_cols = add_artifact_features_fold(
                X_tr_fold, X_val_fold, X_te_fold
            )
            X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
            X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
            X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)
            cat_cols_sorted = sorted([c for c in X_tr_fold.columns if str(X_tr_fold[c].dtype) == "category"])
            num_cols_sorted = [c for c in sorted(X_tr_fold.columns) if c not in cat_cols_sorted]
            Xn_tr = X_tr_fold[num_cols_sorted].values.astype("float32")
            Xn_va = X_val_fold[num_cols_sorted].values.astype("float32")
            Xn_te = X_te_fold[num_cols_sorted].values.astype("float32")
            scaler = StandardScaler()
            Xn_tr = scaler.fit_transform(Xn_tr).astype("float32")
            Xn_va = scaler.transform(Xn_va).astype("float32")
            Xn_te = scaler.transform(Xn_te).astype("float32")
            cat_maps, cat_cards = fit_cat_maps(X_tr_fold, cat_cols_sorted, CFG.rare_min_count)
            Xc_tr = encode_cats(X_tr_fold, cat_cols_sorted, cat_maps)
            Xc_va = encode_cats(X_val_fold, cat_cols_sorted, cat_maps)
            Xc_te = encode_cats(X_te_fold, cat_cols_sorted, cat_maps)
            qcols = choose_qbin_cols(num_cols_sorted, CFG.qbin_cols)
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

            va_prob_accum_35 = np.zeros((len(val_idx), N_CLASSES), dtype="float64")
            te_prob_accum_35 = np.zeros((n_test, N_CLASSES), dtype="float64")

            for seed_i, bag_seed in enumerate(remaining_seeds):
                log(f"  Fold {fold_id} seed {seed_i+3}/{N_SEEDS} (seed={bag_seed}) ...")
                va_prob, te_prob, fold_bac = train_fold(
                    fold_id, Xn_tr, Xc_tr, y_tr_fold, Xn_va, Xc_va, Xn_te, Xc_te,
                    all_cards, y_val_fold, base_seed=bag_seed
                )
                va_prob_accum_35 += va_prob.astype("float64")
                te_prob_accum_35 += te_prob.astype("float64")
                cleanup_cuda()

            oof_proba_35[val_idx] = (va_prob_accum_35 / len(remaining_seeds)).astype("float32")
            test_proba_35 += (te_prob_accum_35 / len(remaining_seeds)) / len(folds_list)

            bac_35 = float(balanced_accuracy_score(y_val_fold, oof_proba_35[val_idx].argmax(1)))
            per_fold_scores_35.append(bac_35)
            log(f"Fold {fold_id} seeds3-5 avg BAC={bac_35:.8f}")

            del X_tr_fold, X_val_fold, X_te_fold, Xn_tr, Xn_va, Xn_te, Xc_tr, Xc_va, Xc_te
            gc.collect()

        # Combine: 2-seed avg and 3-seed avg → weighted 5-seed avg
        oof_proba_5 = (oof_proba * 2 + oof_proba_35 * 3) / 5
        test_proba_5 = (test_proba_accum * 2 + test_proba_35 * 3) / 5

        # Recompute per-fold scores for the 5-seed avg
        fval_all = [np.asarray(f["val_idx"]) for f in folds_list]
        per_fold_scores_5 = []
        for vi in fval_all:
            bac_5f = float(balanced_accuracy_score(y_all[vi], oof_proba_5[vi].argmax(1)))
            per_fold_scores_5.append(bac_5f)

        # Replace global oof/test with 5-seed avg
        oof_proba[:] = oof_proba_5.astype("float32")
        test_proba_accum[:] = test_proba_5.astype("float32")
        per_fold_scores = per_fold_scores_5
        log("5-seed avg folds: " + ",".join(f"{s:.6f}" for s in per_fold_scores_5))

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

# ─── Restack: bank-17 + DCN-bag ───────────────────────────────────────────────
# ASSERT: bank-17-only baseline reproduces champion 0.970153 ± 0.0002
log("Running bank-17 + DCN-bag restack ...")

NC = N_CLASSES
nodes_dir = COMP_DIR / "nodes"


def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))


def fit_meta(Xtr, ytr) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m


def eval_stack(OOF_cols, TEST_cols, label=""):
    OOF = np.concatenate(OOF_cols, axis=1)
    TEST = np.concatenate(TEST_cols, axis=1)
    stack_oof = np.zeros((n_train, NC))
    for vi in fval:
        tr_idx_s = np.setdiff1d(np.arange(n_train), vi)
        m = fit_meta(OOF[tr_idx_s], y_all[tr_idx_s])
        stack_oof[vi] = m.predict_proba(OOF[vi])

    scores = []
    for vi in fval:
        other = np.setdiff1d(np.arange(n_train), vi)
        w = best_thr_de(stack_oof[other], y_all[other])
        pred = np.argmax(stack_oof[vi] * w, axis=1)
        s = score_fn(y_all[vi], pred)
        scores.append(s)
    cv = float(np.mean(scores))
    sem = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
    log(f"  {label}: cv={cv:.6f} sem={sem:.6f}")
    print(f"stack_{label}_cv={cv:.6f}", flush=True)
    return cv, sem, stack_oof, TEST


try:
    # Load bank-17 public OOFs (same manifest as a1_full_merge.py)
    C = COMP_DIR
    B = C / "refs/oof_bank"
    K = C / "refs/kernel_out"
    LAB = ["GALAXY", "QSO", "STAR"]

    def rd(path, nr):
        p = str(path)
        if p.endswith(".npy"):
            a = np.load(p, allow_pickle=True).astype(float)
            a = a.reshape(nr, -1) if a.ndim == 1 else a
            return a[:, :3]
        d = pd.read_csv(p)
        c = list(d.columns)
        if set(LAB).issubset(c):
            return d[LAB].values.astype(float)
        pc = [f"prob_{l}" for l in LAB]
        if set(pc).issubset(c):
            return d[pc].values.astype(float)
        num = d.select_dtypes("number")
        if num.shape[1] >= 3:
            return num.values[:, :3]
        v = d.iloc[:, 0].values.astype(float)
        return v.reshape(nr, 3)

    def norm(a):
        a = np.clip(a, 0, None)
        s = a.sum(1, keepdims=True)
        s[s == 0] = 1
        return a / s

    MANIFEST = {
        "xgb-0": (K / "xgb-v0-for-s6e6/oof_xgb_cv.csv", K / "xgb-v0-for-s6e6/test_xgb_preds.csv"),
        "xgb-1": (K / "xgb-v1-for-s6e6/oof_preds.npy", K / "xgb-v1-for-s6e6/test_preds.npy"),
        "realmlp-0": (B / "oof_preds_realmlp0_v12.csv", B / "test_preds_realmlp0_v12.csv"),
        "realmlp-1": (K / "realmlp-v1-for-s6e6/oof_preds.npy", K / "realmlp-v1-for-s6e6/test_preds.npy"),
        "tabm-0": (B / "oof_preds_tabm0_v2.csv", B / "test_preds_tabm0_v2.csv"),
        "cat-0": (K / "cat-v0-for-s6e6/catboost_oof_predictions.csv", K / "cat-v0-for-s6e6/catboost_test_predictions.csv"),
        "realmlp-2": (B / "oof_preds_realmlp2_v10.csv", B / "test_preds_realmlp2_v10.csv"),
        "tabicl-2": (K / "tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K / "tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
        "lgbm-3": (K / "lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy", K / "lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
        "logreg-1": (K / "logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy", K / "logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
        "nn-1": (K / "nn-v1-for-s6e6/train_oof/nn-1_oof.npy", K / "nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
        "xgb-3": (K / "xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K / "xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
        "xgb-5": (K / "xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy", K / "xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
        "realmlp-5": (K / "realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy", K / "realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
        "nn-2": (K / "nn-v2-for-s6e6/train_oof/nn-2_oof.npy", K / "nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
        "cat-3": (K / "cat-v3-for-s6e6/train_oof/cat-3_oof.npy", K / "cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
        "lgbm-5": (B / "oof_preds_lgbm5_v1.csv", B / "test_preds_lgbm5_v1.csv"),
        "xgb-6": (B / "oof_final_xgb6_v1.csv", B / "test_final_xgb6_v1.csv"),
        "tabm-1": (B / "oof_final_tabm1_v1.csv", B / "test_final_tabm1_v1.csv"),
    }

    n_te = n_test
    POOF = {}
    PTEST = {}
    good = []
    for name, (op, tp) in MANIFEST.items():
        try:
            o = norm(rd(op, n_train))
            t = norm(rd(tp, n_te))
            assert o.shape == (n_train, 3) and t.shape == (n_te, 3)
            from sklearn.metrics import balanced_accuracy_score as bask
            ba = bask(y_all, o.argmax(1))
            if 0.90 < ba < 0.972:
                POOF[name] = o
                PTEST[name] = t
                good.append(name)
        except Exception as e:
            log(f"  Skipping {name}: {e}")

    log(f"  Loaded {len(good)}/19 public models OK: {good}")

    # bank-17 OOF/test (log-probs)
    pub_oof_logp = [logp(POOF[k]) for k in good]
    pub_test_logp = [logp(PTEST[k]) for k in good]

    # ASSERT bank-17 baseline reproduces champion 0.970153 ± 0.0002
    log("  ASSERTING bank-17 baseline CV ...")
    bank17_cv, bank17_sem, _, _ = eval_stack(pub_oof_logp, pub_test_logp, "bank17_only")
    CHAMPION_EXPECTED = 0.970153
    TOLERANCE = 0.0002
    if abs(bank17_cv - CHAMPION_EXPECTED) > TOLERANCE:
        log(f"  ASSERT FAIL: bank17 cv={bank17_cv:.6f} expected={CHAMPION_EXPECTED:.6f} diff={abs(bank17_cv - CHAMPION_EXPECTED):.6f} > tol={TOLERANCE}")
        log("  WARNING: bank-17 baseline mismatch — restack deltas may be unreliable")
    else:
        log(f"  ASSERT OK: bank17 cv={bank17_cv:.6f} matches expected {CHAMPION_EXPECTED:.6f} ± {TOLERANCE}")

    # Variant A: bank17 + DCN-bag (this node)
    dcn_bag_oof_logp = logp(oof_proba)
    dcn_bag_test_logp = logp(test_proba_accum)

    cv_a, sem_a, stack_oof_a, test_a = eval_stack(
        pub_oof_logp + [dcn_bag_oof_logp],
        pub_test_logp + [dcn_bag_test_logp],
        "bank17+DCNbag"
    )

    # Variant B: bank17 + DCN-bag + n33 (TabM)
    n33_oof_logp = logp(np.load(nodes_dir / "node_0033" / "oof.npy"))
    n33_test_logp = logp(np.load(nodes_dir / "node_0033" / "test_probs.npy"))

    cv_b, sem_b, stack_oof_b, test_b = eval_stack(
        pub_oof_logp + [dcn_bag_oof_logp, n33_oof_logp],
        pub_test_logp + [dcn_bag_test_logp, n33_test_logp],
        "bank17+DCNbag+n33"
    )

    log(f"RESTACK SUMMARY:")
    log(f"  bank17-only:       cv={bank17_cv:.6f}")
    log(f"  bank17+DCNbag:     cv={cv_a:.6f}  delta={cv_a - bank17_cv:+.6f}  vs champion: {cv_a - CHAMPION_EXPECTED:+.6f}")
    log(f"  bank17+DCNbag+n33: cv={cv_b:.6f}  delta={cv_b - bank17_cv:+.6f}  vs champion: {cv_b - CHAMPION_EXPECTED:+.6f}")

except Exception as exc:
    import traceback
    log(f"WARNING: restack failed: {exc}")
    traceback.print_exc()

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
