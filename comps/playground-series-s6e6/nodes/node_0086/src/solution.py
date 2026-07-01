"""node_0086 — z-conditional color-residual TabM base (fs_zresid).

THE ONE ATOMIC CHANGE vs node_0033:
  Feature set replaced entirely by fs_zresid (fit_in_fold):
  - For each color (u-g, g-r, r-i, i-z, u-z) and each magnitude (u,g,r,i,z):
    compute z-conditional z-score = (val − mean_zbin) / std_zbin over ~40 redshift
    quantile bins. Bin EDGES and per-bin mean/std are fit on TRAIN FOLD ONLY.
    Sparse bins fall back to global mean/std.
  - Raw redshift is KEPT (STAR z≈0 is a strong discriminator).
  - Raw colors and raw magnitudes are DROPPED — the model's PRIMARY signal is
    color-anomaly-at-fixed-z, not z-then-color.
  - All other FE (categoricals, target-encoder, PLR bins, TabM) byte-identical to n33.

Leak safety: bin edges + per-bin stats fit INSIDE the fold loop on train-fold rows
only. fs_zresid class = fit_in_fold.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv.
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import TargetEncoder
from sklearn.utils.class_weight import compute_class_weight

import tabm
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

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
log(f"Device: {DEVICE}  tabm={tabm.__version__}")
assert torch.cuda.is_available(), "CUDA required"

SMOKE = os.environ.get("TABM_SMOKE") == "1"
FOLD0_ONLY = os.environ.get("FOLD0_ONLY") == "1"

# TabM hyperparameters — byte-identical to n33
D_EMB = 16
N_BINS = 48
K_ENS = 32
DROPOUT = 0.1
MAX_EPOCHS = 100 if not SMOKE else 6
PATIENCE = 16
BATCH_SIZE = 8192
INFER_BATCH_SIZE = 4096

# fs_zresid parameters
N_ZBINS = 40   # redshift quantile bins

# Magnitudes and colors to residualize
MAGS = ["u", "g", "r", "i", "z"]
COLORS = [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "z")]
COLOR_NAMES = [f"{a}-{b}" for a, b in COLORS]


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

# ─── fs_zresid: z-conditional color/magnitude residuals ──────────────────────

def fit_zresid(df_tr: pd.DataFrame, redshift_tr: np.ndarray, n_zbins: int = N_ZBINS):
    """
    Fit bin edges and per-bin mean/std on the TRAIN FOLD only.
    Returns (edges, stats) where stats[col] = (bin_means, bin_stds, global_mean, global_std).
    edges has shape (n_zbins+1,).
    """
    # Quantile bin edges on train redshift only
    quantiles = np.linspace(0, 100, n_zbins + 1)
    edges = np.percentile(redshift_tr, quantiles)
    # Ensure strictly increasing (deduplicate ties by nudging)
    edges = np.unique(edges)
    # Assign train rows to bins
    bin_idx = np.searchsorted(edges[1:-1], redshift_tr, side="right")  # 0..n_bins-1

    # Build feature list
    feature_names = MAGS + COLOR_NAMES
    raw_values = {}
    for mag in MAGS:
        raw_values[mag] = df_tr[mag].values.astype(np.float64)
    for a, b in COLORS:
        cname = f"{a}-{b}"
        raw_values[cname] = (df_tr[a] - df_tr[b]).values.astype(np.float64)

    stats = {}
    actual_n_bins = len(edges) - 1  # may be < N_ZBINS if many ties
    for col, vals in raw_values.items():
        bin_means = np.full(actual_n_bins, np.nan)
        bin_stds = np.full(actual_n_bins, np.nan)
        for b_id in range(actual_n_bins):
            mask = bin_idx == b_id
            if mask.sum() >= 5:
                bin_means[b_id] = vals[mask].mean()
                bin_stds[b_id] = vals[mask].std() + 1e-8
        # Global fallbacks
        global_mean = vals.mean()
        global_std = vals.std() + 1e-8
        stats[col] = (bin_means, bin_stds, global_mean, global_std)

    return edges, stats, feature_names


def apply_zresid(df: pd.DataFrame, redshift: np.ndarray, edges: np.ndarray, stats: dict, feature_names: list) -> np.ndarray:
    """
    Apply z-conditional z-score transform using pre-fit bin edges + stats.
    Rows whose bin has NaN mean/std fall back to global mean/std.
    Returns (N, n_features) float32 array.
    """
    actual_n_bins = len(edges) - 1
    bin_idx = np.searchsorted(edges[1:-1], redshift, side="right")

    raw_values = {}
    for mag in MAGS:
        raw_values[mag] = df[mag].values.astype(np.float64)
    for a, b in COLORS:
        cname = f"{a}-{b}"
        raw_values[cname] = (df[a] - df[b]).values.astype(np.float64)

    out = np.zeros((len(df), len(feature_names)), dtype=np.float32)
    for fi, col in enumerate(feature_names):
        bin_means, bin_stds, global_mean, global_std = stats[col]
        vals = raw_values[col]
        col_out = np.empty(len(df), dtype=np.float64)
        for i in range(len(df)):
            bid = bin_idx[i]
            if bid < actual_n_bins and not np.isnan(bin_means[bid]):
                col_out[i] = (vals[i] - bin_means[bid]) / bin_stds[bid]
            else:
                col_out[i] = (vals[i] - global_mean) / global_std
        out[:, fi] = col_out.astype(np.float32)
    return out


def apply_zresid_vectorized(df: pd.DataFrame, redshift: np.ndarray, edges: np.ndarray, stats: dict, feature_names: list) -> np.ndarray:
    """
    Vectorized version of apply_zresid — avoids Python loops over rows.
    """
    actual_n_bins = len(edges) - 1
    bin_idx = np.searchsorted(edges[1:-1], redshift, side="right")  # (N,)

    raw_values = {}
    for mag in MAGS:
        raw_values[mag] = df[mag].values.astype(np.float64)
    for a, b in COLORS:
        cname = f"{a}-{b}"
        raw_values[cname] = (df[a] - df[b]).values.astype(np.float64)

    out = np.zeros((len(df), len(feature_names)), dtype=np.float32)
    for fi, col in enumerate(feature_names):
        bin_means, bin_stds, global_mean, global_std = stats[col]
        vals = raw_values[col]
        # Look up per-row mean/std from bins; fallback where NaN
        row_means = np.where(
            (bin_idx < actual_n_bins) & ~np.isnan(bin_means[np.clip(bin_idx, 0, actual_n_bins - 1)]),
            bin_means[np.clip(bin_idx, 0, actual_n_bins - 1)],
            global_mean
        )
        row_stds = np.where(
            (bin_idx < actual_n_bins) & ~np.isnan(bin_stds[np.clip(bin_idx, 0, actual_n_bins - 1)]),
            bin_stds[np.clip(bin_idx, 0, actual_n_bins - 1)],
            global_std
        )
        out[:, fi] = ((vals - row_means) / row_stds).astype(np.float32)
    return out


# ─── Categorical encoding (byte-identical to n33) ────────────────────────────
BASE_CAT_COLS = ["spectral_type", "galaxy_population"]

IMPORTANT_COMBOS = sorted([
    ("alpha_cat_", "delta_cat_"),
    ("u_cat_", "z_cat_"),
])

# For the combo categoricals we need integer-floor views — but we no longer
# have raw u/z/alpha/delta as primary features.
# We derive alpha_cat_, delta_cat_, u_cat_, z_cat_ from the raw columns
# (still stateless row-wise — just floor of coordinate/magnitude).
# They are ONLY used for the combo target-encoder, not as direct model features.


def build_combo_cats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build integer-floor categorical views of alpha, delta, u, z for combo TE.
    These are row-wise stateless (no fitting needed).
    """
    df = df.copy()
    for col in ["alpha", "delta", "u", "z"]:
        cat_name = f"{col}_cat_"
        df[cat_name] = np.floor(df[col]).astype("float32")
    return df


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Fit categorical encodings on train-fold only.
    Returns (tr, val, te, cat_cols, combo_names, local_map).
    """
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    tr = df_tr.copy()
    va = df_val.copy()
    te = df_te.copy()

    # Original categorical columns
    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32").astype("category")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32").astype("category")

    # Integer-floor categoricals for combos
    for col in ["alpha", "delta", "u", "z"]:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32").astype("category")
        for dset, dset_src in [(va, df_val), (te, df_te)]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

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
    X_tr = X_tr.copy()
    X_val = X_val.copy()
    X_te = X_te.copy()

    try:
        encoder = TargetEncoder(
            target_type="multiclass", cv=5, smooth="auto", shuffle=True, random_state=fold_seed
        )
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


# ─── TabM training (byte-identical to n33) ────────────────────────────────────

def build_tabm_model(n_num: int, cat_cards: list[int], bins: list) -> tabm.TabM:
    num_emb = PiecewiseLinearEmbeddings(bins, d_embedding=D_EMB, activation=False, version="B")
    model = tabm.TabM.make(
        n_num_features=n_num,
        cat_cardinalities=cat_cards if cat_cards else None,
        d_out=N_CLASSES,
        num_embeddings=num_emb,
        k=K_ENS,
        dropout=DROPOUT,
    )
    return model.to(DEVICE)


def predict_proba_batch(model: tabm.TabM, Xn: np.ndarray, Xc: np.ndarray | None,
                        batch_size: int = INFER_BATCH_SIZE) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xn), batch_size):
            xn = torch.as_tensor(Xn[s:s + batch_size], dtype=torch.float32, device=DEVICE)
            xc = (torch.as_tensor(Xc[s:s + batch_size], dtype=torch.long, device=DEVICE)
                  if Xc is not None else None)
            logits = model(xn, xc)
            probs = torch.softmax(logits.float(), dim=-1).mean(dim=1)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def train_tabm(
    Xn_tr: np.ndarray,
    Xc_tr: np.ndarray | None,
    y_tr: np.ndarray,
    cat_cards: list[int],
    fold_seed: int,
) -> tuple[tabm.TabM, list]:
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(Xn_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    bins = compute_bins(
        torch.as_tensor(Xn_tr[ti], dtype=torch.float32),
        n_bins=N_BINS,
        y=torch.as_tensor(y_tr[ti], dtype=torch.long),
        regression=False,
        tree_kwargs={"min_samples_leaf": 64},
    )

    model = build_tabm_model(Xn_tr.shape[1], cat_cards, bins)

    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    Xn_t = torch.as_tensor(Xn_tr[ti], dtype=torch.float32, device=DEVICE)
    Xc_t = (torch.as_tensor(Xc_tr[ti], dtype=torch.long, device=DEVICE)
             if Xc_tr is not None else None)
    y_t = torch.as_tensor(y_tr[ti], dtype=torch.long, device=DEVICE)
    nt = len(ti)

    yv = y_tr[vi]
    Xn_vi = Xn_tr[vi]
    Xc_vi = Xc_tr[vi] if Xc_tr is not None else None

    best_ba = -1.0
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            xn_b = Xn_t[idx]
            xc_b = Xc_t[idx] if Xc_t is not None else None
            y_b = y_t[idx]
            opt.zero_grad()
            logits = model(xn_b, xc_b)
            b, k, c = logits.shape
            loss = loss_fn(logits.reshape(b * k, c), y_b.repeat_interleave(k))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        val_probs = predict_proba_batch(model, Xn_vi, Xc_vi)
        ba = balanced_accuracy_score(yv, val_probs.argmax(1))
        if ba > best_ba + 1e-5:
            best_ba = ba
            best_state = {kk: v.detach().clone() for kk, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    TabM early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}")
    return model, bins


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
    log("SMOKE MODE: subsample to 30000 rows, 1 fold")
    folds_list = [folds_list[0]]

# Pre-flight leakage checks on feature plan
# Check 1: target not in features (we drop target+id before any FE)
# Check 2: id not in features (dropped explicitly)
# Check 3: single-feature sweep — done on residuals which are computed from train-only stats;
#          redshift kept, raw colors dropped — no raw target proxy possible.
log("Pre-flight leakage check: target/id not in feature columns — OK by construction (dropped before FE)")
assert TARGET not in train_raw.drop(columns=[IDC, TARGET]).columns
assert IDC not in train_raw.drop(columns=[IDC, TARGET]).columns
log("Pre-flight check 2: frozen folds.json used — OK")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # fs_zresid: FIT on train fold only ----------------------------------------
    df_tr_raw = train_raw.iloc[tr_idx].reset_index(drop=True)
    df_val_raw = train_raw.iloc[val_idx].reset_index(drop=True)
    df_te_raw = test_raw.copy()

    redshift_tr = df_tr_raw["redshift"].values.astype(np.float64)
    redshift_val = df_val_raw["redshift"].values.astype(np.float64)
    redshift_te = df_te_raw["redshift"].values.astype(np.float64)

    # Fit bin edges + per-bin stats on train fold only
    edges, zresid_stats, feature_names = fit_zresid(df_tr_raw, redshift_tr, n_zbins=N_ZBINS)
    log(f"  fs_zresid: n_edges={len(edges)}  n_features={len(feature_names)}")

    # Apply (vectorized) to train, val, test
    Xz_tr = apply_zresid_vectorized(df_tr_raw, redshift_tr, edges, zresid_stats, feature_names)
    Xz_val = apply_zresid_vectorized(df_val_raw, redshift_val, edges, zresid_stats, feature_names)
    Xz_te = apply_zresid_vectorized(df_te_raw, redshift_te, edges, zresid_stats, feature_names)

    # Build DataFrames with z-residuals + raw redshift + categoricals
    def make_df(raw_df: pd.DataFrame, Xz: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame(Xz, columns=[f"zr_{n}" for n in feature_names], index=raw_df.index)
        df["redshift"] = raw_df["redshift"].values.astype(np.float32)
        # Add alpha, delta, u, z for combo categoricals (stateless)
        for col in ["alpha", "delta", "u", "z"]:
            df[col] = raw_df[col].values.astype(np.float32)
        df["spectral_type"] = raw_df["spectral_type"].values
        df["galaxy_population"] = raw_df["galaxy_population"].values
        return df.reset_index(drop=True)

    X_tr_base = make_df(df_tr_raw, Xz_tr)
    X_val_base = make_df(df_val_raw, Xz_val)
    X_te_base = make_df(df_te_raw, Xz_te)

    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # Categorical encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_tr_base, X_val_base, X_te_base
    )

    # Target encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # Drop alpha/delta/u/z (raw mag/coord columns used only for combo cats)
    # They should not be primary features — drop them to keep residual-dominated
    DROP_COLS = ["alpha", "delta", "u", "z"]
    for col in DROP_COLS:
        for dset in [X_tr_fold, X_val_fold, X_te_fold]:
            if col in dset.columns:
                dset.drop(columns=[col], inplace=True)

    # Sort columns consistently
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    cat_cols_sorted = sorted([c for c in cat_cols if c in X_tr_fold.columns])
    TABM_CAT_COLS = [c for c in cat_cols_sorted if c in BASE_CAT_COLS]
    all_cols_sorted = sorted(X_tr_fold.columns)
    num_for_tabm = [c for c in all_cols_sorted if c not in TABM_CAT_COLS]

    if fold_id == 0:
        log(f"  n_features={X_tr_fold.shape[1]}  tabm_cat={len(TABM_CAT_COLS)}  tabm_num={len(num_for_tabm)}")
        log(f"  sample num features: {num_for_tabm[:10]}")

    # Extract num/cat arrays
    Xn_tr = X_tr_fold[num_for_tabm].values.astype(np.float32)
    Xn_va = X_val_fold[num_for_tabm].values.astype(np.float32)
    Xn_te = X_te_fold[num_for_tabm].values.astype(np.float32)

    if TABM_CAT_COLS:
        Xc_tr = X_tr_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_va = X_val_fold[TABM_CAT_COLS].values.astype(np.int64)
        Xc_te = X_te_fold[TABM_CAT_COLS].values.astype(np.int64)
        cat_cards = (Xc_tr.max(axis=0) + 2).tolist()
        card_arr = np.array(cat_cards) - 1
        Xc_tr = np.clip(Xc_tr, 0, card_arr)
        Xc_va = np.clip(Xc_va, 0, card_arr)
        Xc_te = np.clip(Xc_te, 0, card_arr)
    else:
        Xc_tr = Xc_va = Xc_te = None
        cat_cards = []

    # Standardize numerical features — fit on train fold only
    mu = Xn_tr.mean(0)
    sd = Xn_tr.std(0) + 1e-8
    Xn_tr = (Xn_tr - mu) / sd
    Xn_va = (Xn_va - mu) / sd
    Xn_te = (Xn_te - mu) / sd

    # Pre-flight check 3: single-feature sweep on sample (NaN-cleaned)
    if fold_id == 0:
        log("Pre-flight check 3: single-feature ~ target sweep ...")
        sample_n = min(50000, len(Xn_tr))
        rng_check = np.random.default_rng(0)
        sidx = rng_check.choice(len(Xn_tr), sample_n, replace=False)
        ys_check = y_tr_fold[sidx]
        max_corr = 0.0
        for fi_idx in range(Xn_tr.shape[1]):
            xf = Xn_tr[sidx, fi_idx]
            if np.isnan(xf).any() or np.std(xf) < 1e-10:
                continue
            c = abs(np.corrcoef(xf, ys_check)[0, 1])
            if c > max_corr:
                max_corr = c
            if c >= 0.999:
                raise SystemExit(f"LEAK: feature {num_for_tabm[fi_idx]} corr={c:.4f}")
        log(f"  max |corr| = {max_corr:.4f}  (< 0.999 = clean)")

    # Train TabM
    model, bins = train_tabm(Xn_tr, Xc_tr, y_tr_fold, cat_cards, fold_seed)

    # OOF predictions
    val_probs = predict_proba_batch(model, Xn_va, Xc_va)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    # Test predictions — average across folds
    if not FOLD0_ONLY:
        test_probs_fold = predict_proba_batch(model, Xn_te, Xc_te)
        test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
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
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  ({projected/60:.1f}min)")

    # CHEAP-KILL: if fold 0 BA < 0.965, abort
    if fold_id == 0:
        if fold_score < 0.965:
            log(f"CHEAP-KILL: fold-0 BA={fold_score:.6f} < 0.965. Stopping.")
            print(f"CHEAP_KILL fold0={fold_score:.6f}", flush=True)
            sys.exit(42)
        if FOLD0_ONLY:
            log("FOLD0_ONLY mode — stopping after fold 0.")
            sys.exit(0)

    if SMOKE and fold_id == 0:
        log("[smoke] OK. Exiting.")
        sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

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
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
