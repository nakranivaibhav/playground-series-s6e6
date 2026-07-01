"""node_0056 — Two-Branch 1D-CNN spectral NN over wavelength-ordered photometric bands.

THE ONE ATOMIC CHANGE vs node_0033 (TabM):
  Replace TabM with a TWO-BRANCH network:
  - Branch A (1D-CNN): raw [u,g,r,i,z] bands in wavelength order as a length-5 signal
    with 3 input channels: (1) raw magnitude, (2) mean-subtracted SED shape,
    (3) z-normalized magnitude. Two Conv1d layers (kernel 2/3, padding same,
    32->64 channels, SiLU) -> global avg+max pool -> conv feature vector.
  - Branch B (scalar MLP): redshift + alpha/delta + all fs_realmlp_fe engineered
    features (colors, TE, cat codes) -> 2-layer SiLU MLP with dropout.
  - Concat (A + B) -> MLP head (256, dropout 0.1) -> 3-class softmax.
  Class-balanced cross-entropy, AdamW lr=1e-3, cosine schedule, early stopping.
  All scaling/normalization fit-in-fold (train-fold only).

FE pipeline: byte-identical to node_0033/node_0028.

Leakage discipline:
  - Stateless FE: no target, no cross-row stats, no fitting — safe to compute once.
  - KBinsDiscretizer, TargetEncoder: fit on train-fold rows only.
  - Standardization (mean/std): fit on train-fold numerical features only.
  - Band normalization (channels): fit on train-fold band rows only.
  - Frozen folds.json used throughout; no refitting of folds.

Outputs:
  oof.npy (577347, 3), test_probs.npy (247435, 3), submission.csv, features.txt.
  Also performs re-stack: CORE15 + node_0056 log-probs (16 bases), balanced LogReg
  meta + DE threshold, printed as restack_cv=...
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
import torch.nn.functional as F
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
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

# Wavelength-ordered bands (ascending wavelength: u~355nm, g~469nm, r~617nm, i~748nm, z~893nm)
BAND_COLS = ["u", "g", "r", "i", "z"]
N_BANDS = len(BAND_COLS)
N_CHANNELS = 3   # raw, mean-subtracted, z-normalized

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Device: {DEVICE}")

# Verify GPU is doing real matmul
if torch.cuda.is_available():
    _a = torch.randn(512, 512, device=DEVICE)
    _b = torch.randn(512, 512, device=DEVICE)
    _c = _a @ _b
    log(f"GPU matmul check OK: {_c.shape} on {DEVICE}")
    del _a, _b, _c

SMOKE = os.environ.get("CNN_SMOKE") == "1"

# Model hyperparameters
CONV1_CH = 32
CONV2_CH = 64
SCALAR_HIDDEN = 256
HEAD_HIDDEN = 256
DROPOUT = 0.1
MAX_EPOCHS = 80 if not SMOKE else 4
PATIENCE = 12
BATCH_SIZE = 16384
INFER_BATCH_SIZE = 32768


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
            floored = np.floor(dset[col]).astype("float32")
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


# ─── Two-Branch CNN model ─────────────────────────────────────────────────────

class SiLU(nn.Module):
    def forward(self, x): return F.silu(x)


class TwoBranchSEDNet(nn.Module):
    """
    Branch A: 1D-CNN over wavelength-ordered bands [u,g,r,i,z] as length-5 signal
              with N_CHANNELS input channels. Learns spectral shape filters.
    Branch B: small MLP over all scalar features (redshift, coords, colors, TE, etc.)
    Head: concat(A, B) -> MLP -> 3-class softmax.
    """

    def __init__(self, n_scalar: int):
        super().__init__()

        # Branch A: 1D-CNN
        # Input: (batch, N_CHANNELS=3, 5) — 3 channels over 5 wavelength positions
        self.conv1 = nn.Conv1d(N_CHANNELS, CONV1_CH, kernel_size=2, padding=1)  # output: (B,32,6)
        self.bn1   = nn.BatchNorm1d(CONV1_CH)
        self.conv2 = nn.Conv1d(CONV1_CH, CONV2_CH, kernel_size=3, padding=1)   # output: (B,64,6)
        self.bn2   = nn.BatchNorm1d(CONV2_CH)
        # global avg+max pool: each contributes CONV2_CH = 64
        conv_out_dim = CONV2_CH * 2  # 128

        # Branch B: scalar MLP
        self.scalar_fc1 = nn.Linear(n_scalar, SCALAR_HIDDEN)
        self.scalar_fc2 = nn.Linear(SCALAR_HIDDEN, SCALAR_HIDDEN)
        self.scalar_dropout = nn.Dropout(DROPOUT)

        # Head: concat branches
        head_in = conv_out_dim + SCALAR_HIDDEN
        self.head_fc1 = nn.Linear(head_in, HEAD_HIDDEN)
        self.head_dropout = nn.Dropout(DROPOUT)
        self.head_out = nn.Linear(HEAD_HIDDEN, N_CLASSES)

    def forward(self, bands: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        bands:   (batch, N_CHANNELS, N_BANDS=5)
        scalars: (batch, n_scalar)
        returns: (batch, N_CLASSES) logits
        """
        # Branch A
        x = F.silu(self.bn1(self.conv1(bands)))   # (B, 32, L)
        x = F.silu(self.bn2(self.conv2(x)))        # (B, 64, L)
        x_avg = x.mean(dim=2)                      # (B, 64)
        x_max = x.max(dim=2).values                # (B, 64)
        x_conv = torch.cat([x_avg, x_max], dim=1) # (B, 128)

        # Branch B
        s = F.silu(self.scalar_fc1(scalars))
        s = self.scalar_dropout(s)
        s = F.silu(self.scalar_fc2(s))

        # Head
        h = torch.cat([x_conv, s], dim=1)
        h = F.silu(self.head_fc1(h))
        h = self.head_dropout(h)
        logits = self.head_out(h)
        return logits


def build_band_tensor(X_bands_np: np.ndarray, band_mu: np.ndarray, band_sd: np.ndarray) -> torch.Tensor:
    """
    Build 3-channel band tensor from raw band values.
    Channel 0: raw magnitude (standardized).
    Channel 1: per-row mean-subtracted SED shape.
    Channel 2: per-row z-normalized magnitude.
    All normalization stats (band_mu, band_sd) fit on train fold only.

    X_bands_np: (N, N_BANDS) raw magnitudes in wavelength order.
    Returns: (N, N_CHANNELS, N_BANDS) tensor.
    """
    # Channel 0: global standardization (per-band mean/std from train fold)
    ch0 = (X_bands_np - band_mu) / (band_sd + 1e-8)   # (N, 5)

    # Channel 1: per-row mean-subtracted SED shape
    row_mean = X_bands_np.mean(axis=1, keepdims=True)
    ch1 = X_bands_np - row_mean                         # (N, 5)
    # normalize channel 1 with same band_sd for scale consistency
    ch1 = ch1 / (band_sd + 1e-8)

    # Channel 2: per-row z-normalized (per-row mean and std)
    row_std = X_bands_np.std(axis=1, keepdims=True) + 1e-8
    ch2 = (X_bands_np - row_mean) / row_std             # (N, 5)

    # Stack: (N, 3, 5)
    out = np.stack([ch0, ch1, ch2], axis=1).astype(np.float32)
    return torch.as_tensor(out)


def predict_proba_batch(model: TwoBranchSEDNet,
                        bands_t: torch.Tensor,
                        scalars_np: np.ndarray,
                        band_mu, band_sd) -> np.ndarray:
    """Run inference in batches. bands_t already processed. Returns (N, 3) probs."""
    model.eval()
    out = []
    N = scalars_np.shape[0]
    with torch.no_grad():
        for s in range(0, N, INFER_BATCH_SIZE):
            b_t = bands_t[s:s + INFER_BATCH_SIZE].to(DEVICE)
            s_t = torch.as_tensor(scalars_np[s:s + INFER_BATCH_SIZE], dtype=torch.float32, device=DEVICE)
            logits = model(b_t, s_t)
            probs = torch.softmax(logits, dim=-1)
            out.append(probs.cpu().numpy().astype(np.float32))
    return np.concatenate(out, 0)


def train_cnn(
    bands_tr: np.ndarray,
    scalars_tr: np.ndarray,
    y_tr: np.ndarray,
    band_mu: np.ndarray,
    band_sd: np.ndarray,
    fold_seed: int,
) -> TwoBranchSEDNet:
    """Train two-branch CNN with early stopping on internal 10% val split."""
    torch.manual_seed(fold_seed)
    np.random.seed(fold_seed)

    n = len(bands_tr)
    rng = np.random.default_rng(fold_seed)
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    vi, ti = perm[:n_val], perm[n_val:]

    model = TwoBranchSEDNet(n_scalar=scalars_tr.shape[1]).to(DEVICE)

    # Class weights (balanced)
    counts = np.bincount(y_tr, minlength=N_CLASSES).astype(np.float64)
    class_w = torch.tensor(
        counts.sum() / (N_CLASSES * counts), dtype=torch.float32, device=DEVICE
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS, eta_min=1e-5)

    # Build band tensors (all normalization uses band_mu/band_sd fit on train fold)
    bands_t_tr = build_band_tensor(bands_tr[ti], band_mu, band_sd)
    bands_t_vi = build_band_tensor(bands_tr[vi], band_mu, band_sd)

    # Move train split to GPU once
    B_t  = bands_t_tr.to(DEVICE)
    S_t  = torch.as_tensor(scalars_tr[ti], dtype=torch.float32, device=DEVICE)
    Y_t  = torch.as_tensor(y_tr[ti], dtype=torch.long, device=DEVICE)
    nt   = len(ti)

    y_vi = y_tr[vi]
    sc_vi = scalars_tr[vi]

    best_ba = -1.0
    best_state = None
    bad = 0

    for ep in range(MAX_EPOCHS):
        model.train()
        bperm = torch.randperm(nt, device=DEVICE)
        for s in range(0, nt, BATCH_SIZE):
            idx = bperm[s:s + BATCH_SIZE]
            b_b = B_t[idx]
            s_b = S_t[idx]
            y_b = Y_t[idx]
            opt.zero_grad()
            logits = model(b_b, s_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        val_probs = predict_proba_batch(model, bands_t_vi, sc_vi, band_mu, band_sd)
        ba = balanced_accuracy_score(y_vi, val_probs.argmax(1))
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
    log(f"    CNN early-stop: best_int_ba={best_ba:.5f}  ep_stopped={ep+1}")
    return model


# ─── Stack helpers ────────────────────────────────────────────────────────────
def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))


def score_fn(y_true, y_pred) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(N_CLASSES) if (y_true == c).any()]
    ))


def fit_meta(Xtr, ytr) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m


def best_thr_de(probs, labels) -> np.ndarray:
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)],
                                maxiter=40, tol=1e-7, seed=0, polish=False, workers=1)
    return np.array([r.x[0], r.x[1], 1.0])


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw  = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw   = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all   = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test  = len(test_raw)

if SMOKE:
    log("SMOKE MODE: subsample 30000 rows, 1 fold")
    rng_sm  = np.random.default_rng(0)
    keep_sm = rng_sm.choice(n_train, 30000, replace=False)
    folds_list = [folds_list[0]]

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE ...")
X_raw        = train_raw.drop(columns=[IDC, TARGET])
X_test_raw   = test_raw.drop(columns=[IDC])
X_stateless  = stateless_fe(X_raw)
X_test_stat  = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba        = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test,  N_CLASSES), dtype=np.float32)
per_fold_scores  = []
features_written = False
all_scalar_cols  = None

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id  = fi["fold"]
    val_idx  = np.asarray(fi["val_idx"])
    tr_idx   = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100
    seed_everything(fold_seed)

    if SMOKE:
        keep_set = set(keep_sm.tolist())
        tr_idx  = np.array([i for i in tr_idx  if i in keep_set])
        val_idx = np.array([i for i in val_idx if i in keep_set])

    log(f"Fold {fold_id}: train={len(tr_idx)}  val={len(val_idx)}")

    # Categorical encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stat.copy(),
    )

    y_tr_fold  = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # Target encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # Sort consistently
    X_tr_fold  = X_tr_fold.reindex(sorted(X_tr_fold.columns),  axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold  = X_te_fold.reindex(sorted(X_te_fold.columns),  axis=1)

    # Extract raw band arrays (wavelength-ordered, already in correct order) — NOT standardized yet
    # Band normalization is fit on train fold only (fit_in_fold)
    bands_tr_raw  = X_tr_fold[BAND_COLS].values.astype(np.float32)
    bands_val_raw = X_val_fold[BAND_COLS].values.astype(np.float32)
    bands_te_raw  = X_te_fold[BAND_COLS].values.astype(np.float32)

    # Fit band normalization stats on train fold only
    band_mu = bands_tr_raw.mean(axis=0)  # (5,) per-band mean
    band_sd = bands_tr_raw.std(axis=0)   # (5,) per-band std

    # Scalar features = ALL columns (including bands — overlap is fine; different inductive bias)
    # Filter out original categorical-type columns that can't be standardized meaningfully
    all_cols_sorted = sorted(X_tr_fold.columns)
    # Treat everything numeric (incl. integer-floor cat codes) as scalar floats
    # (same approach as node_0033's "num_for_tabm")
    cat_cols_sorted = sorted(cat_cols)
    SCALAR_CAT_COLS = [c for c in cat_cols_sorted if c in BASE_CAT_COLS]  # only low-card cats
    scalar_cols = [c for c in all_cols_sorted if c not in SCALAR_CAT_COLS]

    if all_scalar_cols is None:
        all_scalar_cols = scalar_cols
        log(f"  n_scalar={len(scalar_cols)}  n_bands={N_BANDS}  n_channels={N_CHANNELS}")

    # Extract scalar arrays
    Xs_tr  = X_tr_fold[scalar_cols].values.astype(np.float32)
    Xs_val = X_val_fold[scalar_cols].values.astype(np.float32)
    Xs_te  = X_te_fold[scalar_cols].values.astype(np.float32)

    # Standardize scalar features — fit on train fold only
    mu_s = Xs_tr.mean(0)
    sd_s = Xs_tr.std(0) + 1e-8
    Xs_tr  = (Xs_tr  - mu_s) / sd_s
    Xs_val = (Xs_val - mu_s) / sd_s
    Xs_te  = (Xs_te  - mu_s) / sd_s

    # Write features.txt from first fold
    if not features_written:
        (NODE_SRC / "features.txt").write_text("\n".join(scalar_cols + BAND_COLS) + "\n")
        features_written = True

    # Train CNN
    model = train_cnn(bands_tr_raw, Xs_tr, y_tr_fold, band_mu, band_sd, fold_seed)

    # OOF predictions
    bands_val_t = build_band_tensor(bands_val_raw, band_mu, band_sd)
    val_probs = predict_proba_batch(model, bands_val_t, Xs_val, band_mu, band_sd)
    oof_proba[val_idx] = val_probs.astype(np.float32)

    # Test predictions
    bands_te_t = build_band_tensor(bands_te_raw, band_mu, band_sd)
    test_probs_fold = predict_proba_batch(model, bands_te_t, Xs_te, band_mu, band_sd)
    test_proba_accum += test_probs_fold.astype(np.float32) / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(oof_proba[val_idx], axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    if torch.cuda.is_available():
        vram_gb = torch.cuda.max_memory_allocated() / 1e9
        log(f"  peak VRAM so far: {vram_gb:.2f} GB")

    del model, X_tr_fold, X_val_fold, X_te_fold, Xs_tr, Xs_val, Xs_te
    del bands_tr_raw, bands_val_raw, bands_te_raw, bands_val_t, bands_te_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Timing projection after first fold
    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s ({projected/60:.1f}min)")

if SMOKE:
    log("[smoke] OK — pipeline ran. Exiting before saving artifacts.")
    sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Save OOF / test_probs ────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy",        oof_proba)
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved oof.npy={oof_proba.shape}  test_probs.npy={test_proba_accum.shape}")

# ─── Write submission ─────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── Full OOF metric ──────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

# ─── DE per-class threshold tuning (standalone) ───────────────────────────────
log("Computing DE per-class threshold (standalone) ...")
fval = [np.asarray(f["val_idx"]) for f in folds_list]

per_fold_de_scores = []
for i, vi in enumerate(fval):
    other = np.setdiff1d(np.arange(n_train), vi)
    w = best_thr_de(oof_proba[other], y_all[other])
    pred = np.argmax(oof_proba[vi] * w, axis=1)
    s = score_fn(y_all[vi], pred)
    per_fold_de_scores.append(s)
    log(f"  DE fold {i}: score={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

de_mean = float(np.mean(per_fold_de_scores))
de_sem  = float(np.std(per_fold_de_scores, ddof=1) / np.sqrt(len(per_fold_de_scores)))
log(f"standalone_de_cv={de_mean:.6f}  sem={de_sem:.6f}")
print(f"standalone_de_cv={de_mean:.6f}", flush=True)

# ─── Re-stack: CORE15 + node_0056 ────────────────────────────────────────────
log("Re-stack: CORE15 (15 bases) + node_0056 ...")

BASES_CORE15 = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030",
    "node_0039",
]

nodes_dir = COMP_DIR / "nodes"
try:
    OOF_CORE  = np.concatenate([logp(np.load(nodes_dir / b / "oof.npy"))  for b in BASES_CORE15], axis=1)
    TEST_CORE = np.concatenate([logp(np.load(nodes_dir / b / "test_probs.npy")) for b in BASES_CORE15], axis=1)
    log(f"  CORE15 OOF={OOF_CORE.shape}  TEST={TEST_CORE.shape}")

    # Add this node's log-probs as 16th base
    n56_log      = logp(oof_proba)         # (N_train, 3)
    n56_test_log = logp(test_proba_accum)  # (N_test,  3)

    OOF_STACK  = np.concatenate([OOF_CORE,  n56_log],      axis=1)
    TEST_STACK = np.concatenate([TEST_CORE, n56_test_log],  axis=1)
    log(f"  stacked OOF={OOF_STACK.shape}  TEST={TEST_STACK.shape}")

    # Fold-honest meta + DE threshold
    stack_oof = np.zeros((n_train, N_CLASSES))
    for vi in fval:
        tr = np.setdiff1d(np.arange(n_train), vi)
        m = fit_meta(OOF_STACK[tr], y_all[tr])
        stack_oof[vi] = m.predict_proba(OOF_STACK[vi])

    per_fold_restack = []
    for i, vi in enumerate(fval):
        other = np.setdiff1d(np.arange(n_train), vi)
        w = best_thr_de(stack_oof[other], y_all[other])
        pred = np.argmax(stack_oof[vi] * w, axis=1)
        s = score_fn(y_all[vi], pred)
        per_fold_restack.append(s)
        log(f"  restack fold {i}: score={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

    restack_mean = float(np.mean(per_fold_restack))
    restack_sem  = float(np.std(per_fold_restack, ddof=1) / np.sqrt(len(per_fold_restack)))
    log(f"restack_cv={restack_mean:.6f}  sem={restack_sem:.6f}")
    print(f"restack_cv={restack_mean:.6f}", flush=True)

    # Error correlation: node_0056 OOF vs CORE15 mean
    core15_mean_probs = np.exp(OOF_CORE.reshape(n_train, 15, 3).mean(axis=1))  # (N, 3) approx
    core15_pred = core15_mean_probs.argmax(1)
    n56_pred    = oof_proba.argmax(1)
    # Compute error vectors and their correlation
    core15_err = (core15_pred != y_all).astype(float)
    n56_err    = (n56_pred    != y_all).astype(float)
    err_corr   = float(np.corrcoef(core15_err, n56_err)[0, 1])
    log(f"error_correlation(node_0056 vs CORE15 mean): {err_corr:.4f}")
    print(f"error_corr={err_corr:.4f}", flush=True)

    # Save re-stacked submission
    # Fit meta on full train for final submission
    m_final = fit_meta(OOF_STACK, y_all)
    test_meta_probs = m_final.predict_proba(TEST_STACK)
    w_final = best_thr_de(stack_oof, y_all)
    test_pred_final = np.argmax(test_meta_probs * w_final, axis=1)
    pred_labels_rs = np.array([CLASSES[i] for i in test_pred_final])
    sub_rs = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels_rs})
    sub_rs = sub_rs[list(sample_sub.columns)]
    sub_rs.to_csv(NODE_DIR / "submission_restack.csv", index=False)
    log(f"Saved submission_restack.csv")

except Exception as e:
    log(f"Re-stack failed (non-fatal): {e}")
    restack_mean = None
    err_corr = None

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
