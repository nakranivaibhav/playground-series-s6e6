"""node_0131 — improve (gbdt): LightGBM on fs_realmlp_fe + fs_aerecon.

THE ONE ATOMIC CHANGE vs node_0030:
  Add fs_aerecon (fit_in_fold) — per-class autoencoder reconstruction-error GAPS.
  Everything else byte-identical from node_0030.

fs_aerecon recipe:
  - Per fold: train 1 tiny AE per class (GALAXY/QSO/STAR) on standardized
    (ugriz, redshift, 7 color-diffs) of TRAIN-FOLD rows of THAT class only.
  - Architecture: [input→16→4→16→input] (small bottleneck).
  - For every row emit {err_GAL, err_QSO, err_STAR} (MSE recon error under each
    class-AE) + 3 DIFFS (err_STAR−err_GAL, err_QSO−err_GAL, err_QSO−err_STAR) +
    argmin = 7 new features total.
  - Scaler fit on train-fold only (fit_in_fold).
  - Val/test rows transformed (never fit).
  - Implementation: torch (libraries-first; MLPRegressor fallback not needed).

Leakage discipline:
  - fs_aerecon: scaler + class-AEs fit on TRAIN-FOLD rows ONLY; val/test only
    transformed. Verified by reading the fold loop below.
  - All node_0030 fit_in_fold discipline preserved byte-identical.

Source: Marks/Griffin/Corso 2024 arXiv:2412.02596 (per-class recon-error-ratio).
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler, TargetEncoder

warnings.filterwarnings("ignore")

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

# AE input features — standardized before fitting
AE_BASE_COLS = ["u", "g", "r", "i", "z", "redshift"]
AE_COLOR_PAIRS = [
    ("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
    ("u", "r"), ("g", "i"), ("r", "z"),
]
# Total: 6 base + 7 colors = 13 features
AE_INPUT_DIM = len(AE_BASE_COLS) + len(AE_COLOR_PAIRS)  # 13

# ─── Feature engineering globals ─────────────────────────────────────────────
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


# ─── Autoencoder (torch) ──────────────────────────────────────────────────────
class TinyAutoencoder(nn.Module):
    """[input→16→4→16→input] — tiny bottleneck per-class AE."""

    def __init__(self, input_dim: int = AE_INPUT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def build_ae_input(df: pd.DataFrame) -> np.ndarray:
    """Extract + compute AE input features (row-wise, no fitting)."""
    out = df[AE_BASE_COLS].copy().astype("float32")
    for a, b in AE_COLOR_PAIRS:
        out[f"_{a}-{b}_ae"] = (df[a] - df[b]).astype("float32")
    return out.values.astype(np.float32)


def train_class_ae(X_class: np.ndarray, seed: int, n_epochs: int = 30) -> TinyAutoencoder:
    """Train a tiny AE on X_class (already standardized). Returns trained model."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyAutoencoder(input_dim=X_class.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_class).to(device)
    batch_size = min(2048, len(X_class))

    model.train()
    n = len(X_class)
    for epoch in range(n_epochs):
        # shuffle
        perm = torch.randperm(n, generator=rng if device.type == "cpu" else None)
        X_t_shuf = X_t[perm]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            batch = X_t_shuf[start:start + batch_size]
            opt.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

    model.eval()
    return model


def compute_recon_errors(model: TinyAutoencoder, X: np.ndarray) -> np.ndarray:
    """Compute per-row MSE reconstruction error. Returns (n,) array."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        recon = model(X_t)
        # per-row MSE
        errors = ((recon - X_t) ** 2).mean(dim=1).cpu().numpy()
    return errors.astype(np.float32)


def compute_fs_aerecon(
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    X_val_raw: np.ndarray,
    X_te_raw: np.ndarray,
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit fs_aerecon features on train-fold only; transform val and test.
    LEAK DISCIPLINE: StandardScaler + class-AEs fit on TRAIN-FOLD rows ONLY.
    Val and test rows: scaler.transform() + recon error from fitted AEs.

    Returns (train_feats, val_feats, test_feats) each shape (n, 7):
      [err_GAL, err_QSO, err_STAR, diff_STAR-GAL, diff_QSO-GAL, diff_QSO-STAR, argmin]
    """
    # 1. Fit scaler on ALL train-fold rows
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_raw)  # fit on train only
    X_val_scaled = scaler.transform(X_val_raw)     # transform only
    X_te_scaled = scaler.transform(X_te_raw)       # transform only

    # 2. Train one AE per class on THAT class's train-fold rows only
    aes = {}
    for cls_idx, cls_name in enumerate(CLASSES):
        mask = y_tr == cls_idx
        X_cls = X_tr_scaled[mask]
        if len(X_cls) < 10:
            # degenerate fold (shouldn't happen): zero-fill error
            aes[cls_idx] = None
            continue
        ae = train_class_ae(X_cls, seed=fold_seed + cls_idx * 17, n_epochs=30)
        aes[cls_idx] = ae

    def _get_errors(X_scaled):
        errs = []
        for cls_idx in range(N_CLASSES):
            if aes[cls_idx] is None:
                errs.append(np.zeros(len(X_scaled), dtype=np.float32))
            else:
                errs.append(compute_recon_errors(aes[cls_idx], X_scaled))
        return np.stack(errs, axis=1)  # (n, 3)

    # 3. Compute reconstruction errors
    tr_errs = _get_errors(X_tr_scaled)   # (n_tr, 3) — each row vs each class-AE
    val_errs = _get_errors(X_val_scaled)
    te_errs = _get_errors(X_te_scaled)

    def _build_feats(errs):
        e_gal = errs[:, 0]
        e_qso = errs[:, 1]
        e_str = errs[:, 2]
        diff_sg = e_str - e_gal
        diff_qg = e_qso - e_gal
        diff_qs = e_qso - e_str
        argmin = errs.argmin(axis=1).astype(np.float32)
        return np.stack([e_gal, e_qso, e_str, diff_sg, diff_qg, diff_qs, argmin], axis=1)

    return _build_feats(tr_errs), _build_feats(val_errs), _build_feats(te_errs)


AERECON_FEAT_NAMES = [
    "ae_err_GAL", "ae_err_QSO", "ae_err_STAR",
    "ae_diff_STAR_GAL", "ae_diff_QSO_GAL", "ae_diff_QSO_STAR",
    "ae_argmin",
]


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure row-wise / stateless feature engineering — safe to apply to the full
    dataframe before any fold split. No fitting, no target, no cross-row stats.
    """
    df = df.copy()

    # Redshift ratios
    df["_g_div_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0).astype("float32")

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
    Returns (df_tr, df_val, df_te, cat_cols, combo_names, local_map).
    Called INSIDE the fold loop — fit_in_fold.
    """
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

    # Work on copies
    tr = df_tr.copy()
    va = df_val.copy()
    te = df_te.copy()

    # Original categorical columns (spectral_type, galaxy_population) — low cardinality
    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32")

    # Integer-floor categorical views of every base numeric — high cardinality, treated as numeric
    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32")
        for dset in [va, te]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32")

    # Delta quantile bins (100 and 500) — fit_in_fold via KBinsDiscretizer
    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32")

    # Interaction cross-combos — high cardinality, treated as numeric
    combo_names = []
    for cols in IMPORTANT_COMBOS:
        combo_name = "__".join(cols) + "__"
        combo_names.append(combo_name)
        combo_tr = tr[cols[0]].astype(str)
        for col in cols[1:]:
            combo_tr = combo_tr + "|" + tr[col].astype(str)
        codes_tr, uniques = pd.factorize(combo_tr, sort=False)
        local_map[combo_name] = uniques
        tr[combo_name] = pd.Series(codes_tr.astype("int32"), index=tr.index).astype("int32")
        for dset in [va, te]:
            combo_s = dset[cols[0]].astype(str)
            for col in cols[1:]:
                combo_s = combo_s + "|" + dset[col].astype(str)
            codes = factorize_transform(combo_s, uniques)
            dset[combo_name] = pd.Series(codes, index=dset.index).astype("int32")

    # Only low-cardinality BASE_CAT_COLS are flagged for native LightGBM cat treatment
    lgbm_cat_cols = BASE_CAT_COLS[:]
    # All columns produced (including int-floor cats, bins, combos)
    all_new_cols = (
        BASE_CAT_COLS
        + [f"{c}_cat_" for c in BASE_NUM_COLS]
        + [f"delta_{n}_quantile_bin_" for n in [100, 500]]
        + combo_names
    )
    all_new_cols = [c for c in all_new_cols if c in tr.columns]

    return tr, va, te, all_new_cols, combo_names, local_map, lgbm_cat_cols


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names: list, fold_seed: int):
    """
    TargetEncoder fit on train fold only (fit_in_fold), transform val and test.
    Returns modified copies and the list of new TE column names.
    """
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


def make_lgbm(fold_seed: int) -> LGBMClassifier:
    """
    Well-tuned LightGBM for Balanced Accuracy (macro per-class recall).
    """
    return LGBMClassifier(
        objective="multiclass",
        num_class=N_CLASSES,
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        min_child_weight=1e-3,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=fold_seed,
        verbosity=-1,
        device="cpu",
    )


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

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── Pre-compute AE raw inputs (stateless row-wise; no fitting) ───────────────
log("Building AE raw inputs (stateless) ...")
X_ae_raw = build_ae_input(X_raw)          # (n_train, 13)
X_te_ae_raw = build_ae_input(X_test_raw)  # (n_test, 13)
log(f"  X_ae_raw={X_ae_raw.shape}  X_te_ae_raw={X_te_ae_raw.shape}")

# ─── Pre-flight leakage checks ────────────────────────────────────────────────
log("PRE-FLIGHT: checking features do not include target or id ...")
# These checks will run once we have the feature matrix in the fold loop;
# the stateless-FE column check runs now.
assert TARGET not in X_stateless.columns, f"LEAK: {TARGET} in features"
assert IDC not in X_stateless.columns, f"LEAK: {IDC} in features"
assert TARGET not in [c for c in AE_BASE_COLS], f"LEAK: target in AE inputs"
log("  target/id not in stateless FE columns: OK")

# Quick single-feature correlation sweep on stateless features (<=50k sample)
log("PRE-FLIGHT: single-feature sweep on stateless columns ...")
sample_idx = np.random.RandomState(0).choice(n_train, min(50_000, n_train), replace=False)
ys_sample = y_all[sample_idx]
s_df = X_stateless.iloc[sample_idx]
max_abs_corr = 0.0
max_corr_col = None
for c in s_df.columns:
    x_c = pd.to_numeric(s_df[c], errors="coerce").fillna(0).values
    if x_c.std() < 1e-12:
        continue
    corr_val = abs(float(np.corrcoef(x_c, ys_sample)[0, 1]))
    if corr_val > max_abs_corr:
        max_abs_corr = corr_val
        max_corr_col = c
    if corr_val >= 0.999:
        raise SystemExit(f"LEAK SMELL: {c} has |corr|={corr_val:.4f} vs target — stop and inspect")
log(f"  max |corr| in stateless features: {max_abs_corr:.4f} (col={max_corr_col}) — OK (<0.999)")

# AE feature names — for logging
log(f"  AE feature names: {AERECON_FEAT_NAMES}")
log("PRE-FLIGHT: folds loaded from frozen folds.json — not recomputed. OK")
log("PRE-FLIGHT: all checks passed. Launching OOF loop.")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
all_cols_final = None
best_iters = []

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # ── fs_aerecon: fit_in_fold ────────────────────────────────────────────────
    # LEAK DISCIPLINE:
    #   - scaler.fit() on X_ae_raw[tr_idx] (train-fold rows ONLY)
    #   - class-AE trained on train-fold rows of THAT class ONLY
    #   - val/test: scaler.transform() only, then recon error from fitted AEs
    log(f"  [Fold {fold_id}] Training class-AEs (fit_in_fold) ...")
    ae_t0 = time.perf_counter()
    tr_ae_feats, val_ae_feats, te_ae_feats = compute_fs_aerecon(
        X_ae_raw[tr_idx],           # train-fold AE inputs (raw, unstandardized)
        y_all[tr_idx],              # train-fold labels (for class-split)
        X_ae_raw[val_idx],          # val AE inputs (raw)
        X_te_ae_raw,                # test AE inputs (raw)
        fold_seed=fold_seed,
    )
    ae_elapsed = time.perf_counter() - ae_t0
    log(f"  [Fold {fold_id}] AE training done in {ae_elapsed:.1f}s; "
        f"tr_ae_feats={tr_ae_feats.shape}")

    # ── Categorical encoding — fit_in_fold ────────────────────────────────────
    X_tr_fold, X_val_fold, X_te_fold, all_cat_cols, combo_names, local_map, lgbm_cat_cols = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # ── Target encoding — fit_in_fold ─────────────────────────────────────────
    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # ── Append fs_aerecon columns ──────────────────────────────────────────────
    # tr: aligned to tr_idx (already sorted), reset_index not needed since df was reset
    ae_tr_df = pd.DataFrame(tr_ae_feats, columns=AERECON_FEAT_NAMES, dtype="float32")
    ae_val_df = pd.DataFrame(val_ae_feats, columns=AERECON_FEAT_NAMES, dtype="float32")
    ae_te_df = pd.DataFrame(te_ae_feats, columns=AERECON_FEAT_NAMES, dtype="float32")

    # Reset indices to allow clean concat
    X_tr_fold = X_tr_fold.reset_index(drop=True)
    X_val_fold = X_val_fold.reset_index(drop=True)
    X_te_fold = X_te_fold.reset_index(drop=True)

    X_tr_fold = pd.concat([X_tr_fold, ae_tr_df.reset_index(drop=True)], axis=1)
    X_val_fold = pd.concat([X_val_fold, ae_val_df.reset_index(drop=True)], axis=1)
    X_te_fold = pd.concat([X_te_fold, ae_te_df.reset_index(drop=True)], axis=1)

    # ── Sort columns consistently ──────────────────────────────────────────────
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    if all_cols_final is None:
        all_cols_final = list(X_tr_fold.columns)
        log(f"  n_features={X_tr_fold.shape[1]}  lgbm_cat={lgbm_cat_cols}")
        log(f"  ae_feature_names={AERECON_FEAT_NAMES}")
        # Post-FE leakage check: target and id not in feature list
        assert TARGET not in all_cols_final, f"LEAK: {TARGET} in final features"
        assert IDC not in all_cols_final, f"LEAK: {IDC} in final features"
        log("  POST-FE: target/id not in feature list — OK")

    model = make_lgbm(fold_seed=fold_seed)

    model.fit(
        X_tr_fold, y_tr_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(stopping_rounds=150, verbose=False),
            log_evaluation(period=200),
        ],
        categorical_feature=lgbm_cat_cols,
    )

    best_iter = model.best_iteration_
    best_iters.append(best_iter)
    log(f"  best_iteration={best_iter}")

    # OOF probabilities
    val_proba = model.predict_proba(X_val_fold)
    oof_proba[val_idx] = val_proba.astype("float32")

    # Test predictions — average across folds
    test_proba_fold = model.predict_proba(X_te_fold)
    test_proba_accum += test_proba_fold.astype("float32") / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_proba, axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")
        # CHEAP-KILL CHECK
        if fold_score < 0.965:
            log(f"  CHEAP-KILL TRIGGERED: fold-0 BA={fold_score:.6f} < 0.965 threshold")
            print(f"CHEAP_KILL: fold0_BA={fold_score:.6f} < 0.965 threshold", flush=True)
            sys.exit(0)
        else:
            log(f"  CHEAP-KILL: fold-0 BA={fold_score:.6f} >= 0.965 — continuing")

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
log(f"best_iters_per_fold={best_iters}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Save OOF ────────────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# ─── Save test_probs ─────────────────────────────────────────────────────────
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

# ─── Write submission ─────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── Write features.txt ───────────────────────────────────────────────────────
(NODE_SRC / "features.txt").write_text("\n".join(sorted(all_cols_final)) + "\n")
log(f"Wrote features.txt ({len(all_cols_final)} features)")

# ─── Final OOF metric ────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

# ─── Post-run leakage checks ─────────────────────────────────────────────────
log("POST-RUN leakage checks ...")
# OOF covers every train row once, no NaN
assert oof_proba.shape == (n_train, N_CLASSES), f"OOF shape mismatch: {oof_proba.shape}"
assert not np.any(np.isnan(oof_proba)), "OOF contains NaN"
assert np.allclose(oof_proba.sum(axis=1), 1.0, atol=1e-4), "OOF probabilities don't sum to 1"
log(f"  OOF: shape={oof_proba.shape}, no NaN, probs sum to 1 — OK")
# Distribution sanity
assert test_proba_accum.shape == (n_test, N_CLASSES), f"test_probs shape mismatch"
assert not np.any(np.isnan(test_proba_accum)), "test_probs contains NaN"
log(f"  test_probs: shape={test_proba_accum.shape}, no NaN — OK")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
