"""node_0124 — improve (gbdt): LightGBM on fs_realmlp_fe + fs_physloc features.

THE ONE ATOMIC CHANGE vs node_0030:
  Add the stateless fs_physloc feature block:
    feh_phot  — photometric metallicity (Ivezic 2008, tomographyII eq.4)
    P1s       — stellar-locus principal color 1 (along the locus)
    P2s       — stellar-locus principal color 2 (perp, ~=0 for stars)
    z_warp    — log10(z + 3e-4), widens the z~=0 crush into a stable margin
    feh_in_range — 1.0 if -3 < feh_phot < 0.6, else 0.0 (galaxy signal)
  All formulas are STATELESS row-wise transforms: no .fit, no target, no
  cross-row stats — applied identically to train and test. FE/folds/tree-params
  from node_0030 are kept byte-identical.

Leakage discipline (same as node_0030 + new block):
  - fs_physloc: STATELESS (same row-wise math on train AND test, no fit, no target).
    Computed on full data before fold loop — safe.
  - Stateless FE (color pairs, mag stats, redshift ratio, log1p_redshift) same as n030.
  - KBinsDiscretizer, factorize maps: fit on train-fold only (fit_in_fold).
  - TargetEncoder: fit on train-fold only (fit_in_fold).
  - Frozen folds.json used; no refit of folds.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
# comp dir is two levels up from node dir: nodes/node_0124 → nodes → comp
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


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure row-wise / stateless feature engineering — safe to apply to the full
    dataframe before any fold split. No fitting, no target, no cross-row stats.
    Includes fs_realmlp_fe features (from node_0030) AND the new fs_physloc block.
    """
    df = df.copy()

    # ── fs_realmlp_fe features (byte-identical to node_0030) ──────────────────

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

    # ── fs_physloc features (NEW — the ONE atomic change) ─────────────────────
    # Use colors already computed above (reuse for efficiency)
    ug = df["_u-g"].astype("float64")
    gr = df["_g-r"].astype("float64")
    z_col = df["redshift"].astype("float64")

    # Ivezic 2008 (tomographyII eq.4) photometric metallicity
    # x = (u-g) if (g-r) <= 0.4 else (u-g) - 2*(g-r) + 0.8
    x = np.where(gr <= 0.4, ug, ug - 2.0 * gr + 0.8)
    y = gr.values

    feh_phot = (
        -13.13
        + 14.09 * x
        + 28.04 * y
        - 5.51 * x * y
        - 5.90 * x ** 2
        - 58.68 * y ** 2
        + 9.14 * x ** 2 * y
        - 20.61 * x * y ** 2
        + 0.00 * x ** 3
        + 58.20 * y ** 3
    )

    df["_feh_phot"] = feh_phot.astype("float32")

    # Stellar-locus principal colors (Bond 2010 coefficients)
    df["_P1s"] = (0.910 * ug + 0.415 * gr - 1.28).astype("float32")
    df["_P2s"] = (-0.249 * ug + 0.545 * gr + 0.234).astype("float32")

    # Redshift warp: log10(z + 3e-4), expands z~=0 boundary
    # Guard: clamp the argument to a tiny positive floor so log10 is safe
    z_arg = z_col + 3e-4
    z_arg_safe = np.where(z_arg <= 0, 1e-10, z_arg)
    df["_z_warp"] = np.log10(z_arg_safe).astype("float32")

    # feh_in_range: 1.0 if physically valid stellar range, else 0.0
    # Out-of-range = galaxy signal — do NOT clip feh_phot itself
    df["_feh_in_range"] = np.where(
        (feh_phot > -3.0) & (feh_phot < 0.6), 1.0, 0.0
    ).astype("float32")

    return df


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Fit categorical encodings on train-fold only, transform val and test.
    Returns (df_tr, df_val, df_te, cat_cols, combo_names, local_map).
    Called INSIDE the fold loop — fit_in_fold.

    Note: columns are encoded as int32 (integer codes, NOT dtype='category').
    LightGBM will treat these as numeric continuous features — safe for LightGBM
    since threshold splits work for ordinal integers. Only a small subset will be
    passed as categorical_feature (spectral_type, galaxy_population).
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
    Identical to node_0030 — no param changes.
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

# ─── Pre-flight leakage checks ────────────────────────────────────────────────
log("Running pre-flight leakage checks ...")

# Check 1+2: target and id not in features
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])
assert TARGET not in X_raw.columns, f"LEAK: target {TARGET} in feature cols"
assert IDC not in X_raw.columns, f"LEAK: id {IDC} in feature cols"
log("  check1+2 PASS: target/id absent from features")

# Check 3: single-feature↔target sweep on ≤50k sample (before adding derived features)
# We check the raw numeric features plus the physloc features
_sample_size = min(50_000, n_train)
_rng = np.random.RandomState(0)
_sample_idx = _rng.choice(n_train, _sample_size, replace=False)
_s_raw = train_raw.iloc[_sample_idx].copy()
_ys = np.array([LABEL_MAP[c] for c in _s_raw[TARGET]])

# Apply stateless FE to get derived features for sweep
_s_fe = stateless_fe(_s_raw.drop(columns=[IDC, TARGET]))
_physloc_cols = ["_feh_phot", "_P1s", "_P2s", "_z_warp", "_feh_in_range"]
_suspicious = []
for c in _physloc_cols:
    x_arr = pd.to_numeric(_s_fe[c], errors="coerce")
    if x_arr.nunique() > 1:
        corr_val = abs(np.corrcoef(x_arr.fillna(x_arr.mean()), _ys)[0, 1])
        if corr_val >= 0.999:
            _suspicious.append((c, corr_val))
            log(f"  LEAK SMELL: {c} ~ target corr={corr_val:.6f}")
if not _suspicious:
    log(f"  check3 PASS: no physloc feature has |corr| >= 0.999 vs target on {_sample_size} rows")
else:
    raise SystemExit(f"LEAK DETECTED (check3): {_suspicious}")

# Check 4: fs_physloc is STATELESS — verify identical transform on train vs test
# (no fit needed; math is row-wise). Checked by construction: same function applied to both.
log("  check4 PASS: fs_physloc is STATELESS (same row-wise math on train and test, no .fit)")

# Check 5: folds from frozen folds.json — already loaded above
log(f"  check5 PASS: folds loaded from frozen folds.json ({len(folds_list)} folds)")

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE (fs_realmlp_fe + fs_physloc) ...")
X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# Log z_warp guard usage
z_series = train_raw["redshift"]
n_clipped = (z_series + 3e-4 <= 0).sum()
if n_clipped > 0:
    log(f"  z_warp guard: {n_clipped} rows had z + 3e-4 <= 0, clamped to 1e-10")
else:
    log(f"  z_warp guard: no rows needed clamping (all z + 3e-4 > 0)")

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

    # Categorical encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, all_cat_cols, combo_names, local_map, lgbm_cat_cols = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # Target encoding — fit_in_fold
    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # Sort columns consistently
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    if all_cols_final is None:
        all_cols_final = list(X_tr_fold.columns)
        log(f"  n_features={X_tr_fold.shape[1]}  lgbm_cat={lgbm_cat_cols}")

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

    # Cheap-kill after fold 0
    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")
        if fold_score < 0.965:
            log(f"  CHEAP-KILL: fold-0 BA={fold_score:.6f} < 0.965 threshold")
            print(f"CHEAP_KILLED fold0_BA={fold_score:.6f}", flush=True)
            sys.exit(1)
        log(f"  fold-0 BA={fold_score:.6f} >= 0.965 — proceeding to full 5-fold run")

    del model, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

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

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
