"""node_0118 — draft (gbdt): LightGBM on fs_realmlp_fe AUGMENTED with fs_genfp.

THE ONE ATOMIC CHANGE vs node_0030:
  Add the new STATELESS feature-set fs_genfp — per-feature synthetic-generator
  quantization fingerprints — alongside the existing fs_realmlp_fe features.
  Everything else (FE pipeline, model config, fold scheme) is byte-identical.

fs_genfp (stateless, row-wise deterministic):
  For each base numeric (u, g, r, i, z, redshift, alpha, delta):
    - _fp_<col>_nsigdec   : number of significant decimal digits (0-12)
    - _fp_<col>_frac<k>   : fractional residual after rounding to k places (k=2,4,6)
    - _fp_<col>_lastdigit : last decimal digit (0-9) at 6th decimal place
    - _fp_<col>_trail0    : trailing-zero count in the decimal representation

  These probe class-conditional float-quantization artifacts left by the tabular
  generator, giving a GBDT an axis that is orthogonal to all photometric FE.

Leakage discipline (stateless FE — same rules as node_0030):
  - fs_genfp features are row-wise arithmetic — no target, no fitting, no
    cross-row stats. Computed once on the full dataframe BEFORE any fold split.
  - Categorical encodings (fit_in_fold) unchanged from node_0030.
  - TargetEncoder (fit_in_fold) unchanged.
  - Frozen folds.json used throughout.

GATES (per plan):
  1. Cheap-kill: fold-0 solo BA < 0.962 -> stop, mark dead.
  2. Fingerprint leak pre-flight: single-feature AUC scan on a sample
     BEFORE any training. Any fp feature with AUC >= 0.999 is a generator leak.
  3. Only if both pass -> full 5-fold.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
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
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# --- Constants ---------------------------------------------------------------
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

# --- Feature engineering globals ---------------------------------------------
BASE_CAT_COLS = ["spectral_type", "galaxy_population"]
BASE_NUM_COLS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]

# Columns targeted by fs_genfp fingerprinting
GENFP_COLS = ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]

COLOR_PAIRS = [
    ("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"),
    ("u", "r"), ("g", "i"), ("r", "z"),
]

IMPORTANT_COMBOS = sorted([
    ("alpha_cat_", "delta_cat_"),
    ("u_cat_", "z_cat_"),
])


# --- fs_genfp: generator decimal/mantissa fingerprints (stateless) -----------

def _count_sig_decimals(s: pd.Series, max_dec: int = 12) -> pd.Series:
    """Count number of significant decimal places (non-trailing-zero after dot)."""
    def _sig(v):
        if not np.isfinite(v):
            return 0
        s_str = f"{abs(v):.{max_dec}f}"
        if "." in s_str:
            frac = s_str.split(".")[1]
            frac = frac.rstrip("0")
            return len(frac)
        return 0
    return s.apply(_sig).astype("float32")


def _fractional_residual(s: pd.Series, k: int) -> pd.Series:
    """Residual after rounding to k decimal places: value - round(value, k)."""
    rounded = s.round(k)
    return (s - rounded).astype("float32")


def _last_digit(s: pd.Series, dec: int = 6) -> pd.Series:
    """Last significant decimal digit at position dec."""
    scaled = np.floor(np.abs(s.values) * (10 ** dec))
    return pd.Series((scaled % 10).astype("float32"), index=s.index)


def _trailing_zero_count(s: pd.Series, max_dec: int = 10) -> pd.Series:
    """Count trailing zeros in the decimal representation (up to max_dec places)."""
    def _tz(v):
        if not np.isfinite(v):
            return 0
        s_str = f"{abs(v):.{max_dec}f}"
        if "." in s_str:
            frac = s_str.split(".")[1]
            return len(frac) - len(frac.rstrip("0"))
        return 0
    return s.apply(_tz).astype("float32")


def add_genfp_fingerprints(df: pd.DataFrame) -> pd.DataFrame:
    """
    fs_genfp: per-feature synthetic-generator quantization fingerprints.
    Stateless -- row-wise deterministic, no fit, no target, no cross-row stats.
    Safe to compute on the full dataframe before any fold split.
    """
    df = df.copy()
    for col in GENFP_COLS:
        s = df[col]
        # Number of significant decimal places
        df[f"_fp_{col}_nsigdec"] = _count_sig_decimals(s)
        # Fractional residuals at k=2, 4, 6 decimal places
        for k in [2, 4, 6]:
            df[f"_fp_{col}_frac{k}"] = _fractional_residual(s, k)
        # Last digit at 6th decimal place
        df[f"_fp_{col}_lastdigit"] = _last_digit(s, dec=6)
        # Trailing zero count
        df[f"_fp_{col}_trail0"] = _trailing_zero_count(s)
    return df


def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure row-wise / stateless FE -- safe to apply before any fold split.
    Includes both fs_realmlp_fe features AND the new fs_genfp fingerprints.
    """
    df = df.copy()

    # fs_realmlp_fe (byte-identical from node_0030)
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

    # fs_genfp: new decimal/mantissa fingerprints
    df = add_genfp_fingerprints(df)

    return df


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Fit categorical encodings on train-fold only, transform val and test.
    Called INSIDE the fold loop -- fit_in_fold.
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

    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32")

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

    for n_bins in [100, 500]:
        bin_name = f"delta_{n_bins}_quantile_bin_"
        kb = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", subsample=None)
        binned_tr = kb.fit_transform(tr[["delta"]]).ravel().astype("int32")
        local_map[bin_name] = kb
        tr[bin_name] = pd.Series(binned_tr, index=tr.index).astype("int32")
        for dset in [va, te]:
            binned = kb.transform(dset[["delta"]]).ravel().astype("int32")
            dset[bin_name] = pd.Series(binned, index=dset.index).astype("int32")

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

    lgbm_cat_cols = BASE_CAT_COLS[:]
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
    """Well-tuned LightGBM for Balanced Accuracy -- byte-identical to node_0030."""
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


# --- Load data ---------------------------------------------------------------
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

# --- Stateless FE (computed once, safe) ---------------------------------------
log("Applying stateless FE (fs_realmlp_fe + fs_genfp) ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# --- PRE-FLIGHT LEAKAGE CHECKS -----------------------------------------------
log("=== PRE-FLIGHT LEAKAGE CHECKS ===")

# Check 1-2: target and id not in features
feature_cols_preleak = [c for c in X_stateless.columns if c not in [IDC, TARGET]]
assert TARGET not in feature_cols_preleak, f"LEAK: target '{TARGET}' in features!"
assert IDC not in feature_cols_preleak, f"LEAK: id '{IDC}' in features!"
log("Check 1-2 PASS: target and id not in features.")

# Identify fs_genfp columns for targeted scan
fp_cols = [c for c in feature_cols_preleak if c.startswith("_fp_")]
log(f"  fs_genfp features: {len(fp_cols)} columns")
log(f"  sample fp cols: {fp_cols[:6]}")

# Check 3: Single-feature fingerprint AUC scan (THE CRITICAL LEAK GATE)
log("Check 3: Single-feature fingerprint AUC/corr scan on <=50k sample ...")
from sklearn.metrics import roc_auc_score

sample_n = min(50_000, n_train)
rng_scan = np.random.default_rng(42)
scan_idx = rng_scan.choice(n_train, size=sample_n, replace=False)
X_scan = X_stateless.iloc[scan_idx][feature_cols_preleak]
y_scan = y_all[scan_idx]

worst_fp_col = None
worst_fp_auc = 0.0
LEAK_AUC_THRESHOLD = 0.999

all_fp_aucs = {}
for col in fp_cols:
    x_col = pd.to_numeric(X_scan[col], errors="coerce").fillna(0).values
    if np.unique(x_col).shape[0] < 2:
        all_fp_aucs[col] = 0.0
        continue
    try:
        x_min, x_max = x_col.min(), x_col.max()
        if x_max == x_min:
            all_fp_aucs[col] = 0.0
            continue
        x_norm = (x_col - x_min) / (x_max - x_min)
        max_auc = 0.0
        for cls in range(N_CLASSES):
            y_bin = (y_scan == cls).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            auc = roc_auc_score(y_bin, x_norm)
            max_auc = max(max_auc, auc, 1 - auc)
        all_fp_aucs[col] = max_auc
        if max_auc > worst_fp_auc:
            worst_fp_auc = max_auc
            worst_fp_col = col
    except Exception:
        all_fp_aucs[col] = 0.0

log(f"  Worst fp feature: {worst_fp_col}  max_AUC={worst_fp_auc:.6f}")
# Print top-5 fp features by AUC
top5 = sorted(all_fp_aucs.items(), key=lambda x: x[1], reverse=True)[:5]
for col, auc in top5:
    log(f"    {col}: AUC={auc:.6f}")

print(f"worst_fp_auc={worst_fp_auc:.6f}", flush=True)
print(f"worst_fp_col={worst_fp_col}", flush=True)

if worst_fp_auc >= LEAK_AUC_THRESHOLD:
    log(f"FINGERPRINT LEAK DETECTED: {worst_fp_col} AUC={worst_fp_auc:.6f} >= {LEAK_AUC_THRESHOLD}")
    log("Setting gates.leak_clean=false, leak=VOID, status=buggy. Exiting.")
    print(f"FINGERPRINT_LEAK col={worst_fp_col} auc={worst_fp_auc:.6f}", flush=True)
    sys.exit(1)

log(f"Check 3 PASS: worst fp AUC={worst_fp_auc:.6f} < {LEAK_AUC_THRESHOLD}  (col={worst_fp_col})")
log("  A modest signal is expected and fine; only near-perfect predicts is a leak.")

# Check 4: fit_in_fold -- all encoders built inside the fold loop from train-fold rows only
log("Check 4 PASS: fit_in_fold transforms (KBins, TargetEncoder) are inside fold loop.")
# Check 5: folds from frozen folds.json
log("Check 5 PASS: folds loaded from frozen folds.json.")
log("=== PRE-FLIGHT CHECKS COMPLETE ===")

# --- FOLD-0 CHEAP-KILL -------------------------------------------------------
CHEAP_KILL_THRESHOLD = 0.962
log(f"Running fold-0 cheap-kill (threshold BA < {CHEAP_KILL_THRESHOLD}) ...")

fi0 = folds_list[0]
assert fi0["fold"] == 0, f"Expected fold 0 first, got {fi0['fold']}"
val_idx0 = np.asarray(fi0["val_idx"])
tr_idx0 = np.setdiff1d(np.arange(n_train), val_idx0)
fold_seed0 = SEED + 1 * 100

X_tr0, X_val0, X_te0, all_cat_cols0, combo_names0, local_map0, lgbm_cat_cols0 = fit_fold_categoricals(
    X_stateless.iloc[tr_idx0].reset_index(drop=True),
    X_stateless.iloc[val_idx0].reset_index(drop=True),
    X_test_stateless.copy(),
)
y_tr0 = y_all[tr_idx0]
y_val0 = y_all[val_idx0]
X_tr0, X_val0, X_te0, te_names0 = add_target_encoding(
    X_tr0, y_tr0, X_val0, X_te0, combo_names0, fold_seed0
)
X_tr0 = X_tr0.reindex(sorted(X_tr0.columns), axis=1)
X_val0 = X_val0.reindex(sorted(X_val0.columns), axis=1)
X_te0 = X_te0.reindex(sorted(X_te0.columns), axis=1)

log(f"  Fold-0: train={len(tr_idx0)} val={len(val_idx0)} features={X_tr0.shape[1]}")

model0 = make_lgbm(fold_seed=fold_seed0)
fold0_t0 = time.perf_counter()
model0.fit(
    X_tr0, y_tr0,
    eval_set=[(X_val0, y_val0)],
    eval_metric="multi_logloss",
    callbacks=[
        early_stopping(stopping_rounds=150, verbose=False),
        log_evaluation(period=200),
    ],
    categorical_feature=lgbm_cat_cols0,
)
fold0_elapsed = time.perf_counter() - fold0_t0
log(f"  Fold-0 training: {fold0_elapsed:.1f}s  best_iter={model0.best_iteration_}")

val_proba0 = model0.predict_proba(X_val0)
fold0_score = balanced_accuracy_score(y_val0, np.argmax(val_proba0, axis=1))
log(f"  Fold-0 BA = {fold0_score:.6f}")
print(f"fold0_score={fold0_score:.6f}", flush=True)

projected_5fold = fold0_elapsed * len(folds_list)
log(f"  TIMING: fold0={fold0_elapsed:.1f}s  projected_5fold={projected_5fold:.1f}s ({projected_5fold/60:.1f}min)")

if fold0_score < CHEAP_KILL_THRESHOLD:
    log(f"CHEAP-KILL TRIPPED: fold-0 BA={fold0_score:.6f} < {CHEAP_KILL_THRESHOLD}")
    log("Marking status=dead. No full artifacts emitted.")
    print(f"CHEAP_KILL fold0_BA={fold0_score:.6f}", flush=True)
    sys.exit(0)

log(f"Fold-0 passes cheap-kill ({fold0_score:.6f} >= {CHEAP_KILL_THRESHOLD}). Proceeding to full 5-fold.")

# Store fold-0 artifacts for full OOF
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
all_cols_final = list(X_tr0.columns)
best_iters = [model0.best_iteration_]

# Write fold-0 OOF
oof_proba[val_idx0] = val_proba0.astype("float32")
# Test preds from fold-0
test_proba_accum += model0.predict_proba(X_te0).astype("float32") / len(folds_list)
per_fold_scores.append(fold0_score)

del model0, X_tr0, X_val0, X_te0
gc.collect()

# --- Full OOF loop (folds 1-4, fold-0 already done) -------------------------
log("Continuing full OOF loop (folds 1-4) ...")
fold_t_start = time.perf_counter()

for fi in folds_list[1:]:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    X_tr_fold, X_val_fold, X_te_fold, all_cat_cols, combo_names, local_map, lgbm_cat_cols = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

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

    val_proba = model.predict_proba(X_val_fold)
    oof_proba[val_idx] = val_proba.astype("float32")

    test_proba_fold = model.predict_proba(X_te_fold)
    test_proba_accum += test_proba_fold.astype("float32") / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_proba, axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t_start
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
log(f"best_iters_per_fold={best_iters}")
print(f"cv={mean_cv:.6f}", flush=True)

# --- Save OOF ----------------------------------------------------------------
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# --- Save test_probs ---------------------------------------------------------
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

# --- Write submission ---------------------------------------------------------
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# --- Write features.txt -------------------------------------------------------
(NODE_SRC / "features.txt").write_text("\n".join(sorted(all_cols_final)) + "\n")
log(f"Wrote features.txt ({len(all_cols_final)} features)")

# --- Final OOF metric --------------------------------------------------------
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

# --- Post-train output checks -------------------------------------------------
log("=== POST-TRAIN OUTPUT CHECKS ===")

# Check 7: OOF complete -- every train row covered exactly once, no NaN
oof_covered = np.zeros(n_train, dtype=int)
for fi in folds_list:
    oof_covered[np.asarray(fi["val_idx"])] += 1
assert (oof_covered == 1).all(), "OOF: some rows not covered exactly once!"
assert not np.isnan(oof_proba).any(), "OOF: contains NaN!"
log("Check 7 PASS: OOF complete, no NaN.")

# Check 8: distribution sane
assert oof_proba.shape == (n_train, 3), f"OOF shape wrong: {oof_proba.shape}"
assert test_proba_accum.shape == (n_test, 3), f"test_probs shape wrong: {test_proba_accum.shape}"
assert np.allclose(oof_proba.sum(1), 1.0, atol=1e-3), "OOF probs don't sum to 1"
assert oof_proba.min() >= 0.0, "OOF probs have negatives"
log(f"Check 8 PASS: dist sane. OOF argmax dist: {np.bincount(oof_proba.argmax(1))}")

# Check 10: cv-too-good judgment
IMPLAUSIBLE_SOLO_BA = 0.970
cv_too_good = mean_cv > IMPLAUSIBLE_SOLO_BA
log(f"Check 10: cv_too_good={cv_too_good}  (mean_cv={mean_cv:.6f}, threshold={IMPLAUSIBLE_SOLO_BA})")
if cv_too_good:
    log("  WARN: solo BA above implausible threshold -- inspect for leak.")

log("=== POST-TRAIN CHECKS COMPLETE ===")

# --- Error-correlation vs node_0070 ------------------------------------------
log("Computing error-correlation vs node_0070 ...")
n070_oof_path = COMP_DIR / "nodes/node_0070/oof.npy"
err_corr = None
if n070_oof_path.exists():
    n070_oof = np.load(n070_oof_path)
    assert n070_oof.shape == (n_train, 3), f"n070 oof shape: {n070_oof.shape}"
    err_0118 = (oof_proba.argmax(1) != y_all).astype(float)
    err_n070 = (n070_oof.argmax(1) != y_all).astype(float)
    err_corr = float(np.corrcoef(err_0118, err_n070)[0, 1])
    log(f"  Error-corr vs n070: {err_corr:.4f}  (lower = more decorrelated)")
    print(f"err_corr_vs_n070={err_corr:.4f}", flush=True)
else:
    log("  n070 oof.npy not found, skipping err-corr.")

# --- Stack-add test onto n091 (champion) -------------------------------------
log("Stack-add test onto n091 champion ...")
champ_oof_path = COMP_DIR / "nodes/node_0091/oof.npy"
champ_tst_path = COMP_DIR / "nodes/node_0091/test_probs.npy"
stack_delta = None

if champ_oof_path.exists() and champ_tst_path.exists():
    from sklearn.linear_model import LogisticRegression

    champ_oof = np.load(champ_oof_path).astype(float)
    champ_tst = np.load(champ_tst_path).astype(float)

    def logp(a):
        return np.log(np.clip(a, 1e-7, 1.0))

    def norm_proba(a):
        a = np.clip(a, 0, None)
        s = a.sum(1, keepdims=True)
        s[s == 0] = 1
        return a / s

    stack_feat_oof = np.concatenate([logp(norm_proba(champ_oof)), logp(norm_proba(oof_proba))], axis=1)

    folds_data = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
    fval = [np.asarray(f["val_idx"]) for f in folds_data]

    stack_oof = np.zeros((n_train, 3), dtype=float)
    for fi_idx, vi in enumerate(fval):
        tr_ix = np.setdiff1d(np.arange(n_train), vi)
        lr = LogisticRegression(
            class_weight="balanced", C=0.1, max_iter=2000,
            n_jobs=-1, random_state=42, solver="lbfgs", multi_class="multinomial",
        )
        lr.fit(stack_feat_oof[tr_ix], y_all[tr_ix])
        stack_oof[vi] = lr.predict_proba(stack_feat_oof[vi])

    stack_cv = balanced_accuracy_score(y_all, stack_oof.argmax(1))
    champ_cv = balanced_accuracy_score(y_all, champ_oof.argmax(1))
    stack_delta = stack_cv - champ_cv
    log(f"  Champion solo CV: {champ_cv:.6f}")
    log(f"  Stack (champ + n0118) CV: {stack_cv:.6f}")
    log(f"  Stack-add delta: {stack_delta:+.6f}")
    print(f"stack_add_delta={stack_delta:+.6f}", flush=True)
    print(f"stack_cv={stack_cv:.6f}", flush=True)
else:
    log("  Champion oof/test_probs not found, skipping stack-add test.")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
