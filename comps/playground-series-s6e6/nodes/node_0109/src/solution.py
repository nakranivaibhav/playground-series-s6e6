"""node_0109 — draft (gbdt): LightGBM on fs_flux_rich features.

THE ONE ATOMIC CHANGE vs node_0030:
  Replace the fs_realmlp_fe feature matrix with fs_flux_rich — linear fluxes,
  pairwise flux ratios, SED simplex, flux aggregates, flux×redshift interactions,
  raw redshift + log1p(redshift), and 2 categoricals. No log-colors, no raw mags.

LightGBM config (byte-identical to node_0030):
  - 2 categoricals (spectral_type, galaxy_population) passed as native LightGBM
    categorical features (low cardinality, proper treatment).
  - class_weight="balanced" for Balanced Accuracy (macro recall)
  - n_estimators=2000, learning_rate=0.05, num_leaves=127
  - early stopping on fold val (150 rounds)
  - CPU mode, n_jobs=-1

GATE LOGIC:
  - Fold-0 only first; solo BA + err-corr vs node_0070 bank
  - BA < 0.965 OR err-corr >= 0.65 → STOP, record, no more folds
  - Both pass → run all 5 folds, full CV/sem/folds

Leakage discipline:
  - fs_flux_rich is stateless — no fit, no target, no cross-row stats.
    Computed once on the full dataframe, safe.
  - Categorical factorize maps: fit on train-fold only, applied to val and test.
    (fit_in_fold)
  - Frozen folds.json used throughout.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv.
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

warnings.filterwarnings("ignore")

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

T0 = time.perf_counter()

# Gate mode: fold-0 first, then all folds if gates pass
FOLD0_BA_GATE = 0.965
FOLD0_ERRCORR_GATE = 0.65
N070_OOF_PATH = COMP_DIR / "nodes/node_0070/oof.npy"


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

# ─── fs_flux_rich feature engineering (VERBATIM from node_0107) ───────────────
MAG_BANDS = ["u", "g", "r", "i", "z"]
BAND_PAIRS = [(a, b) for i, a in enumerate(MAG_BANDS) for b in MAG_BANDS[i + 1:]]  # 10 pairs
BASE_CAT_COLS = ["spectral_type", "galaxy_population"]


def add_flux_rich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure row-wise stateless flux-space features (fs_flux_rich).
    No fit, no target, no cross-row stats. No raw magnitudes, no log colors.

    Numeric features (49 total):
      - 5  linear fluxes  f_b = 10^(-0.4*(mag_b - mag_mean))
      - 10 pairwise flux ratios f_b/f_b'
      - 5  unit-sum SED simplex
      - 2  flux aggregates: flux mean, flux range
      - 5  brightest-band one-hot
      - 5  faintest-band one-hot
      - 5  flux x redshift interactions
      - 10 flux-ratio x redshift interactions
      - 2  raw redshift + log1p(redshift)
    Categorical features (kept from raw df, passed through as-is):
      - spectral_type, galaxy_population
    """
    df = df.copy()

    mags = df[MAG_BANDS].astype(np.float32).values  # (N, 5)
    mag_mean = mags.mean(axis=1, keepdims=True)  # (N, 1)
    rel_mags = mags - mag_mean  # (N, 5)

    # Linear fluxes
    fluxes = np.power(10.0, -0.4 * rel_mags).astype(np.float32)  # (N, 5)

    result = {}

    # 5 fluxes
    for j, b in enumerate(MAG_BANDS):
        result[f"_flux_{b}"] = fluxes[:, j]

    # 10 pairwise flux ratios
    for a, b in BAND_PAIRS:
        fa = fluxes[:, MAG_BANDS.index(a)]
        fb = fluxes[:, MAG_BANDS.index(b)]
        ratio = np.clip(fa / (fb + 1e-9), -1e6, 1e6).astype(np.float32)
        result[f"_fratio_{a}_{b}"] = ratio

    # 5 SED simplex (unit-sum)
    flux_sum = fluxes.sum(axis=1, keepdims=True) + 1e-9
    simplex = (fluxes / flux_sum).astype(np.float32)
    for j, b in enumerate(MAG_BANDS):
        result[f"_fsimplex_{b}"] = simplex[:, j]

    # 2 flux aggregates: mean and range
    result["_flux_mean"] = fluxes.mean(axis=1).astype(np.float32)
    result["_flux_range"] = (fluxes.max(axis=1) - fluxes.min(axis=1)).astype(np.float32)

    # 5 brightest-band one-hot
    brightest = np.argmax(fluxes, axis=1)  # (N,)
    for j, b in enumerate(MAG_BANDS):
        result[f"_bright_{b}"] = (brightest == j).astype(np.float32)

    # 5 faintest-band one-hot
    faintest = np.argmin(fluxes, axis=1)  # (N,)
    for j, b in enumerate(MAG_BANDS):
        result[f"_faint_{b}"] = (faintest == j).astype(np.float32)

    # redshift
    redshift = df["redshift"].astype(np.float32).values  # (N,)
    result["_redshift"] = redshift
    result["_log1p_redshift"] = np.log1p(np.clip(redshift, -1 + 1e-6, None)).astype(np.float32)

    # 5 flux x redshift interactions
    for j, b in enumerate(MAG_BANDS):
        result[f"_flux_z_{b}"] = (fluxes[:, j] * redshift).astype(np.float32)

    # 10 flux-ratio x redshift interactions
    for a, b in BAND_PAIRS:
        ratio_vals = result[f"_fratio_{a}_{b}"]
        result[f"_fratioZ_{a}_{b}"] = (ratio_vals * redshift).astype(np.float32)

    # Build result df (numeric only — no mags)
    num_df = pd.DataFrame(result, index=df.index)

    # Keep the two engineered categoricals from original df
    cat_df = df[BASE_CAT_COLS].copy() if all(c in df.columns for c in BASE_CAT_COLS) else pd.DataFrame(index=df.index)

    return pd.concat([num_df, cat_df], axis=1)


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """
    Factorize the two categoricals on train-fold only.
    Returns (df_tr, df_val, df_te, lgbm_cat_cols).
    Called INSIDE the fold loop — fit_in_fold.
    """
    local_map = {}

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

    # Convert to LightGBM category dtype
    for col in BASE_CAT_COLS:
        tr[col] = tr[col].astype("category")
        va[col] = va[col].astype("category")
        te[col] = te[col].astype("category")

    return tr, va, te, BASE_CAT_COLS[:]


def make_lgbm(fold_seed: int) -> LGBMClassifier:
    """
    Well-tuned LightGBM — byte-identical params to node_0030.
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


def compute_err_corr(oof_proba_node: np.ndarray, oof_n070: np.ndarray, y_all: np.ndarray) -> float:
    """
    Compute mean per-class error correlation vs node_0070 bank.
    Both arrays must be (n_train, 3); errors = 1 - p[true_class].
    """
    errors_mine = np.array([
        1.0 - oof_proba_node[i, y_all[i]] for i in range(len(y_all))
    ], dtype=np.float32)
    errors_n070 = np.array([
        1.0 - oof_n070[i, y_all[i]] for i in range(len(y_all))
    ], dtype=np.float32)
    return float(np.corrcoef(errors_mine, errors_n070)[0, 1])


# ─── PRE-FLIGHT LEAKAGE CHECKS ───────────────────────────────────────────────
log("Pre-flight leakage checks ...")

# Check 1+2: target/id will be dropped; verify they won't be in features (structural check)
# (We drop TARGET and IDC before FE, confirmed below)

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

# ─── Stateless FE (computed once, safe — no target, no cross-row stats) ───────
log("Applying stateless flux FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = add_flux_rich_features(X_raw)
X_test_stateless = add_flux_rich_features(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# Pre-flight check 1+2: target and id not in features
assert TARGET not in X_stateless.columns, f"TARGET {TARGET} leaked into features!"
assert IDC not in X_stateless.columns, f"ID {IDC} leaked into features!"
log("  leak check 1+2: target/id not in features — OK")

# Pre-flight check 3: single-feature↔target sweep on sample
log("  leak check 3: single-feature sweep ...")
sample_size = min(50_000, n_train)
rng = np.random.default_rng(42)
sample_idx = rng.choice(n_train, sample_size, replace=False)
s = X_stateless.iloc[sample_idx]
ys = y_all[sample_idx]
for c in X_stateless.columns:
    if c in BASE_CAT_COLS:
        continue
    x = pd.to_numeric(s[c], errors="coerce")
    if x.nunique() > 1:
        corr = abs(np.corrcoef(x.fillna(x.mean()).values, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK: {c} corr={corr:.4f} with target")
log("  leak check 3: no near-perfect single-feature↔target corr — OK")

# Pre-flight check 4: fit_in_fold verified by code read — only factorize maps
#   inside fold loop. No transform fitted on full train before fold loop.
log("  leak check 4: confirmed by code — factorize maps fit inside fold loop only — OK")

# Pre-flight check 5: folds from frozen folds.json — OK (loaded above)
log("  leak check 5: frozen folds.json loaded — OK")

# Pre-flight check 6: near-dup warning — flux features are deterministic; not image/text
log("  leak check 6: stateless deterministic features, not image/text — skip dup scan")

log("Pre-flight checks PASSED — launching training")

# ─── Load node_0070 OOF for err-corr gate ─────────────────────────────────────
oof_n070 = np.load(N070_OOF_PATH)
log(f"Loaded node_0070 oof: {oof_n070.shape}")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
best_iters = []

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

GATE_TRIPPED = False
GATE_REASON = ""

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"], dtype=int)
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # Categorical encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, lgbm_cat_cols = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # Sort columns consistently
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    if fold_id == 0:
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

    del model
    gc.collect()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

        # GATE: fold-0 BA check
        if fold_score < FOLD0_BA_GATE:
            GATE_TRIPPED = True
            GATE_REASON = f"fold-0 BA={fold_score:.6f} < gate {FOLD0_BA_GATE}"
            log(f"  GATE TRIPPED: {GATE_REASON}")
            break

        # GATE: fold-0 err-corr check (only on val_idx portion of oof)
        err_corr_fold0 = compute_err_corr(
            oof_proba[val_idx],
            oof_n070[val_idx],
            y_all[val_idx]
        )
        log(f"  fold-0 err-corr vs node_0070: {err_corr_fold0:.4f} (gate < {FOLD0_ERRCORR_GATE})")
        print(f"fold0_err_corr={err_corr_fold0:.4f}", flush=True)

        if err_corr_fold0 >= FOLD0_ERRCORR_GATE:
            GATE_TRIPPED = True
            GATE_REASON = f"fold-0 err-corr={err_corr_fold0:.4f} >= gate {FOLD0_ERRCORR_GATE}"
            log(f"  GATE TRIPPED: {GATE_REASON}")
            break

        log(f"  Both fold-0 gates PASS (BA={fold_score:.6f} >= {FOLD0_BA_GATE}, "
            f"err-corr={err_corr_fold0:.4f} < {FOLD0_ERRCORR_GATE}) — proceeding to all folds")

    del X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

if GATE_TRIPPED:
    log(f"STOPPED EARLY — gate tripped: {GATE_REASON}")
    print(f"GATE_TRIPPED: {GATE_REASON}", flush=True)
    mean_cv = float(np.mean(per_fold_scores)) if per_fold_scores else None
    print(f"cv={mean_cv}", flush=True)
    sys.exit(0)

# ─── Full CV metrics ──────────────────────────────────────────────────────────
mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
log(f"best_iters_per_fold={best_iters}")
print(f"cv={mean_cv:.6f}", flush=True)

# Full err-corr over all training rows
full_err_corr = compute_err_corr(oof_proba, oof_n070, y_all)
log(f"Full OOF err-corr vs node_0070: {full_err_corr:.4f}")
print(f"full_err_corr={full_err_corr:.4f}", flush=True)

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

# ─── Final OOF metric ────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
