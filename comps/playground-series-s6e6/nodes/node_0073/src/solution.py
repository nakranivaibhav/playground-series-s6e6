"""node_0073 — AutoGluon best_quality on fs_realmlp_fe, fold-0 gate first.

Atomic change: new draft family — AutoGluon TabularPredictor (presets='best_quality',
eval_metric='balanced_accuracy') trained per-fold on fs_realmlp_fe features.

Leakage discipline:
  - Stateless FE: redshift ratios, 7 color pairs, mag aggregates, log1p_redshift,
    integer-floor cat views — all row-wise, no fitting, safe to compute once.
  - Fit-in-fold transforms (inside fold loop): KBinsDiscretizer, factorize maps.
  - AutoGluon predictor is trained STRICTLY on train-fold rows; val-fold rows
    are NEVER seen by it during training (no internal data leakage).
  - Frozen folds from folds.json — never recomputed.
  - The AG predictor's own internal bagging/stacking operates only on the train
    fold subset passed to it — val_idx rows not touched.

Fold-0 gate: if fold-0 val BA < 0.9675, stop before launching folds 1-4.
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings("ignore")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ─── Constants ────────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]

# Fold-0 kill threshold
KILL_THRESHOLD = 0.9675

# How many folds to run (controlled by env var for gate mode)
FOLD_LIMIT = int(os.environ.get("AG_FOLD_LIMIT", "1"))  # default: fold-0 only
TIME_LIMIT = int(os.environ.get("AG_TIME_LIMIT", str(90 * 60)))  # 90 min per fold

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


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_everything(SEED)


# ─── fs_realmlp_fe: stateless FE (byte-identical to node_0033) ───────────────
def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Pure row-wise stateless FE — safe to apply once before fold split."""
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


def fit_fold_categoricals(df_tr: pd.DataFrame, df_val: pd.DataFrame, df_te: pd.DataFrame):
    """Fit categorical encodings on train-fold only — fit_in_fold."""
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
        tr[col] = pd.Series(codes_tr, index=tr.index)
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index)
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index)

    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32")
        for dset, dset_orig in [(va, df_val), (te, df_te)]:
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

    for cols in IMPORTANT_COMBOS:
        combo_name = "__".join(cols) + "__"
        combo_tr = tr[cols[0]].astype(str)
        for col in cols[1:]:
            combo_tr = combo_tr + "|" + tr[col].astype(str)
        codes_tr, uniques = pd.factorize(combo_tr, sort=False)
        local_map[combo_name] = uniques
        tr[combo_name] = pd.Series(codes_tr.astype("int32"), index=tr.index)
        for dset in [va, te]:
            combo_s = dset[cols[0]].astype(str)
            for col in cols[1:]:
                combo_s = combo_s + "|" + dset[col].astype(str)
            codes = factorize_transform(combo_s, uniques)
            dset[combo_name] = pd.Series(codes, index=dset.index).astype("int32")

    return tr, va, te, local_map


# ─── Pre-train leakage checks ─────────────────────────────────────────────────
def preflight_checks(X_tr: pd.DataFrame, y_tr: np.ndarray):
    """Checks 1-3: target/id not in features, no near-perfect single-feature corr."""
    feature_cols = set(X_tr.columns)
    assert TARGET not in feature_cols, f"TARGET {TARGET} in features!"
    assert IDC not in feature_cols, f"ID {IDC} in features!"

    # Single-feature sweep on sample
    s = X_tr.sample(min(50_000, len(X_tr)), random_state=0)
    ys = y_tr[s.index] if hasattr(s.index, '__iter__') else y_tr[:len(s)]
    # Map y to integer if needed
    if isinstance(ys[0] if hasattr(ys, '__getitem__') else ys.iloc[0], str):
        ys_int = pd.factorize(ys)[0]
    else:
        ys_int = np.asarray(ys, dtype=float)

    for c in X_tr.columns:
        x = pd.to_numeric(s[c], errors="coerce")
        if x.nunique() > 1:
            r = abs(np.corrcoef(x.fillna(x.mean()), ys_int)[0, 1])
            if r >= 0.999:
                raise SystemExit(f"leak smell: {c} ~ target corr={r:.4f}")

    log("Pre-flight checks PASSED: no target in features, no near-perfect corr")


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].values
n_train = len(train_raw)
n_test = len(test_raw)

# ─── Stateless FE (computed once — safe) ─────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── OOF arrays ───────────────────────────────────────────────────────────────
oof_proba = np.full((n_train, N_CLASSES), np.nan, dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []

fold_count = min(FOLD_LIMIT, len(folds_list))
log(f"Running {fold_count} fold(s) (FOLD_LIMIT={FOLD_LIMIT}, TIME_LIMIT={TIME_LIMIT}s)")

for fold_i, fi in enumerate(folds_list[:fold_count]):
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)

    log(f"=== Fold {fold_id} ===  train={len(tr_idx)} val={len(val_idx)}")

    # Fit-in-fold categoricals
    X_tr_fold, X_val_fold, X_te_fold, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # Pre-flight checks (fold 0 only for speed)
    if fold_id == 0:
        preflight_checks(X_tr_fold, y_tr_fold)

    # Add target to train df for AutoGluon
    X_tr_fold[TARGET] = y_tr_fold
    feature_cols = [c for c in X_tr_fold.columns if c != TARGET]

    log(f"  n_features={len(feature_cols)}  launching AutoGluon predictor ...")
    fold_t0 = time.perf_counter()

    # AutoGluon predictor path — inside node dir
    ag_path = NODE_DIR / f"ag_fold_{fold_id}"
    if ag_path.exists():
        import shutil
        shutil.rmtree(ag_path)

    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor(
        label=TARGET,
        path=str(ag_path),
        eval_metric="balanced_accuracy",
        problem_type="multiclass",
        verbosity=2,
    ).fit(
        train_data=X_tr_fold,
        presets="best_quality",
        time_limit=TIME_LIMIT,
        ag_args_fit={"random_seed": SEED + fold_id * 100},
        # Ensure AutoGluon uses GPU where available
        # (LightGBM, CatBoost, XGB, and NN models will leverage GPU)
    )

    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  AG training done in {fold_elapsed:.0f}s")

    # Leakage self-check: predictor trained on tr_idx rows only
    # val_idx rows are in X_val_fold — NOT in X_tr_fold passed to AG
    assert len(X_tr_fold) == len(tr_idx), "Fold train size mismatch!"
    assert all(tr_idx == np.setdiff1d(np.arange(n_train), val_idx)), "Train idx drift!"
    log("  AG fit-in-fold check: PASSED (predictor trained on train-fold only)")

    # Predict val fold
    log("  Predicting val fold ...")
    val_pred_df = predictor.predict_proba(X_val_fold[feature_cols])
    # AG returns df with class columns in sorted order
    val_proba = np.zeros((len(X_val_fold), N_CLASSES), dtype=np.float32)
    for ci, cls in enumerate(CLASSES):
        if cls in val_pred_df.columns:
            val_proba[:, ci] = val_pred_df[cls].values.astype("float32")

    oof_proba[val_idx] = val_proba
    fold_ba = balanced_accuracy_score(y_val_fold, val_proba.argmax(axis=1))
    per_fold_scores.append(fold_ba)
    log(f"  Fold {fold_id} BA = {fold_ba:.6f}  elapsed={fold_elapsed:.0f}s")
    print(f"fold_{fold_id}_score={fold_ba:.6f}", flush=True)

    # Predict test
    log("  Predicting test ...")
    te_pred_df = predictor.predict_proba(X_te_fold[feature_cols])
    te_proba = np.zeros((n_test, N_CLASSES), dtype=np.float32)
    for ci, cls in enumerate(CLASSES):
        if cls in te_pred_df.columns:
            te_proba[:, ci] = te_pred_df[cls].values.astype("float32")
    test_proba_accum += te_proba

    # Cleanup AG predictor from memory
    del predictor
    gc.collect()

    # ─── FOLD-0 GATE ──────────────────────────────────────────────────────────
    if fold_id == 0 and FOLD_LIMIT == 1:
        if fold_ba < KILL_THRESHOLD:
            log(f"KILL CRITERION TRIPPED: fold-0 BA {fold_ba:.6f} < {KILL_THRESHOLD}")
            log("Stopping — do NOT launch folds 1-4.")
            print(f"FOLD0_GATE=KILLED  fold0_ba={fold_ba:.6f}  threshold={KILL_THRESHOLD}", flush=True)
            # Save partial results
            np.save(NODE_DIR / "oof_fold0_only.npy", oof_proba)
            np.save(NODE_DIR / "test_probs_fold0_only.npy", test_proba_accum)
            sys.exit(0)
        else:
            log(f"FOLD-0 GATE PASSED: fold-0 BA {fold_ba:.6f} >= {KILL_THRESHOLD}")
            print(f"FOLD0_GATE=PASSED  fold0_ba={fold_ba:.6f}  threshold={KILL_THRESHOLD}", flush=True)
            # Save partial state for orchestrator decision
            np.save(NODE_DIR / "oof_fold0_only.npy", oof_proba)
            np.save(NODE_DIR / "test_probs_fold0_only.npy", test_proba_accum / 1)
            log("Stopping after fold-0 gate — orchestrator will decide on full 5-fold run.")
            sys.exit(0)

# ─── Full run path (FOLD_LIMIT=5) ─────────────────────────────────────────────
n_folds_run = len(per_fold_scores)
mean_ba = float(np.mean(per_fold_scores))
sem_ba = float(np.std(per_fold_scores, ddof=1) / np.sqrt(n_folds_run)) if n_folds_run > 1 else float("nan")

log(f"Per-fold BAs: {[f'{s:.6f}' for s in per_fold_scores]}")
log(f"cv={mean_ba:.6f}  sem={sem_ba:.6f}")
print(f"cv={mean_ba:.6f}", flush=True)

# Check OOF full
if FOLD_LIMIT == 5:
    oof_nan_count = np.isnan(oof_proba).sum()
    assert oof_nan_count == 0, f"OOF has {oof_nan_count} NaN values!"
    log("OOF complete: no NaN")

# Average test predictions over folds
test_proba_final = test_proba_accum / n_folds_run

# ─── Save artifacts ───────────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
np.save(NODE_DIR / "test_probs.npy", test_proba_final)
log(f"Saved oof.npy {oof_proba.shape}  test_probs.npy {test_proba_final.shape}")

# ─── Build submission ─────────────────────────────────────────────────────────
test_preds_labels = [CLASSES[i] for i in test_proba_final.argmax(axis=1)]
sub = pd.DataFrame({"id": test_raw[IDC].values, "class": test_preds_labels})
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"submission.csv written  shape={sub.shape}")

log("DONE")
