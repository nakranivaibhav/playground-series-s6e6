"""node_0031 — draft (gbdt): XGBoost on fs_realmlp_fe features.

Built on: root (new draft). Template src copied from node_0028/src.
FE pipeline (stateless FE + fit-in-fold KBins/TargetEncoder) is KEPT BYTE-IDENTICAL.

THE ONE ATOMIC CHANGE vs node_0028:
  Replace the RealMLP model with a well-tuned XGBoost classifier.
  - tree_method='hist', device='cpu' (GPU is occupied by concurrent RealMLP node)
  - objective='multi:softprob', num_class=3
  - Encoded categoricals (category dtype → int codes, no enable_categorical needed)
  - Early stopping on the fold val split (50 rounds patience)
  - scale_pos_weight not applicable for multiclass; instead use sample_weight
    computed from class inverse frequencies to improve Balanced Accuracy
  - Sensible hyperparams: n_estimators=3000, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0

WHY: A 2nd strong tree family on the rich FE pipeline, de-correlated from both
     the RealMLP NN base and the LightGBM base (node_0030), adds diversity to
     the stack. Target: solo BA >= 0.965 (vs node_0011 XGBoost 0.964918 on simple FE).

Leakage discipline (IDENTICAL to node_0028):
  - Stateless FE (color pairs, mag stats, redshift ratio, log1p_redshift) is
    computed once on the full X/X_test dataframes -- NO target, NO cross-row stats,
    NO fitting. Safe (stateless, row-wise).
  - KBinsDiscretizer (delta bins), category factorize maps: fit on train-fold only,
    applied to val and test.
  - TargetEncoder: fit on train-fold only (using sklearn's internal CV=5 strategy),
    applied to val and test.
  - XGBoost model: fit on train-fold only, with early-stopping on val-fold.
  - frozen folds.json used throughout; no refitting of folds.

Metric: Balanced Accuracy Score (macro-average per-class recall), maximize.
Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
"""
from __future__ import annotations

import gc
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

# Use CPU for XGBoost (tree_method='hist', device='cpu' in XGB_PARAMS_BASE)
# Do NOT hide GPU (CUDA_VISIBLE_DEVICES="") — this XGBoost binary is GPU-compiled
# and requires CUDA to be accessible even when running on CPU tree_method.

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

REPO_ROOT = NODE_SRC
while not (REPO_ROOT / "tools" / "leakage_scan.py").exists():
    REPO_ROOT = REPO_ROOT.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# --- Constants ---
TARGET = "class"
IDC = "id"
DIRECTION = "maximize"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}


def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_everything(SEED)

# --- XGBoost CONFIG ---
# Tuned: lr=0.3 converges in ~300 rounds (7s/fold, ~1min for 5 folds on CPU).
# depth=6 beats depth=7 on fold-0 BA (0.9667 vs 0.9660).
XGB_PARAMS_BASE = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "tree_method": "hist",
    "device": "cpu",
    "max_depth": 6,
    "learning_rate": 0.3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "gamma": 0.0,
    "verbosity": 0,
    "nthread": -1,
}
N_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 20

# --- Feature engineering globals ---
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
    Pure row-wise / stateless feature engineering -- safe to apply to the full
    dataframe before any fold split. No fitting, no target, no cross-row stats.
    IDENTICAL to node_0028.
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
    Called INSIDE the fold loop -- fit_in_fold.
    IDENTICAL to node_0028.
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

    # Original categorical columns (spectral_type, galaxy_population)
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
        for dset, dset_tr in [(va, df_val), (te, df_te)]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32").astype("category")

    # Delta quantile bins (100 and 500) -- fit_in_fold via KBinsDiscretizer
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
    """
    TargetEncoder fit on train fold only (fit_in_fold), transform val and test.
    Returns modified copies and the list of new TE column names.
    IDENTICAL to node_0028.
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


def df_to_xgb_matrix(df: pd.DataFrame, cat_cols: list, label=None, sample_weight=None) -> xgb.DMatrix:
    """
    Convert a DataFrame to XGBoost DMatrix.
    Category columns are cast to int32 codes; numeric cols to float32.
    """
    df_copy = df.copy()
    for col in cat_cols:
        if str(df_copy[col].dtype) == "category":
            df_copy[col] = df_copy[col].cat.codes.astype("int32")
        else:
            df_copy[col] = df_copy[col].astype("int32")
    for col in df_copy.columns:
        if col not in cat_cols:
            df_copy[col] = df_copy[col].astype("float32")

    kwargs = {}
    if label is not None:
        kwargs["label"] = label
    if sample_weight is not None:
        kwargs["weight"] = sample_weight

    return xgb.DMatrix(df_copy, **kwargs)


# --- Load data ---
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)

# --- Stateless FE (computed once, safe) ---
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# --- OOF loop ---
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
cat_cols_final = None  # will be set on first fold

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    seed_everything(fold_seed)

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # Categorical encoding -- fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, cat_cols, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # Target encoding -- fit_in_fold
    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # Sort columns consistently
    X_tr_fold = X_tr_fold.reindex(sorted(X_tr_fold.columns), axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold = X_te_fold.reindex(sorted(X_te_fold.columns), axis=1)

    cat_cols_sorted = sorted(cat_cols)
    if cat_cols_final is None:
        cat_cols_final = cat_cols_sorted
        num_cols_final = [c for c in X_tr_fold.columns if c not in cat_cols_sorted]
        log(f"  n_features={X_tr_fold.shape[1]}  n_cat={len(cat_cols_sorted)}  n_num={len(num_cols_final)}")

    # XGBoost: class-balanced sample weights for Balanced Accuracy
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr_fold)

    # Build DMatrix objects
    dtrain = df_to_xgb_matrix(X_tr_fold, cat_cols_sorted, label=y_tr_fold,
                               sample_weight=sample_weights)
    dval = df_to_xgb_matrix(X_val_fold, cat_cols_sorted, label=y_val_fold)
    dtest = df_to_xgb_matrix(X_te_fold, cat_cols_sorted)

    # XGBoost training with early stopping
    xgb_params = {**XGB_PARAMS_BASE, "seed": fold_seed}

    callbacks = [xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS,
                                             metric_name="mlogloss",
                                             save_best=True,
                                             maximize=False)]

    booster = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=N_ESTIMATORS,
        evals=[(dval, "val")],
        callbacks=callbacks,
        verbose_eval=False,
    )

    best_round = booster.best_iteration
    log(f"  fold {fold_id}: best_round={best_round}")

    # OOF probabilities
    val_proba = booster.predict(dval).reshape(-1, N_CLASSES).astype("float32")
    oof_proba[val_idx] = val_proba

    # Test predictions -- average across folds
    test_proba_fold = booster.predict(dtest).reshape(-1, N_CLASSES).astype("float32")
    test_proba_accum += test_proba_fold / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, np.argmax(val_proba, axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del booster, dtrain, dval, dtest, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# --- Save OOF ---
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# --- Save test_probs ---
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

# --- Write submission ---
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# --- Write features.txt ---
all_features = sorted(num_cols_final + cat_cols_final)
(NODE_SRC / "features.txt").write_text("\n".join(all_features) + "\n")
log(f"Wrote features.txt ({len(all_features)} features)")

# --- Final OOF metric ---
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
