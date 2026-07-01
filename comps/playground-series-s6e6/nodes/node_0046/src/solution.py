"""node_0046 — pseudo-label self-training (GBDT bases on fs_realmlp_fe).

THE ONE ATOMIC CHANGE vs node_0041 (champion stack):
  In-distribution self-training. Use the champion's test predictions (node_0041)
  to pick HIGH-CONFIDENCE test rows (max class prob >= 0.99, class-balanced cap
  = min(per-class count)), treat them as extra labeled rows, and retrain:
    - LightGBM base (node_0030 recipe)
    - CatBoost base (node_0039 recipe)
  on (real train_fold + pseudo-labeled test rows). Val folds stay PURE real rows.
  Then re-stack: swap node_0030 -> pseudo-lgbm, node_0039 -> pseudo-cat in CORE15,
  run balanced-LogReg meta + DE threshold. Report stacked CV vs champion 0.969808.

LEAKAGE RIGOR:
  - Pseudo-labels live ONLY on test rows (no true labels there).
  - OOF is computed on PURE real train val rows only (no pseudo rows).
  - The pseudo-labels come from node_0041 which was trained on full train; this
    introduces mild val-optimism (the pseudo-labels saw the val targets indirectly
    through the full-train fit). This is flagged below. For production honesty,
    per-fold pseudo-labels (label test with each fold's own base model) would be
    cleaner; here we use fixed champion pseudo-labels for simplicity.
  - All transforms (TargetEncoder, KBins, factorize) fit on train-fold-only rows.
    The pseudo test rows are TRANSFORMED using the fold's fitted transforms, but
    NOT used to fit them. This is correct — test rows are unlabeled in reality.
  - Stateless FE computed once on full X / X_test (no target, no cross-row).

Outputs:
  oof_lgbm.npy (577347,3), test_lgbm.npy (247435,3)
  oof_cat.npy  (577347,3), test_cat.npy  (247435,3)
  submission.csv (stacked, with DE thresholds)
  features.txt (from LightGBM fold 0)
"""
from __future__ import annotations

import gc
import json
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

REPO_ROOT = NODE_SRC
while not (REPO_ROOT / "tools" / "leakage_scan.py").exists():
    REPO_ROOT = REPO_ROOT.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def log_rss(tag: str = ""):
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_kb / 1024 / 1024
    log(f"  PEAK_RSS{' ' + tag if tag else ''}: {rss_gb:.2f} GB")


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_MAP = {v: k for k, v in LABEL_MAP.items()}

# Pseudo-label hyperparams
CONF_THRESHOLD = 0.99    # min max-class prob
# Cap per class (balanced): will be set to min across classes after filtering

# CORE15 base set — we swap node_0030->pseudo-lgbm, node_0039->pseudo-cat
CORE15_ORIG = [
    "node_0006", "node_0004", "node_0001", "node_0009", "node_0011",
    "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035", "node_0033",
    "node_0030",   # <-- will be replaced by pseudo-lgbm OOF
    "node_0039",   # <-- will be replaced by pseudo-cat OOF
]

# ─── Feature engineering ─────────────────────────────────────────────────────
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


def factorize_fit(series):
    codes, uniques = pd.factorize(series, sort=False)
    return codes.astype("int32"), uniques


def factorize_transform(series, uniques):
    code_map = {cat: i for i, cat in enumerate(uniques)}
    return series.map(code_map).fillna(-1).astype("int32")


def fit_fold_categoricals(df_tr, df_val, df_te):
    """Fit categoricals on train-fold + apply to val and te. fit_in_fold."""
    local_map: dict = {}
    tr = df_tr.copy(); va = df_val.copy(); te = df_te.copy()

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

    all_new_cols = (
        BASE_CAT_COLS
        + [f"{c}_cat_" for c in BASE_NUM_COLS]
        + [f"delta_{n}_quantile_bin_" for n in [100, 500]]
        + combo_names
    )
    all_new_cols = [c for c in all_new_cols if c in tr.columns]
    lgbm_cat_cols = BASE_CAT_COLS[:]
    return tr, va, te, all_new_cols, combo_names, local_map, lgbm_cat_cols


def add_target_encoding(X_tr, y_tr, X_val, X_te, combo_names, fold_seed):
    """TargetEncoder fit on train-fold only. fit_in_fold."""
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    try:
        encoder = TargetEncoder(target_type="multiclass", cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
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
    return LGBMClassifier(
        objective="multiclass", num_class=N_CLASSES,
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        max_depth=-1, min_child_samples=20, min_child_weight=1e-3,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, class_weight="balanced",
        n_jobs=-1, random_state=fold_seed, verbosity=-1, device="cpu",
    )


def make_catboost(seed: int = SEED) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=5000, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
        loss_function="MultiClass", eval_metric="Accuracy",
        auto_class_weights="Balanced", random_seed=seed, task_type="CPU",
        thread_count=8, od_type="Iter", od_wait=100, use_best_model=True,
        verbose=False, border_count=128,
    )


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw  = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
y_all_str = train_raw[TARGET].values
n_train = len(train_raw)
n_test = len(test_raw)

# ─── Load champion test_probs → pseudo-labels ─────────────────────────────────
log("Building pseudo-label index from node_0041 test_probs ...")
champ_test_probs = np.load(COMP_DIR / "nodes/node_0041/test_probs.npy")  # (247435, 3)
max_prob = champ_test_probs.max(axis=1)
pseudo_class = champ_test_probs.argmax(axis=1)

# High-confidence filter
conf_mask = max_prob >= CONF_THRESHOLD
log(f"  conf>=0.99: {conf_mask.sum()} / {n_test} test rows")

# Class-balanced cap: cap each class to min(per-class conf count)
per_class_conf = {c: np.where((pseudo_class == c) & conf_mask)[0] for c in range(N_CLASSES)}
per_class_cnt = {c: len(per_class_conf[c]) for c in range(N_CLASSES)}
log(f"  per-class conf counts: {per_class_cnt}")
cap = min(per_class_cnt.values())
log(f"  cap per class = {cap}")

rng = np.random.default_rng(SEED)
pseudo_idx = []
for c in range(N_CLASSES):
    idxs = per_class_conf[c]
    chosen = rng.choice(idxs, size=cap, replace=False)
    pseudo_idx.extend(chosen.tolist())
pseudo_idx = np.array(sorted(pseudo_idx))
pseudo_labels = pseudo_class[pseudo_idx]

log(f"  total pseudo rows = {len(pseudo_idx)} ({cap} per class)")
log(f"  pseudo class dist = {dict(zip(*np.unique(pseudo_labels, return_counts=True)))}")

# ─── Stateless FE ─────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw        = train_raw.drop(columns=[IDC, TARGET])
X_test_raw   = test_raw.drop(columns=[IDC])
X_stateless  = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
# The pseudo-label subset of test (stateless — safe)
X_pseudo_stateless = X_test_stateless.iloc[pseudo_idx].reset_index(drop=True)
log(f"  X_stateless={X_stateless.shape}  X_pseudo={X_pseudo_stateless.shape}")

# ─── OOF loop — LightGBM ─────────────────────────────────────────────────────
log("=== LightGBM pseudo-label OOF loop ===")
oof_lgbm  = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_lgbm = np.zeros((n_test,  N_CLASSES), dtype=np.float32)
per_fold_lgbm = []
all_cols_final = None

fold_t0 = time.perf_counter()
for fi in folds_list:
    fold_id  = fi["fold"]
    val_idx  = np.asarray(fi["val_idx"])
    tr_idx   = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"LightGBM Fold {fold_id}: real_train={len(tr_idx)} val={len(val_idx)} pseudo={len(pseudo_idx)}")

    # Categorical encoding — fit on REAL train-fold rows only
    X_tr_real = X_stateless.iloc[tr_idx].reset_index(drop=True)
    X_val_real = X_stateless.iloc[val_idx].reset_index(drop=True)
    X_te_full = X_test_stateless.copy()

    X_tr_fold, X_val_fold, X_te_fold, all_cat_cols, combo_names, local_map, lgbm_cat_cols = fit_fold_categoricals(
        X_tr_real, X_val_real, X_te_full
    )
    # Also transform pseudo rows with this fold's fitted maps (NOT used for fitting)
    # The pseudo rows come from X_te_fold (already transformed), so just index into it
    X_pseudo_fold = X_te_fold.iloc[pseudo_idx].reset_index(drop=True)

    # Target encoding — fit on REAL train-fold rows only
    y_tr_fold  = y_all[tr_idx]
    y_val_fold = y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )
    # Re-index pseudo after TE transform (X_te_fold was already modified in-place for test)
    X_pseudo_fold = X_te_fold.iloc[pseudo_idx].reset_index(drop=True)

    # Sort columns
    col_order = sorted(X_tr_fold.columns)
    X_tr_fold     = X_tr_fold[col_order]
    X_val_fold    = X_val_fold[col_order]
    X_te_fold     = X_te_fold[col_order]
    X_pseudo_fold = X_pseudo_fold[col_order]

    if all_cols_final is None:
        all_cols_final = col_order
        log(f"  n_features={len(col_order)}")

    # Append pseudo rows to training set
    y_pseudo_fold = pseudo_labels  # int labels
    X_train_aug = pd.concat([X_tr_fold, X_pseudo_fold], ignore_index=True)
    y_train_aug  = np.concatenate([y_tr_fold, y_pseudo_fold])

    model = make_lgbm(fold_seed=fold_seed)
    model.fit(
        X_train_aug, y_train_aug,
        eval_set=[(X_val_fold, y_val_fold)],
        eval_metric="multi_logloss",
        callbacks=[
            early_stopping(stopping_rounds=150, verbose=False),
            log_evaluation(period=500),
        ],
        categorical_feature=lgbm_cat_cols,
    )
    best_iter = model.best_iteration_
    log(f"  best_iteration={best_iter}")

    # OOF on PURE real val rows
    val_proba = model.predict_proba(X_val_fold)
    oof_lgbm[val_idx] = val_proba.astype("float32")

    # Test predictions average across folds
    test_proba_fold = model.predict_proba(X_te_fold)
    test_lgbm += test_proba_fold.astype("float32") / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, val_proba.argmax(1))
    per_fold_lgbm.append(fold_score)
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}")
    print(f"lgbm_fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold, X_pseudo_fold, X_train_aug
    gc.collect()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={fold_time*5:.1f}s ({fold_time*5/60:.1f}min)")

cv_lgbm = float(np.mean(per_fold_lgbm))
sem_lgbm = float(np.std(per_fold_lgbm, ddof=1) / np.sqrt(len(per_fold_lgbm)))
log(f"LightGBM CV={cv_lgbm:.6f}+/-{sem_lgbm:.6f}  folds={per_fold_lgbm}")
print(f"lgbm_cv={cv_lgbm:.6f}", flush=True)

np.save(NODE_DIR / "oof_lgbm.npy",  oof_lgbm)
np.save(NODE_DIR / "test_lgbm.npy", test_lgbm)
log("Saved oof_lgbm.npy and test_lgbm.npy")

# ─── OOF loop — CatBoost ─────────────────────────────────────────────────────
log("=== CatBoost pseudo-label OOF loop ===")
oof_cat  = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_cat = np.zeros((n_test,  N_CLASSES), dtype=np.float32)
per_fold_cat = []

fold_t0 = time.perf_counter()
for fi in folds_list:
    fold_id   = fi["fold"]
    val_idx   = np.asarray(fi["val_idx"])
    tr_idx    = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"CatBoost Fold {fold_id}: real_train={len(tr_idx)} val={len(val_idx)} pseudo={len(pseudo_idx)}")
    log_rss(f"fold{fold_id}_start")

    X_tr_real  = X_stateless.iloc[tr_idx].reset_index(drop=True)
    X_val_real = X_stateless.iloc[val_idx].reset_index(drop=True)
    X_te_full  = X_test_stateless.copy()

    X_tr_fold, X_val_fold, X_te_fold, cat_col_names, combo_names, local_map, _lgbm_cats = fit_fold_categoricals(
        X_tr_real, X_val_real, X_te_full
    )

    y_tr_fold     = y_all[tr_idx]
    y_val_fold    = y_all[val_idx]
    y_tr_fold_str = y_all_str[tr_idx]
    y_val_fold_str = y_all_str[val_idx]

    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )
    X_pseudo_fold = X_te_fold.iloc[pseudo_idx].reset_index(drop=True)

    all_feat_cols = sorted(X_tr_fold.columns.tolist())
    X_tr_fold     = X_tr_fold[all_feat_cols]
    X_val_fold    = X_val_fold[all_feat_cols]
    X_te_fold     = X_te_fold[all_feat_cols]
    X_pseudo_fold = X_pseudo_fold[all_feat_cols]

    cat_cols_sorted = sorted(cat_col_names)
    cat_feature_indices = [all_feat_cols.index(c) for c in cat_cols_sorted if c in all_feat_cols]

    # Append pseudo rows (string labels for CatBoost)
    y_pseudo_fold_str = np.array([CLASSES[c] for c in pseudo_labels])
    X_train_aug = pd.concat([X_tr_fold, X_pseudo_fold], ignore_index=True)
    y_train_aug_str = np.concatenate([y_tr_fold_str, y_pseudo_fold_str])

    # Convert int cat cols to string for CatBoost native handling
    for col in cat_cols_sorted:
        if col in X_train_aug.columns:
            X_train_aug[col] = X_train_aug[col].astype(str)
            X_val_fold[col]  = X_val_fold[col].astype(str)
            X_te_fold[col]   = X_te_fold[col].astype(str)

    train_pool = Pool(X_train_aug, label=y_train_aug_str, cat_features=cat_feature_indices)
    val_pool   = Pool(X_val_fold,  label=y_val_fold_str,  cat_features=cat_feature_indices)
    test_pool  = Pool(X_te_fold,   cat_features=cat_feature_indices)

    del X_tr_fold, X_val_fold, X_te_fold, X_pseudo_fold, X_train_aug, local_map
    gc.collect()
    log_rss(f"fold{fold_id}_after_pool")

    model = make_catboost(seed=fold_seed)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    log_rss(f"fold{fold_id}_after_fit")

    class_order = list(model.classes_)
    val_proba = model.predict_proba(val_pool)
    val_proba_aligned = np.zeros((len(val_idx), N_CLASSES), dtype=np.float32)
    for lbl in CLASSES:
        dest_col = LABEL_MAP[lbl]; src_col = class_order.index(lbl)
        val_proba_aligned[:, dest_col] = val_proba[:, src_col]
    oof_cat[val_idx] = val_proba_aligned

    test_proba = model.predict_proba(test_pool)
    test_proba_aligned = np.zeros((n_test, N_CLASSES), dtype=np.float32)
    for lbl in CLASSES:
        dest_col = LABEL_MAP[lbl]; src_col = class_order.index(lbl)
        test_proba_aligned[:, dest_col] = test_proba[:, src_col]
    test_cat += test_proba_aligned / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, val_proba_aligned.argmax(1))
    per_fold_cat.append(fold_score)
    best_iter = model.best_iteration_
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  best_iter={best_iter}")
    print(f"cat_fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, train_pool, val_pool, test_pool
    gc.collect()
    log_rss(f"fold{fold_id}_after_cleanup")

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={fold_time*5:.1f}s ({fold_time*5/60:.1f}min)")

cv_cat = float(np.mean(per_fold_cat))
sem_cat = float(np.std(per_fold_cat, ddof=1) / np.sqrt(len(per_fold_cat)))
log(f"CatBoost CV={cv_cat:.6f}+/-{sem_cat:.6f}  folds={per_fold_cat}")
print(f"cat_cv={cv_cat:.6f}", flush=True)

np.save(NODE_DIR / "oof_cat.npy",  oof_cat)
np.save(NODE_DIR / "test_cat.npy", test_cat)
log("Saved oof_cat.npy and test_cat.npy")

# ─── Re-stack: CORE15 with pseudo-lgbm + pseudo-cat swapped in ───────────────
log("=== Re-stacking CORE15 with pseudo bases ===")
nodes_dir = COMP_DIR / "nodes"


def logp(a):
    return np.log(np.clip(a, 1e-7, 1.0))


def score_fn(y_true, y_pred):
    return float(np.mean([(y_pred[y_true == c] == c).mean() for c in range(N_CLASSES) if (y_true == c).any()]))


def best_thr_de(probs, labels):
    def neg(w):
        return -score_fn(labels, np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1))
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)], maxiter=40, tol=1e-7, seed=0, polish=False, workers=1)
    return np.array([r.x[0], r.x[1], 1.0])


# Build OOF feature matrix for CORE15, swapping node_0030->oof_lgbm, node_0039->oof_cat
CORE15_SWAPPED_NODES = [
    "node_0006", "node_0004", "node_0001", "node_0009", "node_0011",
    "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035", "node_0033",
    # node_0030 -> pseudo-lgbm
    # node_0039 -> pseudo-cat
]

oof_chunks = [logp(np.load(nodes_dir / b / "oof.npy")) for b in CORE15_SWAPPED_NODES]
oof_chunks.append(logp(oof_lgbm))   # pseudo-lgbm replaces node_0030
oof_chunks.append(logp(oof_cat))    # pseudo-cat replaces node_0039
OOF = np.concatenate(oof_chunks, axis=1)  # (n_train, 15*3)
log(f"  OOF matrix shape: {OOF.shape}")

test_chunks = [logp(np.load(nodes_dir / b / "test_probs.npy")) for b in CORE15_SWAPPED_NODES]
test_chunks.append(logp(test_lgbm))
test_chunks.append(logp(test_cat))
TEST_MAT = np.concatenate(test_chunks, axis=1)
log(f"  TEST matrix shape: {TEST_MAT.shape}")

fval = [np.asarray(f["val_idx"]) for f in folds_list]
stack_oof = np.zeros((n_train, N_CLASSES), dtype=np.float32)

for vi in fval:
    tr_idx_s = np.setdiff1d(np.arange(n_train), vi)
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(OOF[tr_idx_s], y_all[tr_idx_s])
    stack_oof[vi] = m.predict_proba(OOF[vi])

# Per-fold DE threshold (fit on OTHER folds, applied to held fold)
stack_folds = []
for i, vi in enumerate(fval):
    oth = np.setdiff1d(np.arange(n_train), vi)
    w = best_thr_de(stack_oof[oth], y_all[oth])
    fold_cv = score_fn(y_all[vi], np.argmax(stack_oof[vi] * w, axis=1))
    stack_folds.append(fold_cv)
    log(f"  stack fold {i}: cv={fold_cv:.6f}  w={w}")

cv_stack = float(np.mean(stack_folds))
sem_stack = float(np.std(stack_folds, ddof=1) / np.sqrt(len(stack_folds)))
log(f"Stack CV={cv_stack:.6f}+/-{sem_stack:.6f}  folds={stack_folds}")
print(f"cv={cv_stack:.6f}", flush=True)

# Full-train stack for submission
meta_full = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
meta_full.fit(OOF, y_all)
test_stack = meta_full.predict_proba(TEST_MAT)
w_full = best_thr_de(test_stack, y_all[:0])  # dummy — use full OOF for threshold

# Actually use threshold fit on full OOF stack
w_full = best_thr_de(stack_oof, y_all)
pred_labels = np.argmax(test_stack * w_full, axis=1)
pred_str = np.array([CLASSES[i] for i in pred_labels])

# ─── Write submission ─────────────────────────────────────────────────────────
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_str})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class dist:\n{sub[TARGET].value_counts().to_string()}")

# ─── Write features.txt ───────────────────────────────────────────────────────
(NODE_SRC / "features.txt").write_text("\n".join(sorted(all_cols_final)) + "\n")
log(f"Wrote features.txt ({len(all_cols_final)} features)")

# ─── OOF full metric (LightGBM base — main cv) ───────────────────────────────
oof_lgbm_metric = balanced_accuracy_score(y_all, oof_lgbm.argmax(1))
oof_cat_metric  = balanced_accuracy_score(y_all, oof_cat.argmax(1))
log(f"OOF full balanced_accuracy — lgbm={oof_lgbm_metric:.6f}  cat={oof_cat_metric:.6f}")
log(f"Stack CV={cv_stack:.6f}  (champion baseline=0.969808)")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
