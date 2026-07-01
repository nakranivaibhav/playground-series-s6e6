"""node_0039 — CatBoost on fs_realmlp_fe features (CPU) — memory-fixed debug of node_0034.

THE ONE ATOMIC CHANGE vs node_0034:
  Fix the OOM kill (node_0034 died at fold 4, 29.5 GB RSS with depth=8, border_count=254).
  Two reductions to cut per-split histogram memory:
    - border_count 254 → 128  (halves per-feature histogram RAM)
    - depth 8 → 6             (reduces split-candidate memory per tree)
  Additionally: explicit del + gc.collect() of CatBoost Pools and the model after
  each fold to aggressively free memory before the next fold starts.
  Peak RSS is logged at each fold via resource.getrusage so any remaining pressure
  is visible. Everything else is byte-identical to node_0034.

Feature-set fs_realmlp_fe (unchanged from node_0034 / node_0028):
  - Stateless: redshift ratios (g/z, i/z), log1p_redshift, 7 color pairs,
    mag_mean, mag_range over u/g/r/i/z.
  - fit_in_fold: integer-floor categorical views of every base numeric,
    delta quantile bins (100 and 500 bins) via KBinsDiscretizer,
    interaction cross-combos (alpha_cat_ x delta_cat_, u_cat_ x z_cat_),
    TargetEncoder on combo cats (cv=5, fit on train-fold only).

Leakage discipline:
  - Stateless FE: computed once on full X/X_test (no target, no cross-row,
    no fitting) — safe.
  - KBinsDiscretizer, factorize maps, TargetEncoder: fit on train-fold rows
    only, applied to val and test.
  - CatBoost early-stopping uses only the fold's own val split — fold-honest.
  - Frozen folds.json throughout; no refitting of folds.

Metric: Balanced Accuracy Score (macro-average per-class recall), maximize.
Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
"""
from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

REPO_ROOT = NODE_SRC
while not (REPO_ROOT / "tools" / "leakage_scan.py").exists():
    REPO_ROOT = REPO_ROOT.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def log_rss(tag: str = ""):
    """Log peak RSS in GB (Linux: ru_maxrss is in KB)."""
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_gb = rss_kb / 1024 / 1024
    log(f"  PEAK_RSS{' ' + tag if tag else ''}: {rss_gb:.2f} GB")
    return rss_gb


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

    # Original categorical columns (spectral_type, galaxy_population)
    for col in BASE_CAT_COLS:
        codes_tr, uniques = factorize_fit(tr[col])
        local_map[col] = uniques
        tr[col] = pd.Series(codes_tr, index=tr.index).astype("int32")
        va[col] = pd.Series(factorize_transform(va[col], uniques), index=va.index).astype("int32")
        te[col] = pd.Series(factorize_transform(te[col], uniques), index=te.index).astype("int32")

    # Integer-floor categorical views of every base numeric
    for col in BASE_NUM_COLS:
        cat_name = f"{col}_cat_"
        floored_tr = np.floor(tr[col]).astype("float32")
        codes_tr, uniques = factorize_fit(floored_tr)
        local_map[cat_name] = uniques
        tr[cat_name] = pd.Series(codes_tr, index=tr.index).astype("int32")
        for dset, dset_src in [(va, df_val), (te, df_te)]:
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
        tr[combo_name] = pd.Series(codes_tr.astype("int32"), index=tr.index).astype("int32")
        for dset in [va, te]:
            combo_s = dset[cols[0]].astype(str)
            for col in cols[1:]:
                combo_s = combo_s + "|" + dset[col].astype(str)
            codes = factorize_transform(combo_s, uniques)
            dset[combo_name] = pd.Series(codes, index=dset.index).astype("int32")

    # Collect all integer-coded categorical column names
    cat_col_names = BASE_CAT_COLS + [f"{c}_cat_" for c in BASE_NUM_COLS] + \
                    [f"delta_{n}_quantile_bin_" for n in [100, 500]] + combo_names
    # Only keep those that actually exist in tr
    cat_col_names = [c for c in cat_col_names if c in tr.columns]

    return tr, va, te, cat_col_names, combo_names, local_map


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


def make_catboost(seed: int = SEED, iterations: int = 5000) -> CatBoostClassifier:
    """Return a memory-reduced CatBoostClassifier for fs_realmlp_fe features.

    Changes vs node_0034:
      - depth: 8 → 6      (fewer per-tree histogram RAM)
      - border_count: 254 → 128  (halves per-feature histogram RAM)
    Everything else identical.
    """
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,          # WAS 8 — reduced to cut per-tree split memory
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        eval_metric="Accuracy",      # Accuracy is a proxy for early stopping; BA is computed manually
        auto_class_weights="Balanced",
        random_seed=seed,
        task_type="CPU",
        thread_count=8,
        od_type="Iter",
        od_wait=100,
        use_best_model=True,
        verbose=False,
        border_count=128,  # WAS 254 — halves per-feature histogram memory
    )


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data …")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
y_all_str = train_raw[TARGET].values   # string labels for CatBoost
n_train = len(train_raw)
n_test = len(test_raw)

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE …")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")
log_rss("after_stateless_fe")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
feature_cols_final = None

log("Starting OOF loop …")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")
    log_rss(f"fold{fold_id}_start")

    # Categorical encoding — fit_in_fold on train rows only
    X_tr_fold, X_val_fold, X_te_fold, cat_col_names, combo_names, local_map = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # Target encoding — fit_in_fold on train rows only
    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]
    y_tr_fold_str = y_all_str[tr_idx]
    y_val_fold_str = y_all_str[val_idx]

    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_tr_fold, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    # Sort columns consistently
    all_feat_cols = sorted(X_tr_fold.columns.tolist())
    X_tr_fold = X_tr_fold[all_feat_cols]
    X_val_fold = X_val_fold[all_feat_cols]
    X_te_fold = X_te_fold[all_feat_cols]

    # Non-TE cat cols (integer-coded) for CatBoost cat_features indices
    # TE columns are numeric (float32) — CatBoost treats them as float automatically
    cat_cols_sorted = sorted(cat_col_names)
    cat_feature_indices = [all_feat_cols.index(c) for c in cat_cols_sorted if c in all_feat_cols]

    if feature_cols_final is None:
        feature_cols_final = all_feat_cols
        log(f"  n_features={len(all_feat_cols)}  n_cat={len(cat_feature_indices)}")

    # Convert int cat columns to string for CatBoost native cat handling
    for col in cat_cols_sorted:
        if col in X_tr_fold.columns:
            X_tr_fold[col] = X_tr_fold[col].astype(str)
            X_val_fold[col] = X_val_fold[col].astype(str)
            X_te_fold[col] = X_te_fold[col].astype(str)

    # Re-get cat_feature_indices after sorting (string cols are now cat features)
    cat_feature_indices = [all_feat_cols.index(c) for c in cat_cols_sorted if c in all_feat_cols]

    train_pool = Pool(X_tr_fold, label=y_tr_fold_str, cat_features=cat_feature_indices)
    val_pool = Pool(X_val_fold, label=y_val_fold_str, cat_features=cat_feature_indices)
    test_pool = Pool(X_te_fold, cat_features=cat_feature_indices)

    # Free the raw fold DataFrames before training (Pools hold their own copy)
    del X_tr_fold, X_val_fold, X_te_fold, local_map
    gc.collect()
    log_rss(f"fold{fold_id}_after_pool_build")

    model = make_catboost(seed=fold_seed)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    log_rss(f"fold{fold_id}_after_fit")

    val_proba = model.predict_proba(val_pool)   # shape (|val|, 3) — classes in model.classes_ order
    class_order = list(model.classes_)           # e.g. ['GALAXY', 'QSO', 'STAR']

    # Re-align columns to CLASSES order
    val_proba_aligned = np.zeros((len(val_idx), N_CLASSES), dtype=np.float32)
    for lbl in CLASSES:
        dest_col = LABEL_MAP[lbl]
        src_col = class_order.index(lbl)
        val_proba_aligned[:, dest_col] = val_proba[:, src_col]

    oof_proba[val_idx] = val_proba_aligned

    # Test predictions — average across folds
    test_proba = model.predict_proba(test_pool)
    test_proba_aligned = np.zeros((n_test, N_CLASSES), dtype=np.float32)
    for lbl in CLASSES:
        dest_col = LABEL_MAP[lbl]
        src_col = class_order.index(lbl)
        test_proba_aligned[:, dest_col] = test_proba[:, src_col]
    test_proba_accum += test_proba_aligned / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, val_proba_aligned.argmax(1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    best_iter = model.best_iteration_
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  best_iter={best_iter}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    # Explicit cleanup to free memory before next fold
    del model, train_pool, val_pool, test_pool
    gc.collect()
    log_rss(f"fold{fold_id}_after_cleanup")

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
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
# feature_cols_final captured on first fold; excludes id/target
(NODE_SRC / "features.txt").write_text("\n".join(feature_cols_final) + "\n")
log(f"Wrote features.txt ({len(feature_cols_final)} features)")

# ─── Final OOF metric ────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
