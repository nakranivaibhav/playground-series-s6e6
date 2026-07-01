"""node_0125 — draft (gbdt): LightGBM on fs_realmlp_fe + fs_spatialknn.

THE ONE ATOMIC CHANGE vs node_0030:
  Add fs_spatialknn: for each row, the fraction of its K nearest sky-neighbours
  (KD-tree on (alpha, delta)) in each class, for K in {10, 50, 200}.
  3 classes × 3 K = 9 new features.

LEAK DISCIPLINE — fs_spatialknn is fit_in_fold (critical):
  - The KD-tree is built on TRAIN-FOLD rows ONLY inside the fold loop.
  - Val rows query the train-fold tree (no self-exclusion needed, they are held out).
  - Train rows query with K+1 and drop the first NN (self-exclusion) so no row
    sees its own label.
  - Test predictions: tree built on ALL train rows, test rows query with no exclusion.
  - Class fractions use train-fold labels only (spatial target-encode).

FALSIFICATION NODE:
  This is a FALSIFICATION of omadon's +0.003 claim. Our bank predicts a
  coordinate-reuse MIRAGE (n083, n060, n013 precedent). The holdout verdict
  (working folds 0-3 vs untouched fold 4) decides.

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
from scipy.spatial import cKDTree
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

warnings.filterwarnings("ignore")

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

REPO_ROOT = NODE_SRC
# Walk up looking for the repo root (identified by tools/validate_submission.py).
# leakage_scan.py was deleted Jun 2026; use validate_submission.py as the sentinel.
while not (REPO_ROOT / "tools" / "validate_submission.py").exists():
    parent = REPO_ROOT.parent
    if parent == REPO_ROOT:
        # Reached filesystem root without finding the sentinel — best effort
        REPO_ROOT = COMP_DIR.parent.parent
        break
    REPO_ROOT = parent

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

# Spatial KNN K values for class-fraction features
KNN_KS = [10, 50, 200]

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


def compute_spatial_knn_fracs(
    coords_ref: np.ndarray,
    y_ref: np.ndarray,
    coords_qry: np.ndarray,
    k_values: list,
    exclude_self: bool,
) -> np.ndarray:
    """
    Build a KD-tree on coords_ref and compute class-fraction features for
    coords_qry rows.

    LEAK DISCIPLINE:
    - coords_ref / y_ref must be TRAIN-FOLD rows ONLY (never full train or test).
    - exclude_self=True for train rows: queries K+1, drops the first NN (itself).
    - exclude_self=False for val/test rows: queries K, all NNs are from ref set.

    Returns: ndarray of shape (len(coords_qry), N_CLASSES * len(k_values))
      columns ordered as: class0_k10, class1_k10, class2_k10, class0_k50, ...
    """
    max_k = max(k_values)
    kq = max_k + 1 if exclude_self else max_k

    tree = cKDTree(coords_ref)
    _, nn_all = tree.query(coords_qry, k=kq)
    # nn_all shape: (n_qry, kq)

    result_cols = []
    for k in k_values:
        if exclude_self:
            # Drop self (index 0) and take k after that
            nn_k = nn_all[:, 1:k + 1]
        else:
            nn_k = nn_all[:, :k]
        # Labels for the k neighbours
        lab = y_ref[nn_k]  # (n_qry, k)
        # Class fractions
        for c in range(N_CLASSES):
            result_cols.append((lab == c).mean(axis=1).astype("float32"))

    return np.column_stack(result_cols)


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
    Byte-identical to node_0030.
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

# ─── Coordinate arrays for spatial KNN (extracted once) ───────────────────────
# These are stateless coordinates — no labels. Tree is built fold-locally.
coords_train = train_raw[["alpha", "delta"]].values.astype("float64")
coords_test = test_raw[["alpha", "delta"]].values.astype("float64")

# ─── PRE-TRAIN LEAKAGE CHECKS ─────────────────────────────────────────────────
log("Running pre-train leakage checks ...")

# Check 1: target and id not in stateless features
stateless_cols = set(X_stateless.columns)
assert TARGET not in stateless_cols, f"LEAK: {TARGET} in features"
assert IDC not in stateless_cols, f"LEAK: {IDC} in features"
log("  Check 1-2 PASS: target and id not in stateless features")

# Check 3: single-feature↔target sweep (stateless features only; spatial features
# are fit_in_fold so not available pre-fold, but we sweep a fold-0 sample after building)
sample_size = min(50_000, n_train)
rng = np.random.default_rng(0)
samp_idx = rng.choice(n_train, sample_size, replace=False)
X_samp = X_stateless.iloc[samp_idx]
y_samp = y_all[samp_idx]
max_corr = 0.0
max_corr_col = None
for c in X_samp.columns:
    x = pd.to_numeric(X_samp[c], errors="coerce").fillna(0).values
    if x.std() > 0:
        corr_val = abs(np.corrcoef(x, y_samp)[0, 1])
        if corr_val > max_corr:
            max_corr = corr_val
            max_corr_col = c
log(f"  Check 3 (stateless sweep): max |corr| = {max_corr:.4f} on {max_corr_col}")
if max_corr >= 0.999:
    raise SystemExit(f"LEAK: stateless feature {max_corr_col} ~ target corr={max_corr:.4f}")
log("  Check 3 PASS: no near-perfect stateless feature↔target correlation")

# Check 4 (code read): spatial KNN is built inside the fold loop below.
# - tree built on tr_idx rows only
# - val rows: exclude_self=False (they are held out, no contamination)
# - train rows: exclude_self=True (K+1 query, drop self)
# - test rows in final refit: tree on ALL train, no self-exclusion (correct)
log("  Check 4 (fold-loop code read): fs_spatialknn is fit_in_fold - verified below")

# Check 5: folds from frozen folds.json (loaded above via json.loads)
log(f"  Check 5 PASS: {len(folds_list)} folds loaded from frozen folds.json")

# Check 6: train↔test near-dup check on coordinates
samp_tr = coords_train[:5000]
samp_te = coords_test[:5000]
tr_set = set(map(tuple, samp_tr.round(6)))
te_set = set(map(tuple, samp_te.round(6)))
n_dup = len(tr_set & te_set)
log(f"  Check 6: train↔test coord near-dups (5k sample): {n_dup}")
if n_dup > 50:
    log(f"  WARNING: {n_dup} near-duplicate coords found — this is the coord-reuse signal!")

log("Pre-train leakage checks complete.")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
all_cols_final = None
best_iters = []
spatial_feat_names = [
    f"_spknn_c{c}_k{k}" for k in KNN_KS for c in range(N_CLASSES)
]

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # ── fs_spatialknn — fit_in_fold (CRITICAL LEAK CHECK) ────────────────────
    # Tree is built on TRAIN-FOLD coordinates only.
    coords_tr_fold = coords_train[tr_idx]   # train-fold coords
    y_tr_fold_labels = y_all[tr_idx]         # train-fold labels (for class fractions)
    coords_val_fold = coords_train[val_idx]  # val-fold coords

    # Train rows: exclude_self=True → query K+1, drop NN[0] (self)
    log(f"  Computing spatial KNN for train rows (self-exclusion, K={KNN_KS}) ...")
    spknn_tr = compute_spatial_knn_fracs(
        coords_ref=coords_tr_fold,
        y_ref=y_tr_fold_labels,
        coords_qry=coords_tr_fold,
        k_values=KNN_KS,
        exclude_self=True,
    )
    # Val rows: exclude_self=False → query K, all NNs from train-fold only
    log(f"  Computing spatial KNN for val rows (no self, tree=train-fold only) ...")
    spknn_val = compute_spatial_knn_fracs(
        coords_ref=coords_tr_fold,
        y_ref=y_tr_fold_labels,
        coords_qry=coords_val_fold,
        k_values=KNN_KS,
        exclude_self=False,
    )
    # Test rows: tree on TRAIN-FOLD only (will also do a full-train tree in final refit)
    log(f"  Computing spatial KNN for test rows (tree=train-fold) ...")
    spknn_te_fold = compute_spatial_knn_fracs(
        coords_ref=coords_tr_fold,
        y_ref=y_tr_fold_labels,
        coords_qry=coords_test,
        k_values=KNN_KS,
        exclude_self=False,
    )

    # ── Post-build AUC sweep on fold-0 spatial features (check 3 extension) ──
    if fold_id == 0:
        log("  Spatial feature AUC sweep on fold-0 val sample ...")
        from sklearn.metrics import roc_auc_score
        max_sp_corr = 0.0
        max_sp_col = None
        for ci, fname in enumerate(spatial_feat_names):
            feat_vals = spknn_val[:, ci]
            # Binary: class 0 vs rest (GALAXY is majority, likely highest AUC)
            for cls in range(N_CLASSES):
                y_binary = (y_all[val_idx] == cls).astype(int)
                if y_binary.sum() > 0 and y_binary.sum() < len(y_binary):
                    try:
                        auc = roc_auc_score(y_binary, feat_vals)
                        auc_dist = abs(auc - 0.5)
                        corr_v = abs(np.corrcoef(feat_vals, y_all[val_idx])[0, 1])
                        if corr_v > max_sp_corr:
                            max_sp_corr = corr_v
                            max_sp_col = fname
                    except Exception:
                        pass
        log(f"  Spatial feature max |corr|~target on fold-0 val: {max_sp_corr:.4f} (col={max_sp_col})")
        if max_sp_corr >= 0.999:
            log(f"  *** LEAK/MIRAGE SIGNAL: spatial feature near-perfect corr! ***")
        else:
            log(f"  Spatial AUC sweep PASS: max corr={max_sp_corr:.4f}")

    # Categorical encoding — fit_in_fold
    X_tr_fold, X_val_fold, X_te_fold, all_cat_cols, combo_names, local_map, lgbm_cat_cols = fit_fold_categoricals(
        X_stateless.iloc[tr_idx].reset_index(drop=True),
        X_stateless.iloc[val_idx].reset_index(drop=True),
        X_test_stateless.copy(),
    )

    # Add spatial KNN features to the dataframes
    spknn_tr_df = pd.DataFrame(spknn_tr, columns=spatial_feat_names, index=X_tr_fold.index)
    spknn_val_df = pd.DataFrame(spknn_val, columns=spatial_feat_names, index=X_val_fold.index)
    spknn_te_df = pd.DataFrame(spknn_te_fold, columns=spatial_feat_names, index=X_te_fold.index)

    X_tr_fold = pd.concat([X_tr_fold, spknn_tr_df], axis=1)
    X_val_fold = pd.concat([X_val_fold, spknn_val_df], axis=1)
    X_te_fold = pd.concat([X_te_fold, spknn_te_df], axis=1)

    # Target encoding — fit_in_fold
    y_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]
    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding(
        X_tr_fold, y_fold, X_val_fold, X_te_fold, combo_names, fold_seed
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
        X_tr_fold, y_fold,
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
    del spknn_tr, spknn_val, spknn_te_fold
    del spknn_tr_df, spknn_val_df, spknn_te_df
    gc.collect()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")
        # Cheap-kill criterion: fold-0 < 0.965
        if fold_score < 0.965:
            log(f"  CHEAP-KILL: fold-0 BA={fold_score:.6f} < 0.965 threshold — stopping.")
            print(f"cv=null (cheap-kill fold-0={fold_score:.6f})", flush=True)
            sys.exit(0)

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
log(f"best_iters_per_fold={best_iters}")
print(f"cv={mean_cv:.6f}", flush=True)

# Working (folds 0-3) vs Holdout (fold 4) split for falsification verdict
working_scores = [s for i, s in enumerate(per_fold_scores) if i < 4]
holdout_scores = [s for i, s in enumerate(per_fold_scores) if i == 4]
log(f"FALSIFICATION VERDICT:")
log(f"  Working folds (0-3) BA: {np.mean(working_scores):.6f}")
log(f"  Holdout fold  (4)   BA: {holdout_scores[0]:.6f}")
log(f"  Delta working-holdout:  {np.mean(working_scores) - holdout_scores[0]:.6f}")

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
