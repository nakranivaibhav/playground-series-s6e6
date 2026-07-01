"""node_0115 — TRUE-BAGGED CatBoost (5 bags, bootstrap + rsm + subsample) on fs_realmlp_fe.

THE ONE ATOMIC CHANGE vs node_0039:
  Replace the single CatBoost per fold with a 5-bag TRUE BAG:
    (1) Bootstrap-resample train-fold rows WITH REPLACEMENT per bag (row diversity).
    (2) CatBoost with rsm=0.7 (random column subspace per tree),
        bootstrap_type='Bernoulli', subsample=0.7 (per-tree row subsample),
        random_seed varied per bag (model diversity).
    (3) EARLY-STOPPING LEAK FIX: node_0039 early-stopped on the OOF val fold
        (eval_set = val_pool over val_idx) — that is a leak. Here we carve an
        INNER early-stop split from TRAIN-FOLD rows only (~10% of tr_idx,
        bag-seeded). The OOF val_idx rows NEVER appear in any fit or eval_set.
  Bagged fold OOF = mean of 5 bags' val probs.
  Test probs = mean of bag probs, averaged across folds.

Feature-set fs_realmlp_fe (byte-identical to node_0039):
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
  - EARLY-STOP inner split carved from train-fold rows only (val_idx EXCLUDED).
  - Bootstrap sampling is done on tr_minus_inner (train-fold minus inner-val).
  - Frozen folds.json throughout; no refitting of folds.

Mode flags:
  --probe   Run fold-0 only (all 5 bags) for timing + decorrelation verdict.
            Prints 1-bag timing first, then runs remaining 4 bags.

Outputs (full run in NODE_DIR):
  oof.npy           (577347, 3)  — 5-bag mean OOF probs
  test_probs.npy    (247435, 3)  — 5-bag mean test probs, averaged across folds
  oof_bags.npy      (577347, 5, 3) — per-bag OOF probs (for restack)
  submission.csv
  train.log

Metric: Balanced Accuracy Score (macro-average per-class recall), maximize.
"""
from __future__ import annotations

import argparse
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

# ─── Parse args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--probe", action="store_true",
                    help="Run fold-0 only (all 5 bags) for timing + decorrelation verdict")
args, _ = parser.parse_known_args()
PROBE_MODE = args.probe

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

REPO_ROOT = COMP_DIR.parent  # repo root — no parent-walk (tools/leakage_scan.py was removed; vestigial)

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
B = 5  # number of bags

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
    """Return a true-bagging CatBoostClassifier for fs_realmlp_fe features.

    Bagging stochasticity vs node_0039:
      - rsm=0.7: random column subspace per tree
      - bootstrap_type='Bernoulli': per-tree row subsampling
      - subsample=0.7: per-tree row sample fraction
      - thread_count=-1: use all available cores (runs are sequential)
      - random_seed=seed: varied per bag for model-level diversity

    Fixed vs node_0039:
      - depth=6, border_count=128, l2_leaf_reg=3.0 (memory-safe, proven)
      - learning_rate=0.05, iterations=5000
      - auto_class_weights='Balanced'
      - od_type='Iter', od_wait=100, use_best_model=True
      - task_type='CPU'
    """
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3.0,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        random_seed=seed,
        task_type="CPU",
        thread_count=-1,
        od_type="Iter",
        od_wait=100,
        use_best_model=True,
        verbose=False,
        border_count=128,
        rsm=0.7,
        bootstrap_type="Bernoulli",
        subsample=0.7,
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

if PROBE_MODE:
    log("PROBE MODE: running fold-0 only (all 5 bags)")
    folds_list = [fi for fi in folds_list if fi["fold"] == 0]

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE …")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")
log_rss("after_stateless_fe")

# ─── OOF accumulators ─────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
oof_bags = np.zeros((n_train, B, N_CLASSES), dtype=np.float32)
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

    # ── Fit FE ONCE per fold on tr_idx (fit_in_fold — train rows only) ────────
    # val_idx rows are NEVER passed to fit_fold_categoricals or add_target_encoding
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

    # Re-get cat_feature_indices after converting to string
    cat_feature_indices = [all_feat_cols.index(c) for c in cat_cols_sorted if c in all_feat_cols]

    # X_tr_fold is now a clean DataFrame with local positional index 0..len(tr_idx)-1
    n_tr_local = len(X_tr_fold)
    local_positions = np.arange(n_tr_local)

    # ── Bagging loop: B bags per fold ─────────────────────────────────────────
    bag_val_accum = np.zeros((len(val_idx), N_CLASSES), dtype=np.float32)
    bag_te_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)

    fold_bag_t0 = time.perf_counter()

    for b in range(B):
        bag_seed = fold_seed + b + 1  # unique per bag
        rng = np.random.default_rng(bag_seed)

        # ── INNER EARLY-STOP SPLIT (from train-fold rows only, val_idx EXCLUDED) ──
        # Carve ~10% of local train positions as inner val for early stopping
        n_inner_val = max(1, int(0.10 * n_tr_local))
        inner_val_local = rng.choice(local_positions, size=n_inner_val, replace=False)
        inner_val_mask = np.zeros(n_tr_local, dtype=bool)
        inner_val_mask[inner_val_local] = True
        inner_tr_local = local_positions[~inner_val_mask]

        # ── BOOTSTRAP: resample inner_tr_local WITH REPLACEMENT ───────────────
        # Bootstrap gives row diversity; rsm+subsample give per-tree diversity
        boot_local = rng.choice(inner_tr_local, size=len(inner_tr_local), replace=True)

        # ── Build CatBoost Pools for this bag ─────────────────────────────────
        X_boot = X_tr_fold.iloc[boot_local]
        y_boot_str = y_tr_fold_str[boot_local]  # boot_local indexes into tr_idx-aligned array

        X_inner_val = X_tr_fold.iloc[inner_val_local]
        y_inner_val_str = y_tr_fold_str[inner_val_local]

        train_pool = Pool(X_boot, label=y_boot_str, cat_features=cat_feature_indices)
        inner_val_pool = Pool(X_inner_val, label=y_inner_val_str, cat_features=cat_feature_indices)
        val_pool = Pool(X_val_fold, label=y_val_fold_str, cat_features=cat_feature_indices)
        test_pool = Pool(X_te_fold, cat_features=cat_feature_indices)

        model = make_catboost(seed=bag_seed)
        # Early stopping on the INNER val set (train-fold rows only — val_idx excluded)
        model.fit(train_pool, eval_set=inner_val_pool, use_best_model=True)

        bag_elapsed = time.perf_counter() - fold_bag_t0
        log(f"  Fold {fold_id} bag {b}: elapsed={bag_elapsed:.1f}s  best_iter={model.best_iteration_}")

        # Predict OOF val fold (val_idx rows — NEVER in any fit or early-stop)
        val_proba = model.predict_proba(val_pool)
        class_order = list(model.classes_)

        # Re-align to CLASSES order
        val_proba_aligned = np.zeros((len(val_idx), N_CLASSES), dtype=np.float32)
        for lbl in CLASSES:
            dest_col = LABEL_MAP[lbl]
            src_col = class_order.index(lbl)
            val_proba_aligned[:, dest_col] = val_proba[:, src_col]

        bag_val_accum += val_proba_aligned

        # Store per-bag OOF for oof_bags.npy
        oof_bags[val_idx, b, :] = val_proba_aligned

        # Test predictions
        test_proba = model.predict_proba(test_pool)
        test_proba_aligned = np.zeros((n_test, N_CLASSES), dtype=np.float32)
        for lbl in CLASSES:
            dest_col = LABEL_MAP[lbl]
            src_col = class_order.index(lbl)
            test_proba_aligned[:, dest_col] = test_proba[:, src_col]

        bag_te_accum += test_proba_aligned

        # Solo bag BA for diagnostics
        bag_ba = balanced_accuracy_score(y_val_fold, val_proba_aligned.argmax(1))
        log(f"  bag{b}_BA={bag_ba:.6f}")

        # Timing probe: after first bag, project full run
        if fold_id == 0 and b == 0:
            elapsed_1bag = time.perf_counter() - fold_bag_t0
            projected_total = elapsed_1bag * B * len(json.loads((COMP_DIR / "folds.json").read_text())["folds"])
            log(f"  TIMING: 1-bag fold-0={elapsed_1bag:.1f}s  projected_5fold×5bag={projected_total:.1f}s  "
                f"({projected_total/60:.1f}min)")
            if elapsed_1bag > 720:  # > 12 minutes per bag
                log("TIMING ABORT: 1-bag fold-0 exceeds 12 min — reduce iterations")
                sys.exit(1)

        del model, train_pool, inner_val_pool, val_pool, test_pool
        del X_boot, X_inner_val
        gc.collect()

    # ── Bagged fold OOF = mean over 5 bags ────────────────────────────────────
    bagged_val_proba = bag_val_accum / B
    oof_proba[val_idx] = bagged_val_proba

    # Test: accumulate (will be divided by n_folds below)
    test_proba_accum += bag_te_accum / B

    fold_score = balanced_accuracy_score(y_val_fold, bagged_val_proba.argmax(1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: bagged_balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    # Explicit cleanup
    del X_tr_fold, X_val_fold, X_te_fold, local_map
    del bag_val_accum, bag_te_accum, bagged_val_proba
    gc.collect()
    log_rss(f"fold{fold_id}_after_cleanup")

# ── Normalize test probs by number of folds ───────────────────────────────────
n_folds_run = len(folds_list)
test_proba_final = test_proba_accum / n_folds_run

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores))) if len(per_fold_scores) > 1 else 0.0
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}±{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── PROBE-MODE DECORRELATION VERDICT ────────────────────────────────────────
if PROBE_MODE:
    log("Computing decorrelation verdict vs node_0070 oof …")
    bank_oof = np.load(COMP_DIR / "nodes/node_0070/oof.npy")  # (577347, 3)

    # Restrict to fold-0 val rows
    fi0 = [fi for fi in json.loads((COMP_DIR / "folds.json").read_text())["folds"] if fi["fold"] == 0][0]
    val0 = np.asarray(fi0["val_idx"])

    y_val0 = y_all[val0]
    base_pred = oof_proba[val0].argmax(1)
    bank_pred = bank_oof[val0].argmax(1)

    # Error indicators (1 = wrong, 0 = correct)
    err_base = (base_pred != y_val0).astype(float)
    err_bank = (bank_pred != y_val0).astype(float)

    # (i) Overall Pearson corr of error indicators
    overall_corr = float(np.corrcoef(err_base, err_bank)[0, 1])
    log(f"  err-corr (overall): {overall_corr:.4f}")

    # (ii) Per-class-averaged Pearson corr
    per_class_corrs = []
    for c in range(N_CLASSES):
        mask = y_val0 == c
        if mask.sum() < 2:
            continue
        c_corr = float(np.corrcoef(err_base[mask], err_bank[mask])[0, 1])
        per_class_corrs.append(c_corr)
        log(f"  err-corr class={CLASSES[c]}: {c_corr:.4f}")

    per_class_avg_corr = float(np.mean(per_class_corrs))
    log(f"  err-corr (per-class avg): {per_class_avg_corr:.4f}")
    log(f"  DECORRELATION VERDICT: overall={overall_corr:.4f}  per-class-avg={per_class_avg_corr:.4f}  "
        f"(wall ~0.72; <0.65 = real decorrelation)")
    print(f"err_corr_overall={overall_corr:.4f}", flush=True)
    print(f"err_corr_per_class_avg={per_class_avg_corr:.4f}", flush=True)

    total_elapsed = time.perf_counter() - T0
    log(f"PROBE COMPLETE. Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    log("Done.")
    sys.exit(0)

# ─── Full-run outputs ─────────────────────────────────────────────────────────
# Save OOF (5-bag mean)
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

# Save per-bag OOF (577347 x 5 x 3)
np.save(NODE_DIR / "oof_bags.npy", oof_bags)
log(f"Saved oof_bags.npy shape={oof_bags.shape}")

# Save test_probs
np.save(NODE_DIR / "test_probs.npy", test_proba_final)
log(f"Saved test_probs.npy shape={test_proba_final.shape}")

# ─── Write submission ─────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_final.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── Write features.txt ───────────────────────────────────────────────────────
if feature_cols_final is not None:
    (NODE_SRC / "features.txt").write_text("\n".join(feature_cols_final) + "\n")
    log(f"Wrote features.txt ({len(feature_cols_final)} features)")

# ─── Final OOF metric ────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
