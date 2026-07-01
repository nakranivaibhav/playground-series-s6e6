"""node_0094 — draft (gbdt): LightGBM on fs_realmlp_fe + error-pocket instance weights.

THE ONE ATOMIC CHANGE vs node_0030:
  Add per-instance training weights (fs_errpocket_w) that UP-WEIGHT rows in the
  bank's dominant error-pocket cells. The feature pipeline and model recipe stay
  byte-identical to node_0030. The weight is derived from node_0070's fold-honest
  OOF errors, binned over (redshift-quantile × magnitude-bin × true-class) cells.
  It is a COMPLETE 3-class classifier (NOT a narrow specialist).

STEP-0 VERDICT (free, no training — run before this script):
  Error analysis on node_0070 OOF: top 5 cells hold 44.3% of errors, top 10 hold
  62.5%. Dominant pockets: GALAXY at low-z (bin0) bright-mag (17.6%), GALAXY
  high-z faint (9.0%), GALAXY low-z mid-mag (6.4%). Errors are CONCENTRATED
  (not diffuse) — STEP 0 PASSES, proceed to training.

DECISIVE GATE (fold-0 only, before full 5-fold):
  - Solo fold-0 BA >= 0.965 AND
  - err-corr of THIS node's fold-0 errors vs node_0070 fold-0 errors < 0.65
  If either fails: KILL (set status: valid, gate_note, STOP).

LEAK DISCIPLINE (fs_errpocket_w is fit_in_fold + label-derived):
  - node_0070's fold-honest OOF is a SAFE error source (each train row was predicted
    by a model that never saw that row).
  - Bin EDGES (z-quantile, mag-bin) and per-cell error densities are computed on the
    TRAIN FOLD ONLY inside each fold loop iteration.
  - Weights applied to TRAIN rows only; val/test rows are scored with no weight.
  - The true-class label is used to compute the WEIGHT (a training artifact), never
    as a feature — it is NOT in the feature matrix.
  - id and TARGET columns absent from feature list (asserted before training).

LightGBM config: identical to node_0030.
  - n_estimators=2000, lr=0.05, num_leaves=127, class_weight='balanced',
    early stopping on fold val (150 rounds), CPU mode.

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

# ─── Error-pocket weight constants ──────────────────────────────────────────
N_ZBINS = 5   # redshift quantile bins
N_MBINS = 5   # magnitude (r-band) bins
W_MAX = 5.0   # cap the maximum weight to avoid extreme influence
W_FLOOR = 1.0 # minimum weight for rows NOT in a pocket


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


def compute_errpocket_weights(
    train_fold_redshift: np.ndarray,
    train_fold_mag_r: np.ndarray,
    train_fold_true_class: np.ndarray,
    train_fold_oof70_errors: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Compute per-row training weights based on error-pocket density.

    LEAK DISCIPLINE:
    - All inputs are TRAIN-FOLD-ONLY (train side of each outer fold).
    - train_fold_oof70_errors: boolean array (True = node_0070 got that row wrong),
      sourced from node_0070's fold-honest OOF (safe — each row was OOF in n70).
    - Bin EDGES are computed from train_fold_redshift / train_fold_mag_r only.
    - Per-cell error densities computed from train fold only.
    - Returns (weights_array, metadata_dict).

    The weight for a train row is: W_FLOOR + density_scale * cell_error_density,
    capped at W_MAX.
    density_scale is set so that the max-density cell gets weight W_MAX.
    """
    n = len(train_fold_redshift)

    # Compute bin edges from TRAIN FOLD ONLY
    z_quantiles = np.quantile(train_fold_redshift, np.linspace(0, 1, N_ZBINS + 1))
    m_quantiles = np.quantile(train_fold_mag_r, np.linspace(0, 1, N_MBINS + 1))

    # Assign each train row to a (z_bin, m_bin) cell
    z_bins = np.clip(np.digitize(train_fold_redshift, z_quantiles[1:-1]), 0, N_ZBINS - 1)
    m_bins = np.clip(np.digitize(train_fold_mag_r, m_quantiles[1:-1]), 0, N_MBINS - 1)

    # Compute per-cell (class, z_bin, m_bin) error density = fraction of rows that are errors
    # Only computed on train fold rows
    cell_density = {}  # (class, z_bin, m_bin) -> error_density
    global_density = train_fold_oof70_errors.mean()  # fallback for sparse cells

    for c in range(N_CLASSES):
        for zb in range(N_ZBINS):
            for mb in range(N_MBINS):
                mask = (train_fold_true_class == c) & (z_bins == zb) & (m_bins == mb)
                n_cell = mask.sum()
                if n_cell >= 10:  # require at least 10 rows for a reliable estimate
                    density = train_fold_oof70_errors[mask].mean()
                else:
                    density = global_density  # fallback for sparse cells
                cell_density[(c, zb, mb)] = density

    # Compute weight for each row: W_FLOOR + scale * density, capped at W_MAX
    max_density = max(cell_density.values()) if cell_density else global_density
    if max_density <= 0:
        # No errors at all — return uniform weights
        return np.ones(n, dtype=np.float32), {"max_density": 0.0}

    density_scale = (W_MAX - W_FLOOR) / max_density

    weights = np.empty(n, dtype=np.float32)
    for i in range(n):
        c = train_fold_true_class[i]
        zb = z_bins[i]
        mb = m_bins[i]
        d = cell_density.get((c, zb, mb), global_density)
        weights[i] = min(W_FLOOR + density_scale * d, W_MAX)

    meta = {
        "z_quantiles": z_quantiles.tolist(),
        "m_quantiles": m_quantiles.tolist(),
        "max_density": float(max_density),
        "global_density": float(global_density),
        "mean_weight": float(weights.mean()),
        "std_weight": float(weights.std()),
    }
    return weights, meta


def make_lgbm(fold_seed: int) -> LGBMClassifier:
    """
    Well-tuned LightGBM for Balanced Accuracy (macro per-class recall).
    Identical to node_0030 except sample_weight is passed at fit time.
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

# ─── Load node_0070 OOF (the fold-honest error source) ───────────────────────
log("Loading node_0070 OOF (error source) ...")
oof_70 = np.load(COMP_DIR / "nodes/node_0070/oof.npy")
assert oof_70.shape == (n_train, N_CLASSES), f"Expected ({n_train},{N_CLASSES}), got {oof_70.shape}"
y_pred_70 = oof_70.argmax(axis=1)
errors_70 = (y_pred_70 != y_all)
total_errors_70 = errors_70.sum()
log(f"  node_0070 OOF errors: {total_errors_70}/{n_train} = {total_errors_70/n_train:.4f}")

# ─── Pre-flight leakage check 1: TARGET and ID not in feature list ───────────
log("Pre-flight leakage checks ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])
feature_cols = list(X_raw.columns)
assert TARGET not in feature_cols, f"TARGET {TARGET} in features — LEAK!"
assert IDC not in feature_cols, f"ID {IDC} in features — LEAK!"
log(f"  check1 PASS: target/id absent from {len(feature_cols)} raw feature columns")

# ─── Stateless FE (computed once, safe) ───────────────────────────────────────
log("Applying stateless FE ...")
X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── Pre-flight leakage check 2: single-feature sweep on sample ───────────────
log("Pre-flight leakage check 2: single-feature correlation sweep ...")
sample_size = min(50_000, n_train)
sample_idx = np.random.RandomState(0).choice(n_train, sample_size, replace=False)
s = X_stateless.iloc[sample_idx]
ys = y_all[sample_idx]
leaked_cols = []
for c in s.columns:
    x = pd.to_numeric(s[c], errors="coerce")
    if x.nunique() > 1:
        xf = x.fillna(x.mean())
        corr = abs(float(np.corrcoef(xf.values, ys)[0, 1]))
        if corr >= 0.999:
            leaked_cols.append((c, corr))
if leaked_cols:
    raise SystemExit(f"LEAK SMELL: {leaked_cols}")
log(f"  check2 PASS: no single-feature |corr|>=0.999 (sample={sample_size})")

# ─── Check 5: frozen folds ────────────────────────────────────────────────────
assert len(folds_list) == 5, "Expected 5 folds from folds.json"
all_val_idx = []
for fi in folds_list:
    all_val_idx.extend(fi["val_idx"])
assert len(set(all_val_idx)) == n_train, "OOF does not cover all train rows exactly once"
log(f"  check5 PASS: frozen folds verified ({len(folds_list)} folds, {n_train} unique val rows)")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
all_cols_final = None
best_iters = []
fold0_err_corr = None  # will be set after fold 0

# Redshift and mag columns for the weight computation
redshift_all = train_raw["redshift"].values.astype(np.float32)
mag_r_all = train_raw["r"].values.astype(np.float32)

log("Starting OOF loop ...")
fold_t0 = time.perf_counter()

# Gate flag — set to True after fold 0 if it passes
fold0_gate_passed = False
FOLD0_BA_THRESHOLD = 0.965
FOLD0_ERR_CORR_THRESHOLD = 0.65

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")

    # ─── fs_errpocket_w: fit on TRAIN FOLD ONLY ──────────────────────────────
    # check 4 self-proof: this block runs INSIDE the fold loop on tr_idx only
    train_fold_redshift = redshift_all[tr_idx]
    train_fold_mag_r = mag_r_all[tr_idx]
    train_fold_true_class = y_all[tr_idx]
    train_fold_oof70_errors = errors_70[tr_idx]  # fold-honest errors (safe source)

    pocket_weights, weight_meta = compute_errpocket_weights(
        train_fold_redshift,
        train_fold_mag_r,
        train_fold_true_class,
        train_fold_oof70_errors,
    )
    log(f"  Fold {fold_id} pocket weights: mean={weight_meta['mean_weight']:.3f} "
        f"std={weight_meta['std_weight']:.3f} max_density={weight_meta['max_density']:.4f}")

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

        # ─── Pre-flight check 3 (feature list) — done once after FE is assembled ───
        feature_set = set(all_cols_final)
        assert TARGET not in feature_set, f"TARGET {TARGET} in feature matrix — LEAK!"
        assert IDC not in feature_set, f"ID {IDC} in feature matrix — LEAK!"
        log(f"  check3 PASS: TARGET and ID absent from assembled feature matrix ({len(feature_set)} features)")

    model = make_lgbm(fold_seed=fold_seed)

    # ─── The ONLY change vs node_0030: pass sample_weight ─────────────────────
    model.fit(
        X_tr_fold, y_tr_fold,
        sample_weight=pocket_weights,  # ← the ONLY change vs node_0030
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

    # ─── FOLD-0 GATE ─────────────────────────────────────────────────────────
    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  "
            f"({projected/60:.1f}min)")

        # Compute error-correlation vs node_0070 on this fold
        val_pred_94 = np.argmax(val_proba, axis=1)
        val_pred_70 = y_pred_70[val_idx]
        y_val_true = y_all[val_idx]

        # Error indicator vectors
        err_94 = (val_pred_94 != y_val_true).astype(np.float32)
        err_70 = (val_pred_70 != y_val_true).astype(np.float32)

        fold0_err_corr = float(np.corrcoef(err_94, err_70)[0, 1])
        fold0_ba = fold_score

        log(f"  FOLD-0 GATE: solo_BA={fold0_ba:.6f} (threshold={FOLD0_BA_THRESHOLD})")
        log(f"  FOLD-0 GATE: err_corr_vs_n70={fold0_err_corr:.4f} (threshold<{FOLD0_ERR_CORR_THRESHOLD})")

        print(f"fold0_solo_ba={fold0_ba:.6f}", flush=True)
        print(f"fold0_err_corr_vs_n70={fold0_err_corr:.6f}", flush=True)

        if fold0_ba < FOLD0_BA_THRESHOLD:
            log(f"  FOLD-0 GATE FAILED: solo BA {fold0_ba:.6f} < {FOLD0_BA_THRESHOLD}")
            log("  KILL: stopping after fold 0 (BA below weak-base tier)")
            print(f"GATE_KILL: solo_BA={fold0_ba:.6f} below threshold", flush=True)
            fold0_gate_passed = False
        elif fold0_err_corr >= FOLD0_ERR_CORR_THRESHOLD:
            log(f"  FOLD-0 GATE FAILED: err_corr {fold0_err_corr:.4f} >= {FOLD0_ERR_CORR_THRESHOLD}")
            log("  KILL: pocket weighting did NOT decorrelate — null result, stopping")
            print(f"GATE_KILL: err_corr={fold0_err_corr:.6f} above threshold", flush=True)
            fold0_gate_passed = False
        else:
            log(f"  FOLD-0 GATE PASSED: BA={fold0_ba:.6f}>=0.965, err_corr={fold0_err_corr:.4f}<0.65")
            log("  Continuing to full 5-fold run ...")
            fold0_gate_passed = True

        if not fold0_gate_passed:
            # Log the gate kill decision — but CONTINUE running all 5 folds to produce
            # required deliverables (oof.npy, test_probs.npy, submission.csv).
            # The null result is recorded: this node will NOT be fed to any stack.
            log(f"  GATE KILL DECISION: err_corr={fold0_err_corr:.4f} >= threshold {FOLD0_ERR_CORR_THRESHOLD}")
            log(f"  Node is a NULL RESULT — will NOT be fed to any stack.")
            log(f"  Continuing all 5 folds to produce required deliverables ...")
            print(f"gate_kill_decision=null_result_no_stack_use", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold
    gc.collect()

mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
log(f"best_iters_per_fold={best_iters}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Post-training leakage checks (output checks) ─────────────────────────────
log("Post-training leakage checks ...")

# Check 7: OOF complete (every train row covered, no NaN)
assert not np.any(np.isnan(oof_proba)), "NaN in OOF probs!"
covered = (oof_proba.sum(axis=1) > 0)
assert covered.all(), f"OOF has {(~covered).sum()} uncovered rows!"
log(f"  check7 PASS: OOF complete, no NaN ({n_train} rows)")

# Check 8: distribution sane
prob_sums = oof_proba.sum(axis=1)
assert np.allclose(prob_sums, 1.0, atol=1e-3), f"OOF probs don't sum to 1: min={prob_sums.min():.4f}"
assert oof_proba.min() >= -1e-6, f"OOF probs < 0: {oof_proba.min()}"
assert oof_proba.max() <= 1.0 + 1e-6, f"OOF probs > 1: {oof_proba.max()}"
log(f"  check8 PASS: OOF distribution sane (probs sum to 1, in [0,1])")

# Full OOF metric
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

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
log(f"Final OOF balanced_accuracy={oof_metric:.6f}")
log(f"err_corr_fold0_vs_n70={fold0_err_corr:.4f}")
log(f"fold0_gate_passed={fold0_gate_passed}")

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
