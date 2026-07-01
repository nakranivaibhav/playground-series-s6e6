"""node_0123 — draft (gbdt): LightGBM + multiclass focal loss on fs_realmlp_fe.

THE ONE ATOMIC CHANGE vs node_0030:
  Replace the standard 'multiclass' log-loss objective with a custom
  MULTICLASS FOCAL LOSS (softmax -> focal gradient/hessian, tunable gamma).
  FE pipeline (stateless FE + fit-in-fold TargetEncoder/KBins), frozen folds,
  and all tree params stay byte-identical to node_0030.

Focal loss concentrates gradient on hard / misclassified examples -- targeting
the low-z GALAXY<->STAR boundary that is the BA bottleneck.  gamma is tuned
on fold-0 over {1, 2, 3}; best gamma is used for all 5 folds.

Custom objective formula (from Lin et al. 2017 + Mukhoti et al. 2020):
  p = softmax(raw_scores)  [per sample]
  pt = p[true_class]       [scalar per sample]
  For each class k:
    g_k = (1-pt)^(gamma-1) * (p_k - y_k) * [(1-pt) - gamma*pt*log(pt)]
  Hessian: diagonal approx 2*p_k*(1-p_k)  (same as plain softmax -- stable)

Note on speed: Python custom fobj is called once per boosting round with full
train (577k rows). Each call costs ~50-80ms; with ~500-800 rounds before early
stopping, total overhead is ~30-65s per fold. This is acceptable (adds ~5-10min
total vs 5min standard LightGBM).

Leakage discipline (preserved from node_0030):
  - Stateless FE computed once -- no target, no cross-row stats, no fitting.
  - KBinsDiscretizer, category factorize maps: fit on train-fold only (fit_in_fold).
  - TargetEncoder: fit on train-fold only (fit_in_fold).
  - Frozen folds.json; no refitting of folds.

Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, features.txt.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
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

CHEAP_KILL_THRESHOLD = 0.962  # fold-0 BA must beat this

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


# ─── Focal loss custom objective ─────────────────────────────────────────────

def focal_objective_factory_2d(gamma: float, n_classes: int = N_CLASSES):
    """
    Returns an optimized LightGBM 4.x custom objective for multiclass focal loss.

    LightGBM 4.x Booster.update(fobj=...) signature:
      preds: numpy 2D array (n_samples, n_classes) -- raw margin (before softmax)
      train_data: Dataset
      returns: (grad, hess) both 2D (n_samples, n_classes) float64

    Formula (Lin et al. 2017 + Mukhoti et al. 2020):
      p = softmax(raw)    [n x K]
      pt = p[i, label[i]]
      g_k = (1-pt)^(gamma-1) * (p_k - y_k) * [(1-pt) - gamma*pt*log(pt)]
      h_k = 2 * p_k * (1 - p_k)  [diagonal hessian approx]
    """
    eps = np.float64(1e-7)
    g_minus_1 = float(max(gamma - 1.0, 0.0))

    def focal_obj(preds: np.ndarray, train_data) -> tuple:
        # preds: (n, K) float64 raw margins
        labels = train_data.get_label().astype(np.int32)
        n = len(labels)

        # Numerically stable softmax
        raw_s = preds - preds.max(axis=1, keepdims=True)
        np.exp(raw_s, out=raw_s)
        raw_s /= raw_s.sum(axis=1, keepdims=True)
        p = raw_s  # (n, K)

        # Per-sample true-class probability
        idx = np.arange(n)
        pt = p[idx, labels].copy()
        np.clip(pt, eps, 1.0, out=pt)

        # Focal scale factor per sample
        one_minus_pt = 1.0 - pt
        log_pt = np.log(pt)
        w_base = np.ones(n, dtype=np.float64) if g_minus_1 == 0.0 else one_minus_pt ** g_minus_1
        correction = one_minus_pt - gamma * pt * log_pt
        scale = w_base * correction  # (n,)

        # Gradient: scale[:, None] * (p - y_onehot)
        grad_mat = p * scale[:, None]     # scale * p_k for all k
        grad_mat[idx, labels] -= scale    # subtract scale for true class only

        # Hessian: diagonal approximation
        hess_mat = np.maximum(2.0 * p * (1.0 - p), 1e-6)

        return grad_mat, hess_mat

    return focal_obj


# ─── Feature engineering ─────────────────────────────────────────────────────

def stateless_fe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pure row-wise / stateless feature engineering -- safe to apply to the full
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

    # Original categorical columns -- low cardinality
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
        for dset in [va, te]:
            floored = np.floor(dset[col]).astype("float32")
            codes = factorize_transform(floored, uniques)
            dset[cat_name] = pd.Series(codes, index=dset.index).astype("int32")

    # Delta quantile bins
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


def raw_to_proba(raw: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis. raw is (n, K) float64."""
    rs = raw - raw.max(axis=-1, keepdims=True)
    exp_r = np.exp(rs)
    return (exp_r / exp_r.sum(axis=-1, keepdims=True)).astype("float32")


def train_fold_focal(
    X_tr, y_tr, X_val, y_val, X_te,
    lgbm_cat_cols: list,
    fold_seed: int,
    gamma: float,
    n_estimators: int = 2000,
    early_stopping_rounds: int = 150,
) -> tuple:
    """
    Train a LightGBM model with multiclass focal loss using lgb.train() with
    the objective callable passed via params dict (LightGBM 4.x API).

    In LightGBM 4.x, you can pass a Python callable as params['objective'].
    lgb.train then calls it with (preds_2d, dataset) each iteration and uses
    the returned (grad, hess) for the tree update.  The feval function also
    receives 2D raw margins (same format as the custom objective).

    Returns (val_proba, test_proba, best_iteration, model).
    """
    focal_obj = focal_objective_factory_2d(gamma=gamma, n_classes=N_CLASSES)

    def ba_feval(preds, dataset):
        """feval: preds is 2D (n, K) raw margins when custom objective is used."""
        labels = dataset.get_label().astype(int)
        p = raw_to_proba(preds)
        ba = balanced_accuracy_score(labels, p.argmax(axis=1))
        return "ba", ba, True   # higher is better

    params = dict(
        objective=focal_obj,   # Python callable -- LightGBM 4.x supports this in params
        num_class=N_CLASSES,
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
        n_jobs=-1,
        seed=fold_seed,
        verbosity=-1,
        metric="None",          # suppress built-in metrics; we use feval
        num_threads=-1,
    )

    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature=lgbm_cat_cols,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val, label=y_val,
        reference=dtrain,
        categorical_feature=lgbm_cat_cols,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=200),
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dval],
        feval=ba_feval,
        callbacks=callbacks,
    )

    best_iter = model.best_iteration

    # predict returns 2D (n, K) raw margins when objective is custom
    val_raw = model.predict(X_val, raw_score=True)
    val_proba = raw_to_proba(val_raw)

    te_raw = model.predict(X_te, raw_score=True)
    te_proba = raw_to_proba(te_raw)

    return val_proba, te_proba, best_iter, model


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

# ─── PRE-FLIGHT LEAKAGE CHECKS ────────────────────────────────────────────────
log("Pre-flight leakage checks ...")
# 1+2. target/id not in features
feature_names_check = [c for c in train_raw.columns if c not in [IDC, TARGET]]
assert TARGET not in feature_names_check, f"Target {TARGET!r} in features!"
assert IDC not in feature_names_check, f"ID {IDC!r} in features!"
log("  check 1+2: target/id absent from features -- OK")

# 3. single-feature sweep (numeric cols only, 50k sample)
_samp = train_raw.sample(min(50_000, n_train), random_state=0)
_ys = pd.factorize(_samp[TARGET])[0]
for _c in feature_names_check:
    _x = pd.to_numeric(_samp[_c], errors="coerce")
    if _x.nunique() > 1:
        _corr = abs(np.corrcoef(_x.fillna(_x.mean()), _ys)[0, 1])
        if _corr >= 0.999:
            raise SystemExit(f"LEAK: feature {_c!r} corr={_corr:.4f} with target")
log("  check 3: single-feature sweep -- OK (no near-perfect corr)")

# 4. fit-inside-fold -- verified by code inspection (all transforms in fold loop)
log("  check 4: fit_in_fold verified by code inspection")

# 5. frozen folds
log(f"  check 5: folds loaded from folds.json ({len(folds_list)} folds) -- OK")

# 6. near-dup -- warn only for non-image tabular
log("  check 6: near-dup warn -- skipped (tabular, low risk)")
log("Pre-flight checks PASSED.")

# ─── Stateless FE ─────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw = train_raw.drop(columns=[IDC, TARGET])
X_test_raw = test_raw.drop(columns=[IDC])

X_stateless = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}  X_test_stateless={X_test_stateless.shape}")

# ─── GAMMA TUNING ON FOLD-0 ──────────────────────────────────────────────────
log("=== GAMMA TUNING ON FOLD-0 ===")
fi0 = folds_list[0]
val_idx_0 = np.asarray(fi0["val_idx"])
tr_idx_0 = np.setdiff1d(np.arange(n_train), val_idx_0)
fold_seed_0 = SEED + 100

(X_tr_f0, X_val_f0, X_te_f0,
 all_cat_cols_0, combo_names_0, local_map_0, lgbm_cat_cols_0) = fit_fold_categoricals(
    X_stateless.iloc[tr_idx_0].reset_index(drop=True),
    X_stateless.iloc[val_idx_0].reset_index(drop=True),
    X_test_stateless.copy(),
)
y_tr_f0 = y_all[tr_idx_0]
y_val_f0 = y_all[val_idx_0]

X_tr_f0, X_val_f0, X_te_f0, te_names_0 = add_target_encoding(
    X_tr_f0, y_tr_f0, X_val_f0, X_te_f0, combo_names_0, fold_seed_0
)
X_tr_f0 = X_tr_f0.reindex(sorted(X_tr_f0.columns), axis=1)
X_val_f0 = X_val_f0.reindex(sorted(X_val_f0.columns), axis=1)
X_te_f0 = X_te_f0.reindex(sorted(X_te_f0.columns), axis=1)

gamma_candidates = [1.0, 2.0, 3.0]
gamma_results = {}
gamma_tune_t0 = time.perf_counter()

for gamma_cand in gamma_candidates:
    log(f"  Tuning gamma={gamma_cand} on fold-0 ...")
    t_g = time.perf_counter()
    val_p, _, best_it, _ = train_fold_focal(
        X_tr_f0, y_tr_f0, X_val_f0, y_val_f0, X_te_f0,
        lgbm_cat_cols=lgbm_cat_cols_0,
        fold_seed=fold_seed_0,
        gamma=gamma_cand,
    )
    ba_g = balanced_accuracy_score(y_val_f0, val_p.argmax(axis=1))
    gamma_results[gamma_cand] = {"ba": ba_g, "best_iter": best_it}
    log(f"    gamma={gamma_cand}: fold-0 BA={ba_g:.6f}  best_iter={best_it}  ({time.perf_counter()-t_g:.1f}s)")

best_gamma = max(gamma_results, key=lambda g: gamma_results[g]["ba"])
best_fold0_ba = gamma_results[best_gamma]["ba"]
log(f"Best gamma={best_gamma} (fold-0 BA={best_fold0_ba:.6f})")
print(f"gamma_tune: best_gamma={best_gamma} fold0_ba={best_fold0_ba:.6f}", flush=True)

# ─── CHEAP-KILL CHECK ─────────────────────────────────────────────────────────
if best_fold0_ba < CHEAP_KILL_THRESHOLD:
    log(f"CHEAP-KILL: fold-0 BA={best_fold0_ba:.6f} < {CHEAP_KILL_THRESHOLD} -- stopping")
    print(f"CHEAP_KILL fold0_ba={best_fold0_ba:.6f} threshold={CHEAP_KILL_THRESHOLD}", flush=True)
    sys.exit(0)

log(f"Fold-0 BA={best_fold0_ba:.6f} clears cheap-kill ({CHEAP_KILL_THRESHOLD}); proceeding with gamma={best_gamma}")

# ─── FULL 5-FOLD OOF LOOP ────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
all_cols_final = None
best_iters = []

log(f"Starting full OOF loop with gamma={best_gamma} ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
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

    if all_cols_final is None:
        all_cols_final = list(X_tr_fold.columns)
        log(f"  n_features={X_tr_fold.shape[1]}  lgbm_cat={lgbm_cat_cols}")

    val_proba, te_proba, best_iter, model = train_fold_focal(
        X_tr_fold, y_tr_fold, X_val_fold, y_val_fold, X_te_fold,
        lgbm_cat_cols=lgbm_cat_cols,
        fold_seed=fold_seed,
        gamma=best_gamma,
    )

    best_iters.append(best_iter)
    log(f"  best_iteration={best_iter}")

    oof_proba[val_idx] = val_proba.astype("float32")
    test_proba_accum += te_proba.astype("float32") / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, val_proba.argmax(axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del model, X_tr_fold, X_val_fold, X_te_fold
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
