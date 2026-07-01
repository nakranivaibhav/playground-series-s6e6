"""node_0049 — asymmetric binary chain GALAXY-then-QSO/STAR on fs_realmlp_fe.

THE ONE ATOMIC CHANGE vs node_0030:
  Replace the single 3-class LightGBM head with a 2-stage ASYMMETRIC BINARY CHAIN,
  both stages built ENTIRELY INSIDE each frozen fold:
    Stage 1 (full train-fold): GALAXY(0) vs NOT-GALAXY(1=QSO∪STAR)
             → blend 0.60*LGBM + 0.40*XGB; p1 = P(not-galaxy)
    Stage 2 (non-GALAXY train-fold rows only): QSO(0) vs STAR(1)
             → same LGBM+XGB blend recipe; p2 = P(STAR | not-galaxy)
  COMPOSE: P(GALAXY)=1-p1 ; P(QSO)=p1*(1-p2) ; P(STAR)=p1*p2
  Clip to [1e-7,1] then renorm. OOF: held-out fold p1+p2 composed softly.
  TEST: mean of per-fold composed test probs across all 5 folds.

  Hyperparams from RECIPE.md:
    LGBM: objective=binary, lr=0.03, num_leaves=230, max_depth=6, subsample=0.8,
          colsample=0.6, reg_alpha=3.7, reg_lambda=4.7, n_est=10000, ES=150.
    XGB:  objective=binary:logistic, lr=0.015, max_depth=8, reg_alpha=0.5,
          reg_lambda=2.5, subsample=0.8, colsample=0.7, n_est=10000, ES=150.
          (XGBoost 3.x: early_stopping_rounds in constructor, not fit())

  Also reports:
    1) standalone composed-OOF balanced accuracy with DE per-class threshold (mean±sem).
    2) re-stack A/B: CORE15 + this node as 16th base vs champion node_0041 (0.969808).

Leakage discipline (verified for chain):
  - Stateless FE applied once before fold loop — no labels, no fitting. Safe.
  - KBinsDiscretizer, factorize maps, TargetEncoder: fit on TRAIN-FOLD rows only.
  - Stage 1: fit on the FULL train-fold (GALAXY+QSO+STAR); inner ES val carved
    from train-fold (never from the held-out OOF fold). Same model predicts both
    val and test — no separate refit.
  - Stage 2: fit on non-GALAXY rows of the TRAIN-FOLD only (true class != GALAXY,
    using TRAIN-fold labels — fit_in_fold). Same model predicts all held-out rows
    (soft composition gates via multiplicative rule — no hard-threshold).
  - Frozen folds.json used throughout; no refitting of folds.

CLASSES order = ["GALAXY", "QSO", "STAR"] — matches node_0030.

Outputs:
  oof.npy          (N_train, 3)  composed 3-class probs
  test_probs.npy   (N_test,  3)  mean composed probs over 5 folds
  submission.csv               standalone argmax submission
  features.txt                 features used by stage-1 model
"""
from __future__ import annotations

import gc
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder
from xgboost import XGBClassifier

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


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
NC = N_CLASSES

GALAXY_IDX = LABEL_MAP["GALAXY"]   # 0
QSO_IDX    = LABEL_MAP["QSO"]      # 1
STAR_IDX   = LABEL_MAP["STAR"]     # 2

# CORE15 bases (same as node_0041 / node_0047)
BASES = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030",
    "node_0039",
]

# ─── Feature engineering (byte-identical to node_0030) ──────────────────────
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
    """Pure row-wise stateless FE — safe to apply before any fold split."""
    df = df.copy()
    df["_g_div_redshift"] = (df["g"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    df["_i_div_redshift"] = (df["i"] / (df["redshift"] + 1e-6)).replace(
        [np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    for a, b in COLOR_PAIRS:
        df[f"_{a}-{b}"] = (df[a] - df[b]).astype("float32")
    mags = df[["u", "g", "r", "i", "z"]].astype("float32")
    df["_mag_mean"] = mags.mean(axis=1).astype("float32")
    df["_mag_range"] = (mags.max(axis=1) - mags.min(axis=1)).astype("float32")
    shifted_rs = df["redshift"].astype("float32") - min(0.0, float(df["redshift"].min())) + 1e-4
    df["_log1p_redshift"] = np.log1p(shifted_rs).astype("float32")
    return df


def fit_fold_categoricals(df_tr, df_val, df_te):
    """Fit categorical encodings on train-fold only (fit_in_fold)."""
    local_map: dict = {}

    def factorize_fit(series):
        codes, uniques = pd.factorize(series, sort=False)
        return codes.astype("int32"), uniques

    def factorize_transform(series, uniques):
        code_map = {cat: i for i, cat in enumerate(uniques)}
        return series.map(code_map).fillna(-1).astype("int32")

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

    lgbm_cat_cols = BASE_CAT_COLS[:]
    all_new_cols = (
        BASE_CAT_COLS
        + [f"{c}_cat_" for c in BASE_NUM_COLS]
        + [f"delta_{n}_quantile_bin_" for n in [100, 500]]
        + combo_names
    )
    all_new_cols = [c for c in all_new_cols if c in tr.columns]
    return tr, va, te, all_new_cols, combo_names, local_map, lgbm_cat_cols


def add_target_encoding_binary(X_tr, y_tr, X_val, X_te, combo_names, fold_seed):
    """Binary TargetEncoder fit on train fold only (fit_in_fold)."""
    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    try:
        encoder = TargetEncoder(target_type="binary", cv=5, smooth="auto",
                                shuffle=True, random_state=fold_seed)
    except TypeError:
        encoder = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=fold_seed)
    tr_enc = encoder.fit_transform(X_tr[combo_names], y_tr)
    val_enc = encoder.transform(X_val[combo_names])
    tst_enc = encoder.transform(X_te[combo_names])
    te_names = [f"_{col}_TE_bin" for col in combo_names]
    X_tr[te_names] = np.asarray(tr_enc, dtype="float32")
    X_val[te_names] = np.asarray(val_enc, dtype="float32")
    X_te[te_names] = np.asarray(tst_enc, dtype="float32")
    return X_tr, X_val, X_te, te_names


def fit_binary_blend_dual(X_tr, y_tr, X_val_es, y_val_es, X_query1, X_query2, fold_seed):
    """
    Fit LGBM+XGB blend on (X_tr, y_tr) with early-stop on (X_val_es, y_val_es).
    Returns (p_query1, p_query2, best_lgbm, best_xgb) — P(positive class).
    X_val_es is carved from inside the train-fold only (never the held-out OOF fold).
    """
    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves=230,
        max_depth=6,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_alpha=3.7,
        reg_lambda=4.7,
        n_jobs=-1,
        random_state=fold_seed,
        verbosity=-1,
        device="cpu",
    )
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val_es, y_val_es)],
        eval_metric="binary_logloss",
        callbacks=[
            early_stopping(stopping_rounds=150, verbose=False),
            log_evaluation(period=9999),
        ],
    )
    pL_q1 = lgbm.predict_proba(X_query1)[:, 1].astype("float32")
    pL_q2 = lgbm.predict_proba(X_query2)[:, 1].astype("float32")
    best_lgbm = lgbm.best_iteration_
    del lgbm; gc.collect()

    xgb = XGBClassifier(
        objective="binary:logistic",
        n_estimators=10000,
        learning_rate=0.015,
        max_depth=8,
        reg_alpha=0.5,
        reg_lambda=2.5,
        subsample=0.8,
        colsample_bytree=0.7,
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        random_state=fold_seed,
        verbosity=0,
        eval_metric="logloss",
        early_stopping_rounds=150,
    )
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val_es, y_val_es)],
        verbose=False,
    )
    pX_q1 = xgb.predict_proba(X_query1)[:, 1].astype("float32")
    pX_q2 = xgb.predict_proba(X_query2)[:, 1].astype("float32")
    best_xgb = xgb.best_iteration
    del xgb; gc.collect()

    p_q1 = 0.60 * pL_q1 + 0.40 * pX_q1
    p_q2 = 0.60 * pL_q2 + 0.40 * pX_q2
    return p_q1, p_q2, best_lgbm, best_xgb


def compose(p1, p2):
    """Compose binary chain probs to 3-class: GALAXY=1-p1, QSO=p1*(1-p2), STAR=p1*p2."""
    p_galaxy = 1.0 - p1
    p_qso    = p1 * (1.0 - p2)
    p_star   = p1 * p2
    probs = np.stack([p_galaxy, p_qso, p_star], axis=1).astype("float32")
    probs = np.clip(probs, 1e-7, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


# ─── Stack / threshold helpers (from node_0047) ──────────────────────────────
def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))


def score_fn(y_true, y_pred) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))


def fit_meta(Xtr, ytr) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m


def best_thr_de(probs, labels) -> np.ndarray:
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)],
                                maxiter=40, tol=1e-7, seed=0, polish=False, workers=1)
    return np.array([r.x[0], r.x[1], 1.0])


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw  = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw   = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all   = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test  = len(test_raw)

# ─── Stateless FE ────────────────────────────────────────────────────────────
log("Applying stateless FE ...")
X_raw            = train_raw.drop(columns=[IDC, TARGET])
X_test_raw       = test_raw.drop(columns=[IDC])
X_stateless      = stateless_fe(X_raw)
X_test_stateless = stateless_fe(X_test_raw)
log(f"  X_stateless={X_stateless.shape}")

# ─── OOF arrays ──────────────────────────────────────────────────────────────
oof_proba        = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test,  N_CLASSES), dtype=np.float32)
per_fold_scores  = []
all_cols_final   = None

log("Starting BINARY CHAIN OOF loop ...")
fold_t0 = time.perf_counter()

for fi in folds_list:
    fold_id   = fi["fold"]
    val_idx   = np.asarray(fi["val_idx"])
    tr_idx    = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"\nFold {fold_id}: train={len(tr_idx)}  val={len(val_idx)}")

    y_tr  = y_all[tr_idx]
    y_val = y_all[val_idx]

    # Categorical + TE encodings — fit_in_fold with stage-1 binary target
    y_tr_s1 = (y_tr != GALAXY_IDX).astype(int)

    X_tr_fold, X_val_fold, X_te_fold, all_cat_cols, combo_names, local_map, lgbm_cat_cols = \
        fit_fold_categoricals(
            X_stateless.iloc[tr_idx].reset_index(drop=True),
            X_stateless.iloc[val_idx].reset_index(drop=True),
            X_test_stateless.copy(),
        )

    X_tr_fold, X_val_fold, X_te_fold, te_names = add_target_encoding_binary(
        X_tr_fold, y_tr_s1, X_val_fold, X_te_fold, combo_names, fold_seed
    )

    X_tr_fold  = X_tr_fold.reindex(sorted(X_tr_fold.columns),  axis=1)
    X_val_fold = X_val_fold.reindex(sorted(X_val_fold.columns), axis=1)
    X_te_fold  = X_te_fold.reindex(sorted(X_te_fold.columns),  axis=1)

    if all_cols_final is None:
        all_cols_final = list(X_tr_fold.columns)
        log(f"  n_features={X_tr_fold.shape[1]}")

    # Inner val for early stopping — carved from TRAIN-FOLD only (10%)
    rng = np.random.RandomState(fold_seed)
    inner_size = max(int(0.10 * len(tr_idx)), 1000)
    es_local   = rng.choice(len(tr_idx), size=inner_size, replace=False)
    fit_local  = np.setdiff1d(np.arange(len(tr_idx)), es_local)

    X_es   = X_tr_fold.iloc[es_local].reset_index(drop=True)
    y_es   = y_tr_s1[es_local]
    X_fit  = X_tr_fold.iloc[fit_local].reset_index(drop=True)
    y_fit  = y_tr_s1[fit_local]

    # ── Stage 1: GALAXY(0) vs NOT-GALAXY(1) ─────────────────────────────────
    # Fit on train-fold (minus inner ES); predict both OOF val and test simultaneously
    log(f"  Stage-1: fit n={len(fit_local)}, es n={len(es_local)}")
    p1_val, p1_test, b_lgbm1, b_xgb1 = fit_binary_blend_dual(
        X_fit, y_fit, X_es, y_es, X_val_fold, X_te_fold, fold_seed
    )
    log(f"  Stage-1 best iters: lgbm={b_lgbm1}  xgb={b_xgb1}")

    # ── Stage 2: QSO(0) vs STAR(1) — non-GALAXY train-fold rows only ────────
    nongal_mask   = (y_tr != GALAXY_IDX)   # train-fold labels — fit_in_fold safe
    nongal_count  = nongal_mask.sum()
    log(f"  Stage-2: non-GALAXY train rows={nongal_count}")

    X_tr_s2 = X_tr_fold.iloc[nongal_mask].reset_index(drop=True)
    y_tr_s2 = (y_tr[nongal_mask] == STAR_IDX).astype(int)

    rng2         = np.random.RandomState(fold_seed + 99)
    es2_size     = max(int(0.10 * nongal_count), 500)
    es2_size     = min(es2_size, nongal_count - 200)
    es2_local    = rng2.choice(nongal_count, size=es2_size, replace=False)
    fit2_local   = np.setdiff1d(np.arange(nongal_count), es2_local)

    X_es2  = X_tr_s2.iloc[es2_local].reset_index(drop=True)
    y_es2  = y_tr_s2[es2_local]
    X_fit2 = X_tr_s2.iloc[fit2_local].reset_index(drop=True)
    y_fit2 = y_tr_s2[fit2_local]

    # Stage-2 predicts ALL held-out val rows (soft composition gate)
    log(f"  Stage-2: fit n={len(fit2_local)}, es n={len(es2_local)}")
    p2_val, p2_test, b_lgbm2, b_xgb2 = fit_binary_blend_dual(
        X_fit2, y_fit2, X_es2, y_es2, X_val_fold, X_te_fold, fold_seed + 10
    )
    log(f"  Stage-2 best iters: lgbm={b_lgbm2}  xgb={b_xgb2}")

    # ── Compose 3-class probs ────────────────────────────────────────────────
    val_proba  = compose(p1_val,  p2_val)
    test_proba = compose(p1_test, p2_test)

    oof_proba[val_idx] = val_proba
    test_proba_accum  += test_proba / len(folds_list)

    fold_score = balanced_accuracy_score(y_val, np.argmax(val_proba, axis=1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    del X_tr_fold, X_val_fold, X_te_fold, X_tr_s2
    gc.collect()

    if fold_id == 0:
        fold_time = time.perf_counter() - fold_t0
        projected = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected:.1f}s  ({projected/60:.1f}min)")

mean_cv = float(np.mean(per_fold_scores))
sem_cv  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"STANDALONE raw-argmax cv={mean_cv:.6f}+/-{sem_cv:.6f}")

# ─── DE threshold on composed OOF ────────────────────────────────────────────
log("Running DE per-class threshold optimization on composed OOF ...")
de_per_fold = []
for i, fi in enumerate(folds_list):
    val_i = np.asarray(fi["val_idx"])
    other = np.setdiff1d(np.arange(n_train), val_i)
    w    = best_thr_de(oof_proba[other], y_all[other])
    pred = np.argmax(oof_proba[val_i] * w, axis=1)
    s    = score_fn(y_all[val_i], pred)
    de_per_fold.append(s)
    log(f"  fold {i}: DE BA={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

de_mean = float(np.mean(de_per_fold))
de_sem  = float(np.std(de_per_fold, ddof=1) / np.sqrt(len(de_per_fold)))
log(f"STANDALONE DE-threshold cv={de_mean:.6f}+/-{de_sem:.6f}")
print(f"cv={de_mean:.6f}", flush=True)

# ─── Save OOF + test probs ───────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy",        oof_proba)
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved oof.npy={oof_proba.shape}  test_probs.npy={test_proba_accum.shape}")

# ─── Write submission ────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")
log(f"Submission class distribution:\n{sub[TARGET].value_counts().to_string()}")

# ─── Write features.txt ───────────────────────────────────────────────────────
(NODE_SRC / "features.txt").write_text("\n".join(sorted(all_cols_final)) + "\n")
log(f"Wrote features.txt ({len(all_cols_final)} features)")

# ─── OOF full BA ──────────────────────────────────────────────────────────────
oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy (argmax)={oof_metric:.6f}")

# ─── Re-stack A/B: CORE15 + node_0049 as 16th base ──────────────────────────
log("Loading CORE15 OOF + test probs for re-stack ...")
nodes_dir = COMP_DIR / "nodes"

OOF_CORE = np.concatenate(
    [logp(np.load(nodes_dir / b / "oof.npy")) for b in BASES], axis=1
)
TEST_CORE = np.concatenate(
    [logp(np.load(nodes_dir / b / "test_probs.npy")) for b in BASES], axis=1
)
log(f"  CORE15 OOF={OOF_CORE.shape}  TEST={TEST_CORE.shape}")

node49_oof_log  = logp(oof_proba)
node49_test_log = logp(test_proba_accum)

OOF_STACK  = np.concatenate([OOF_CORE,  node49_oof_log],  axis=1)
TEST_STACK = np.concatenate([TEST_CORE, node49_test_log], axis=1)
log(f"  stacked OOF={OOF_STACK.shape}  ({len(BASES)+1} bases)")

fval = [np.asarray(f["val_idx"]) for f in folds_list]

stack_oof = np.zeros((n_train, NC), dtype=np.float32)
for vi in fval:
    tr_s = np.setdiff1d(np.arange(n_train), vi)
    m    = fit_meta(OOF_STACK[tr_s], y_all[tr_s])
    stack_oof[vi] = m.predict_proba(OOF_STACK[vi])

restack_per_fold = []
for i, vi in enumerate(fval):
    other = np.setdiff1d(np.arange(n_train), vi)
    w    = best_thr_de(stack_oof[other], y_all[other])
    pred = np.argmax(stack_oof[vi] * w, axis=1)
    s    = score_fn(y_all[vi], pred)
    restack_per_fold.append(s)
    log(f"  re-stack fold {i}: score={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")
    print(f"restack_fold{i}={s:.6f}", flush=True)

rs_mean = float(np.mean(restack_per_fold))
rs_sem  = float(np.std(restack_per_fold, ddof=1) / np.sqrt(len(restack_per_fold)))
log(f"RE-STACK CORE15+n49 cv={rs_mean:.6f}+/-{rs_sem:.6f}  (champ 0.969808)")
print(f"restack_cv={rs_mean:.6f}", flush=True)

total_elapsed = time.perf_counter() - T0
log(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
log("Done.")
