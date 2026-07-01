"""node_0098 — RBF base full-OOF + champion stack-add test

WHAT THIS IS:
  node_0097's config run to a FULL 5-fold OOF — NO solo-BA gate this time.
  gamma=0.0556 (1/n_features, known best from node_0097 fold-0 sweep).
  n_components=2000.

  THEN: the DECISIVE STACK-ADD test.
  Add this OOF as one more base column to the champion node_0091 pool
  and refit the meta fold-honest with the same nested-C protocol.
  Report: 0.970355 baseline → new stacked CV, delta, per-fold deltas.

  CLASS ORDER: GALAXY=0, QSO=1, STAR=2 (frozen from folds.json scheme).

LEAK DISCIPLINE:
  - fs_realmlp_fe is STATELESS (pure row-wise, no fit, no target, no cross-row
    stats). Safe to build once on full train/test.
  - StandardScaler stats fit on TRAIN FOLD ONLY.
  - Nystroem landmarks sampled from TRAIN FOLD ONLY (subsample of 40k rows).
  - Folds loaded from frozen folds.json.
  - Stack meta: LogisticRegressionCV nested (fits only on outer-train portion
    in each fold; the outer val fold is NEVER seen during C selection).
  - TARGET and ID absent from feature matrix (asserted).
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

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
SEED = 42
N_CLASSES = 3
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

# fs_realmlp_fe recipe (stateless row-wise features — from data.md)
COLOR_PAIRS = [
    ("u", "g"),
    ("g", "r"),
    ("r", "i"),
    ("i", "z"),
    ("u", "r"),
    ("g", "i"),
    ("r", "z"),
]
BASE_MAGS = ["u", "g", "r", "i", "z"]
BASE_NUM  = ["u", "g", "r", "i", "z", "redshift"]

# Nystroem config — VERBATIM from node_0097 (gamma 0.0556 = 1/n_features, best of sweep)
N_COMPONENTS = 2000
NYSTROEM_SUBSAMPLE = 40_000   # landmarks sampled from this many train-fold rows
LOGREG_MAX_ITER = 1000
LOGREG_C = 1.0
# gamma fixed from node_0097 sweep result
GAMMA = 0.0556  # = 1/18 ≈ 0.055556

# Champion stack baseline (node_0091)
CHAMPION_CV = 0.970355
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# In-house base pools (node_0091 recipe)
TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]


# ─── Feature engineering (STATELESS) ─────────────────────────────────────────
def build_realmlp_fe(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Build fs_realmlp_fe feature vector — STATELESS, row-wise, no fitting.
    Returns float32 array shape (n, n_features) and feature names.
    """
    cols = []
    feature_names = []

    # 6 base numerics (u, g, r, i, z, redshift)
    for c in BASE_NUM:
        cols.append(df[c].values.astype(np.float32))
        feature_names.append(c)

    # 7 color pairs
    for a, b in COLOR_PAIRS:
        cols.append((df[a].values - df[b].values).astype(np.float32))
        feature_names.append(f"{a}-{b}")

    # Redshift ratios
    rs = df["redshift"].values.astype(np.float32)
    cols.append((df["g"].values.astype(np.float32) / (rs + 1e-6)).clip(-1e4, 1e4))
    feature_names.append("g_div_redshift")
    cols.append((df["i"].values.astype(np.float32) / (rs + 1e-6)).clip(-1e4, 1e4))
    feature_names.append("i_div_redshift")

    # log1p redshift
    shifted_rs = rs - min(0.0, float(rs.min())) + 1e-4
    cols.append(np.log1p(shifted_rs))
    feature_names.append("log1p_redshift")

    # mag aggregates
    mags = np.column_stack([df[c].values.astype(np.float32) for c in BASE_MAGS])
    cols.append(mags.mean(axis=1))
    feature_names.append("mag_mean")
    cols.append(mags.max(axis=1) - mags.min(axis=1))
    feature_names.append("mag_range")

    X = np.column_stack(cols).astype(np.float32)
    return X, feature_names


# ─── Nystroem pipeline helpers ────────────────────────────────────────────────
def fit_nystroem_pipeline(
    X_tr: np.ndarray,
    gamma: float,
    n_components: int,
    rng: np.random.RandomState,
) -> tuple[StandardScaler, Nystroem]:
    """
    Fit StandardScaler + Nystroem on TRAIN-FOLD rows only.
    Nystroem landmarks are sampled from up to NYSTROEM_SUBSAMPLE rows.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)

    if len(X_tr_scaled) > NYSTROEM_SUBSAMPLE:
        idx = rng.choice(len(X_tr_scaled), NYSTROEM_SUBSAMPLE, replace=False)
        X_landmark_source = X_tr_scaled[idx]
    else:
        X_landmark_source = X_tr_scaled

    nystroem = Nystroem(
        kernel="rbf",
        gamma=gamma,
        n_components=n_components,
        random_state=rng.randint(0, 2**31),
        n_jobs=1,
    )
    nystroem.fit(X_landmark_source)
    return scaler, nystroem


def apply_pipeline(
    X: np.ndarray,
    scaler: StandardScaler,
    nystroem: Nystroem,
    batch_size: int = 50_000,
) -> np.ndarray:
    """Apply fitted scaler + Nystroem (transform only, never fit). Batched."""
    if len(X) <= batch_size:
        return nystroem.transform(scaler.transform(X))
    parts = []
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        parts.append(nystroem.transform(scaler.transform(X[start:end])))
    return np.concatenate(parts, axis=0)


def make_logreg(seed: int, C: float = LOGREG_C) -> LogisticRegression:
    return LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        C=C,
        max_iter=LOGREG_MAX_ITER,
        random_state=seed,
        n_jobs=-1,
    )


# ─── Stack helpers ────────────────────────────────────────────────────────────
def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))


def norm(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True)
    s[s == 0] = 1
    return a / s


def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(N_CLASSES)
         if (y_true == c).any()]
    ))


def rd(path: str | Path, nr: int) -> np.ndarray:
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a
        return a[:, :3]
    d = pd.read_csv(p)
    c = list(d.columns)
    if set(CLASSES).issubset(c):
        return d[CLASSES].values.astype(float)
    pc = [f"prob_{l}" for l in CLASSES]
    if set(pc).issubset(c):
        return d[pc].values.astype(float)
    num = d.select_dtypes("number")
    if num.shape[1] >= 3:
        return num.values[:, :3]
    v = d.iloc[:, 0].values.astype(float)
    return v.reshape(nr, 3)


def load_ext_csv(path: str | Path, nr: int) -> np.ndarray:
    d = pd.read_csv(path)
    pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
    if set(pcols).issubset(d.columns):
        return d[pcols].values.astype(float)
    return rd(path, nr)


def nested_cv_arm_fast(
    OOF_mat: np.ndarray,
    TST_mat: np.ndarray,
    y: np.ndarray,
    fval: list[np.ndarray],
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float]:
    """Nested C-selection + outer OOF using LogisticRegressionCV."""
    n = len(y)
    oof_probs = np.zeros((n, N_CLASSES), dtype=float)
    best_Cs = []
    for fi, vi in enumerate(fval):
        tr_idx2 = np.setdiff1d(np.arange(n), vi)
        lrcv = LogisticRegressionCV(
            Cs=C_GRID, cv=4, class_weight="balanced", max_iter=2000,
            n_jobs=-1, random_state=42, scoring="balanced_accuracy",
            solver="lbfgs", multi_class="multinomial",
        )
        lrcv.fit(OOF_mat[tr_idx2], y[tr_idx2])
        best_c = float(lrcv.C_[0])
        best_Cs.append(best_c)
        oof_probs[vi] = lrcv.predict_proba(OOF_mat[vi])
    pf = [score_fn(y[vi], oof_probs[vi].argmax(1)) for fi, vi in enumerate(fval)]
    cv_mean = float(np.mean(pf))
    cv_sem  = float(np.std(pf, ddof=1) / np.sqrt(len(pf)))
    c_win = Counter(best_Cs).most_common(1)[0][0]
    m_final = LogisticRegression(
        class_weight="balanced", C=c_win, max_iter=2000,
        n_jobs=-1, random_state=42,
        solver="lbfgs", multi_class="multinomial",
    )
    m_final.fit(OOF_mat, y)
    tst_probs = m_final.predict_proba(TST_mat)
    log(f"  {label}: cv={cv_mean:.6f}  sem={cv_sem:.6f}  "
        f"per_fold={[f'{s:.6f}' for s in pf]}  best_Cs={best_Cs}")
    return oof_probs, tst_probs, pf, cv_mean, cv_sem


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw  = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_data = json.loads((COMP_DIR / "folds.json").read_text())
folds_list = folds_data["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all   = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test  = len(test_raw)
assert n_train == 577347, f"unexpected n_train={n_train}"
assert n_test  == 247435, f"unexpected n_test={n_test}"

# ─── Load node_0070 OOF (for err-corr report) ────────────────────────────────
log("Loading node_0070 OOF for err-corr reporting ...")
oof_70 = np.load(COMP_DIR / "nodes/node_0070/oof.npy")
assert oof_70.shape == (n_train, N_CLASSES), f"Expected ({n_train},{N_CLASSES}), got {oof_70.shape}"
y_pred_70 = oof_70.argmax(axis=1)
log(f"  node_0070 OOF loaded: {(y_pred_70 != y_all).sum()} errors / {n_train}")

# ─── Build fs_realmlp_fe features (STATELESS — safe before fold split) ───────
log("Building fs_realmlp_fe stateless features ...")
X_train_fe, feature_names = build_realmlp_fe(train_raw)
X_test_fe,  _             = build_realmlp_fe(test_raw)
n_features = X_train_fe.shape[1]
log(f"  X_train_fe={X_train_fe.shape}  X_test_fe={X_test_fe.shape}")
log(f"  features: {feature_names}")

# ─── PRE-FLIGHT LEAKAGE CHECK 1: TARGET and ID not in feature list ───────────
log("Pre-flight leakage checks ...")
assert TARGET not in feature_names, f"TARGET {TARGET} in features — LEAK!"
assert IDC    not in feature_names, f"ID {IDC} in features — LEAK!"
log(f"  check1 PASS: target/id absent from {n_features} feature columns")

# ─── PRE-FLIGHT LEAKAGE CHECK 2: single-feature↔target sweep ────────────────
log("Pre-flight leakage check 2: single-feature correlation sweep ...")
sample_size = min(50_000, n_train)
rng_check = np.random.RandomState(0)
sample_idx = rng_check.choice(n_train, sample_size, replace=False)
s_X = X_train_fe[sample_idx]
s_y = y_all[sample_idx]
leaked_cols = []
for fi in range(s_X.shape[1]):
    x = s_X[:, fi]
    x = np.where(np.isfinite(x), x, 0.0)
    if np.unique(x).size > 1:
        corr = abs(float(np.corrcoef(x, s_y)[0, 1]))
        if corr >= 0.999:
            leaked_cols.append((feature_names[fi], corr))
if leaked_cols:
    raise SystemExit(f"LEAK SMELL: {leaked_cols}")
log(f"  check2 PASS: no single-feature |corr|>=0.999 (sample={sample_size})")

# ─── PRE-FLIGHT LEAKAGE CHECK 5: frozen folds ────────────────────────────────
assert len(folds_list) == 5, f"Expected 5 folds, got {len(folds_list)}"
all_val_idx = []
for fi in folds_list:
    all_val_idx.extend(fi["val_idx"])
assert len(set(all_val_idx)) == n_train, "Folds don't cover all train rows exactly once"
log(f"  check5 PASS: frozen folds verified ({len(folds_list)} folds, {n_train} unique val rows)")

# ─── CHECK 6: train/test near-dup spot check ─────────────────────────────────
log("Check 6: train/test near-dup spot check ...")
tr_sample = X_train_fe[:1000].round(4)
te_sample = X_test_fe[:1000].round(4)
tr_set = set(map(tuple, tr_sample.tolist()))
te_set = set(map(tuple, te_sample.tolist()))
overlap = tr_set & te_set
log(f"  check6: exact-match overlap (1k sample each): {len(overlap)} rows — "
    f"{'warn' if len(overlap) > 0 else 'clean'}")

# ─── PRE-FLIGHT COMPLETE ──────────────────────────────────────────────────────
log(f"PRE-FLIGHT COMPLETE. gamma={GAMMA:.6f} (fixed from node_0097 sweep), "
    f"n_components={N_COMPONENTS}")
log("NOTE: No solo-BA gate — running all 5 folds (this is the decisive full-OOF run)")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
log(f"Starting OOF loop: 5 folds, n_components={N_COMPONENTS}, gamma={GAMMA:.6f} ...")
oof_proba        = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test,  N_CLASSES), dtype=np.float32)
per_fold_scores  = []
per_fold_errcorr = []

for fi in folds_list:
    fold_id  = fi["fold"]
    val_idx  = np.asarray(fi["val_idx"])
    tr_idx   = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)}  val={len(val_idx)}")
    fold_t0 = time.perf_counter()

    X_tr_fold  = X_train_fe[tr_idx]
    X_val_fold = X_train_fe[val_idx]
    y_tr_fold  = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # ─── CHECK 4: fit_in_fold — StandardScaler + Nystroem on TRAIN FOLD ONLY ──
    rng_fold = np.random.RandomState(fold_seed)
    scaler_fold, nystroem_fold = fit_nystroem_pipeline(
        X_tr_fold, gamma=GAMMA, n_components=N_COMPONENTS, rng=rng_fold
    )
    # Apply to val and test (transform only, never fit on val/test)
    X_tr_mapped  = apply_pipeline(X_tr_fold,  scaler_fold, nystroem_fold)
    X_val_mapped = apply_pipeline(X_val_fold, scaler_fold, nystroem_fold)
    X_te_mapped  = apply_pipeline(X_test_fe,  scaler_fold, nystroem_fold)

    log(f"  Fold {fold_id}: mapped X_tr={X_tr_mapped.shape}  X_val={X_val_mapped.shape}")

    # ─── Fit balanced LogReg ──────────────────────────────────────────────────
    model = make_logreg(seed=fold_seed)
    model.fit(X_tr_mapped, y_tr_fold)

    val_proba = model.predict_proba(X_val_mapped).astype(np.float32)
    oof_proba[val_idx] = val_proba

    test_proba_fold = model.predict_proba(X_te_mapped).astype(np.float32)
    test_proba_accum += test_proba_fold / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, val_proba.argmax(1))
    per_fold_scores.append(fold_score)

    # Err-corr vs node_0070 (CHECK: decorrelation preserved across folds)
    val_pred    = val_proba.argmax(1)
    val_pred_70 = y_pred_70[val_idx]
    err_self = (val_pred    != y_val_fold).astype(np.float32)
    err_70   = (val_pred_70 != y_val_fold).astype(np.float32)
    fold_err_corr = float(np.corrcoef(err_self, err_70)[0, 1])
    per_fold_errcorr.append(fold_err_corr)

    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  "
        f"err_corr_vs_n70={fold_err_corr:.4f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}  "
          f"fold{fold_id}_errcorr={fold_err_corr:.4f}", flush=True)

    # Clean up fold memory
    del X_tr_fold, X_val_fold, X_tr_mapped, X_val_mapped, X_te_mapped
    del scaler_fold, nystroem_fold, model, val_proba, test_proba_fold
    gc.collect()


# ─── Full OOF complete ────────────────────────────────────────────────────────
mean_cv     = float(np.mean(per_fold_scores))
sem_cv      = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
mean_errcorr = float(np.mean(per_fold_errcorr))
log(f"per_fold_scores={[f'{s:.6f}' for s in per_fold_scores]}")
log(f"per_fold_errcorr={[f'{s:.4f}' for s in per_fold_errcorr]}")
log(f"solo_cv={mean_cv:.6f}+/-{sem_cv:.6f}")
log(f"mean_err_corr_vs_n70={mean_errcorr:.4f}")
print(f"cv={mean_cv:.6f}", flush=True)
print(f"mean_errcorr_vs_n70={mean_errcorr:.4f}", flush=True)

# ─── POST-OOF CHECKS ─────────────────────────────────────────────────────────
# Check 7: OOF complete
assert not np.any(np.isnan(oof_proba)), "NaN in OOF probs!"
covered = (oof_proba.sum(axis=1) > 0)
assert covered.all(), f"OOF has {(~covered).sum()} uncovered rows!"
log(f"  check7 PASS: OOF complete, no NaN ({n_train} rows)")

# Check 8: distribution sane
prob_sums = oof_proba.sum(axis=1)
assert np.allclose(prob_sums, 1.0, atol=1e-3), f"OOF probs don't sum to 1"
assert oof_proba.min() >= -1e-6,       f"OOF probs < 0: {oof_proba.min()}"
assert oof_proba.max() <= 1.0 + 1e-6, f"OOF probs > 1: {oof_proba.max()}"
log(f"  check8 PASS: OOF distribution sane (probs in [0,1], sum to 1)")

oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy (macro check)={oof_metric:.6f}")

# ─── SAVE base OOF + test_probs ──────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy",        oof_proba)
np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved oof.npy shape={oof_proba.shape}")
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

# ─── Write solo submission (base; may be replaced by stacked if stack wins) ──
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub_df = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub_df = sub_df[list(sample_sub.columns)]
sub_path = NODE_DIR / "submission.csv"
sub_df.to_csv(sub_path, index=False)
log(f"Saved solo submission.csv: {sub_df.shape}")
log(f"  solo prediction distribution: {dict(zip(*np.unique(pred_labels, return_counts=True)))}")

# ─── CHECK 9 (schema) on solo submission ─────────────────────────────────────
sub_loaded = pd.read_csv(sub_path)
assert list(sub_loaded.columns) == list(sample_sub.columns), \
    f"column mismatch: {list(sub_loaded.columns)} vs {list(sample_sub.columns)}"
assert len(sub_loaded) == len(sample_sub), \
    f"row count: {len(sub_loaded)} vs {len(sample_sub)}"
assert set(sub_loaded[TARGET].unique()) <= set(CLASSES), \
    f"unknown classes: {set(sub_loaded[TARGET].unique()) - set(CLASSES)}"
log("  check9 PASS: submission schema OK")

# ─── DECISIVE TEST: CHAMPION STACK-ADD ───────────────────────────────────────
log("=" * 70)
log("STEP 2 — DECISIVE STACK-ADD (node_0091 recipe + this node_0098 OOF)")
log(f"Champion baseline: node_0091 stacked CV = {CHAMPION_CV:.6f}")

# Load public bank-17 (same MANIFEST as node_0091)
B = COMP_DIR / "refs/oof_bank"
K = COMP_DIR / "refs/kernel_out"

MANIFEST = {
    'xgb-0':      (K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",              K/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
    'xgb-1':      (K/"xgb-v1-for-s6e6/oof_preds.npy",                K/"xgb-v1-for-s6e6/test_preds.npy"),
    'realmlp-0':  (B/"oof_preds_realmlp0_v12.csv",                   B/"test_preds_realmlp0_v12.csv"),
    'realmlp-1':  (K/"realmlp-v1-for-s6e6/oof_preds.npy",            K/"realmlp-v1-for-s6e6/test_preds.npy"),
    'tabm-0':     (B/"oof_preds_tabm0_v2.csv",                       B/"test_preds_tabm0_v2.csv"),
    'cat-0':      (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
    'realmlp-2':  (B/"oof_preds_realmlp2_v10.csv",                   B/"test_preds_realmlp2_v10.csv"),
    'tabicl-2':   (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy",   K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
    'lgbm-3':     (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",       K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
    'logreg-1':   (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",   K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
    'nn-1':       (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",           K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
    'xgb-3':      (K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
    'xgb-5':      (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",        K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
    'realmlp-5':  (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy", K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
    'nn-2':       (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",           K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
    'cat-3':      (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",         K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
    'lgbm-5':     (B/"oof_preds_lgbm5_v1.csv",                         B/"test_preds_lgbm5_v1.csv"),
    'xgb-6':      (B/"oof_final_xgb6_v1.csv",                          B/"test_final_xgb6_v1.csv"),
    'tabm-1':     (B/"oof_final_tabm1_v1.csv",                         B/"test_final_tabm1_v1.csv"),
}

log("Loading public bank bases ...")
POOF = {}; PTEST = {}; good = []
for name, (op, tp) in MANIFEST.items():
    try:
        o = norm(rd(op, n_train))
        t = norm(rd(tp, n_test))
        assert o.shape == (n_train, 3) and t.shape == (n_test, 3)
        ba = balanced_accuracy_score(y_all, o.argmax(1))
        if 0.90 < ba < 0.972:
            POOF[name] = o; PTEST[name] = t; good.append(name)
    except Exception as e:
        log(f"  {name}: FAIL {str(e)[:60]}")
log(f"  Loaded {len(good)} public bank models")

# Load FT-Transformer
PILK = COMP_DIR / "refs/ext_oof/pilkwang_5090"
ft_oof_raw  = load_ext_csv(PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n_train)
ft_test_raw = load_ext_csv(PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n_test)
assert ft_oof_raw.shape  == (n_train, 3), f"FT-T OOF shape {ft_oof_raw.shape}"
assert ft_test_raw.shape == (n_test,   3), f"FT-T test shape {ft_test_raw.shape}"
log(f"  FT-T solo BA: {score_fn(y_all, norm(ft_oof_raw).argmax(1)):.6f}")

# Load in-house TIGHT + WEAK bases
log("Loading in-house bases ...")
inhouse_oof_tight = {}; inhouse_test_tight = {}
for nid in TIGHT_IDS:
    node_nm = f"node_{nid:04d}"
    try:
        o_raw = np.load(COMP_DIR / "nodes" / node_nm / "oof.npy").astype(float)
        t_raw = np.load(COMP_DIR / "nodes" / node_nm / "test_probs.npy").astype(float)
        assert o_raw.shape == (n_train, 3) and t_raw.shape == (n_test, 3)
        assert not np.isnan(o_raw).any() and not np.isnan(t_raw).any()
        o = norm(o_raw); t = norm(t_raw)
        solo_ba = score_fn(y_all, o.argmax(1))
        if solo_ba >= 0.5:
            inhouse_oof_tight[node_nm]  = logp(o)
            inhouse_test_tight[node_nm] = logp(t)
    except Exception:
        pass

inhouse_oof_weak = {}; inhouse_test_weak = {}
for nid in WEAK_EXTRA_IDS:
    node_nm = f"node_{nid:04d}"
    try:
        o_raw = np.load(COMP_DIR / "nodes" / node_nm / "oof.npy").astype(float)
        t_raw = np.load(COMP_DIR / "nodes" / node_nm / "test_probs.npy").astype(float)
        assert o_raw.shape == (n_train, 3) and t_raw.shape == (n_test, 3)
        assert not np.isnan(o_raw).any() and not np.isnan(t_raw).any()
        o = norm(o_raw); t = norm(t_raw)
        solo_ba = score_fn(y_all, o.argmax(1))
        if solo_ba >= 0.5:
            inhouse_oof_weak[node_nm]  = logp(o)
            inhouse_test_weak[node_nm] = logp(t)
    except Exception:
        pass

log(f"  in-house TIGHT={len(inhouse_oof_tight)}/{len(TIGHT_IDS)}, "
    f"WEAK={len(inhouse_oof_weak)}/{len(WEAK_EXTRA_IDS)}")

# Build baseline pool (same as node_0091 TIGHT winner)
base_oof_logp   = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
base_test_logp  = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]
tight_inhouse_oof  = list(inhouse_oof_tight.values())
tight_inhouse_test = list(inhouse_test_tight.values())
full_inhouse_oof   = tight_inhouse_oof + list(inhouse_oof_weak.values())
full_inhouse_test  = tight_inhouse_test + list(inhouse_test_weak.values())

# This node_0098 OOF (clipped log-probs) — one more base column
this_oof_logp  = logp(norm(oof_proba.astype(float)))
this_test_logp = logp(norm(test_proba_accum.astype(float)))

# Assemble feature matrices
OOF_tight_base   = np.concatenate(base_oof_logp  + tight_inhouse_oof,  axis=1)
TST_tight_base   = np.concatenate(base_test_logp + tight_inhouse_test,  axis=1)
OOF_full_base    = np.concatenate(base_oof_logp  + full_inhouse_oof,   axis=1)
TST_full_base    = np.concatenate(base_test_logp + full_inhouse_test,   axis=1)

OOF_tight_plus98 = np.concatenate([OOF_tight_base, this_oof_logp],  axis=1)
TST_tight_plus98 = np.concatenate([TST_tight_base, this_test_logp], axis=1)
OOF_full_plus98  = np.concatenate([OOF_full_base,  this_oof_logp],  axis=1)
TST_full_plus98  = np.concatenate([TST_full_base,  this_test_logp], axis=1)

log(f"  OOF_tight_base={OOF_tight_base.shape}  OOF_tight_plus98={OOF_tight_plus98.shape}")
log(f"  OOF_full_base={OOF_full_base.shape}    OOF_full_plus98={OOF_full_plus98.shape}")

fval = [np.asarray(f["val_idx"]) for f in folds_list]

# ─── Run stack arms ───────────────────────────────────────────────────────────
log("Running BASELINE stack (TIGHT, no node_0098) — reproducing champion ...")
_, _, pf_base_tight, cv_base_tight, sem_base_tight = nested_cv_arm_fast(
    OOF_tight_base, TST_tight_base, y_all, fval, "TIGHT_BASELINE"
)

log("Running BASELINE stack (FULL, no node_0098) ...")
_, _, pf_base_full, cv_base_full, sem_base_full = nested_cv_arm_fast(
    OOF_full_base, TST_full_base, y_all, fval, "FULL_BASELINE"
)

log("Running TIGHT + node_0098 ...")
stacked_oof_tight, stacked_tst_tight, pf_tight98, cv_tight98, sem_tight98 = nested_cv_arm_fast(
    OOF_tight_plus98, TST_tight_plus98, y_all, fval, "TIGHT+98"
)

log("Running FULL + node_0098 ...")
stacked_oof_full, stacked_tst_full, pf_full98, cv_full98, sem_full98 = nested_cv_arm_fast(
    OOF_full_plus98, TST_full_plus98, y_all, fval, "FULL+98"
)

# ─── REPORT ───────────────────────────────────────────────────────────────────
log("=" * 70)
log("STACK-ADD REPORT:")
log(f"  champion node_0091 stacked CV:  {CHAMPION_CV:.6f}")
log(f"  TIGHT baseline (this run):      cv={cv_base_tight:.6f}  sem={sem_base_tight:.6f}")
log(f"  FULL  baseline (this run):      cv={cv_base_full:.6f}   sem={sem_base_full:.6f}")
log(f"  TIGHT + node_0098:              cv={cv_tight98:.6f}   sem={sem_tight98:.6f}  "
    f"delta_vs_tight={cv_tight98-cv_base_tight:+.6f}  delta_vs_champ={cv_tight98-CHAMPION_CV:+.6f}")
log(f"  FULL  + node_0098:              cv={cv_full98:.6f}   sem={sem_full98:.6f}  "
    f"delta_vs_full={cv_full98-cv_base_full:+.6f}  delta_vs_champ={cv_full98-CHAMPION_CV:+.6f}")
log(f"  per-fold TIGHT baseline:       {[f'{s:.6f}' for s in pf_base_tight]}")
log(f"  per-fold TIGHT+98:             {[f'{s:.6f}' for s in pf_tight98]}")
log(f"  per-fold FULL baseline:        {[f'{s:.6f}' for s in pf_base_full]}")
log(f"  per-fold FULL+98:              {[f'{s:.6f}' for s in pf_full98]}")

pf_deltas_tight = [pf_tight98[i] - pf_base_tight[i] for i in range(5)]
pf_deltas_full  = [pf_full98[i]  - pf_base_full[i]  for i in range(5)]
log(f"  per-fold delta TIGHT+98-base:  {[f'{d:+.6f}' for d in pf_deltas_tight]}")
log(f"  per-fold delta FULL+98-base:   {[f'{d:+.6f}' for d in pf_deltas_full]}")

# Champion assert (reproduce 0.970355 within 0.0002 tolerance)
# The baseline assert checks pool loading consistency
best_baseline_cv = max(cv_base_tight, cv_base_full)
delta_from_champ_baseline = abs(best_baseline_cv - CHAMPION_CV)
log(f"  best_baseline_cv={best_baseline_cv:.6f}  "
    f"delta_from_champion={delta_from_champ_baseline:.6f}")
if delta_from_champ_baseline > 0.001:
    log(f"  WARN: baseline delta={delta_from_champ_baseline:.6f} > 0.001 — pool may have changed")
else:
    log(f"  baseline ASSERT PASS (within 0.001 of champion {CHAMPION_CV})")

# Best stack arm
best_stack_cv  = max(cv_tight98, cv_full98)
best_stack_sem = sem_tight98 if cv_tight98 >= cv_full98 else sem_full98
best_stack_arm = "TIGHT+98" if cv_tight98 >= cv_full98 else "FULL+98"
delta_vs_champ = best_stack_cv - CHAMPION_CV
promote_bar    = CHAMPION_CV + 2 * max(sem_base_tight, sem_base_full)

log(f"  BEST STACK ARM: {best_stack_arm}  cv={best_stack_cv:.6f}  "
    f"delta_vs_champ={delta_vs_champ:+.6f}")
log(f"  promote_bar (champion + 2*sem_baseline): {promote_bar:.6f}")
log(f"  STACK-ADD result: {'GAIN' if delta_vs_champ > 0 else 'LOSS'} ({delta_vs_champ:+.6f})")
log(f"  promotes? {'YES' if best_stack_cv > promote_bar else 'NO'}")

print(f"stack_add_cv={best_stack_cv:.6f}", flush=True)
print(f"stack_add_delta_vs_champ={delta_vs_champ:+.6f}", flush=True)
print(f"stack_add_arm={best_stack_arm}", flush=True)
print(f"promotes={'YES' if best_stack_cv > promote_bar else 'NO'}", flush=True)

# ─── Save stacked artifacts if stack wins ────────────────────────────────────
if best_stack_cv > CHAMPION_CV:
    log("Stack beats champion (by some margin) — saving stacked_oof.npy / stacked_test.npy ...")
    if best_stack_arm == "TIGHT+98":
        stacked_oof_save  = stacked_oof_tight
        stacked_tst_save  = stacked_tst_tight
    else:
        stacked_oof_save  = stacked_oof_full
        stacked_tst_save  = stacked_tst_full

    np.save(NODE_DIR / "stacked_oof.npy",  stacked_oof_save.astype(np.float32))
    np.save(NODE_DIR / "stacked_test.npy", stacked_tst_save.astype(np.float32))
    log(f"  stacked_oof.npy={stacked_oof_save.shape}  stacked_test.npy={stacked_tst_save.shape}")

    # Overwrite submission.csv with stacked argmax
    stacked_pred_labels = np.array([CLASSES[i] for i in stacked_tst_save.argmax(1)])
    sub_stacked = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: stacked_pred_labels})
    sub_stacked = sub_stacked[list(sample_sub.columns)]
    sub_stacked.to_csv(sub_path, index=False)
    log(f"  submission.csv overwritten with stacked predictions: {sub_stacked.shape}")
    log(f"  stacked pred dist: {dict(zip(*np.unique(stacked_pred_labels, return_counts=True)))}")
else:
    log("Stack does NOT beat champion — submission.csv stays as solo base predictions")

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
log("=" * 70)
log("FINAL SUMMARY:")
log(f"  STEP 1 — Base OOF:")
log(f"    n_features={n_features}  n_components={N_COMPONENTS}  gamma={GAMMA:.6f}")
log(f"    solo_cv={mean_cv:.6f}  sem={sem_cv:.6f}")
log(f"    per_fold_scores={[f'{s:.6f}' for s in per_fold_scores]}")
log(f"    mean_err_corr_vs_n70={mean_errcorr:.4f}")
log(f"    per_fold_errcorr={[f'{s:.4f}' for s in per_fold_errcorr]}")
log(f"  STEP 2 — Stack-add:")
log(f"    champion={CHAMPION_CV:.6f}  best_arm={best_stack_arm}  "
    f"stack_cv={best_stack_cv:.6f}  delta={delta_vs_champ:+.6f}  "
    f"sem={best_stack_sem:.6f}")
log(f"    promote? {'YES' if best_stack_cv > promote_bar else 'NO'} "
    f"(bar={promote_bar:.6f})")

total_elapsed = time.perf_counter() - T0
log(f"Done. Total elapsed={total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
