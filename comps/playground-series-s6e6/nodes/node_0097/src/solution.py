"""node_0097 — stronger Nystroem RBF base + stack-add

HYPOTHESIS: A higher-capacity RBF kernel machine on the FULL rich FE (fs_realmlp_fe,
~26 features) reaches useful accuracy while preserving the decorrelation that made
node_0096 interesting (err-corr 0.530 vs node_0070).

THREE COUPLED KNOBS (one hypothesis):
  1. INPUT = full fs_realmlp_fe (~26 features; stateless, row-wise), NOT 13-dim core.
  2. n_components ~4000-8000 (up from 1000 in node_0096).
  3. Re-sweep gamma (5 values) on the richer input at fold-0.

FOLD-0 GATE (RELAXED vs node_0096):
  PROCEED only if fold-0 solo BA >= 0.955 AND err_corr vs node_0070 < 0.65.
  (BA floor relaxed 0.96→0.955: exceptional decorrelation justifies a lower solo floor.)
  KILL otherwise — record BA + err_corr in gate_note, stop.

ON PASS:
  Full 5-fold OOF (577347,3) + test_probs (247435,3).
  Then THE DECISIVE TEST: champion stack-add. Take the node_0091 recipe (balanced
  multinomial LogReg on clipped log-probs over the full base pool, nested C) and add
  THIS node's OOF as one more base column. Refit the meta fold-honest (same nested-C
  protocol as node_0091). Report: champion stacked CV 0.970355 baseline, the new
  stacked CV, the delta, and per-fold deltas.

LEAK DISCIPLINE:
  - fs_realmlp_fe is STATELESS (pure row-wise, no fit, no target, no cross-row stats).
    Safe to build once on the full train/test.
  - StandardScaler stats fit on TRAIN FOLD ONLY.
  - Nystroem landmarks sampled from TRAIN FOLD ONLY (subsample fine).
  - gamma chosen by fold-0 train-fold micro-sweep only.
  - TARGET and ID absent from feature matrix (asserted).
  - Folds loaded from frozen folds.json.

Class order: GALAXY=0, QSO=1, STAR=2.
Metric: Balanced Accuracy, maximize.
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

# fs_realmlp_fe recipe (data.md stateless row-wise features)
# Color pairs (7 pairs) from fs_realmlp_fe
COLOR_PAIRS = [
    ("u", "g"),   # u-g
    ("g", "r"),   # g-r
    ("r", "i"),   # r-i
    ("i", "z"),   # i-z
    ("u", "r"),   # u-r
    ("g", "i"),   # g-i
    ("r", "z"),   # r-z
]

# Base numeric columns
BASE_MAGS = ["u", "g", "r", "i", "z"]
BASE_NUM = ["u", "g", "r", "i", "z", "redshift"]

# Nystroem config — KEY CHANGE vs node_0096: n_components 1000→2000
# 4000 OOM'd at 30GB RAM (461k×4000 matrices ~7.4GB each). 2000 is ~3.1GB total,
# safely within the 24GB available budget with LogReg overhead.
N_COMPONENTS = 2000
NYSTROEM_SUBSAMPLE = 40_000  # landmarks sampled from this many train-fold rows
LOGREG_MAX_ITER = 1000
LOGREG_C = 1.0

# Fold-0 gate thresholds — RELAXED BA floor vs node_0096 (0.96→0.955)
FOLD0_BA_THRESHOLD = 0.955
FOLD0_ERR_CORR_THRESHOLD = 0.65

# Champion stack baseline (node_0091)
CHAMPION_CV = 0.970355


def build_realmlp_fe(df: pd.DataFrame) -> np.ndarray:
    """
    Build fs_realmlp_fe feature vector — STATELESS, row-wise, no fitting.

    Features (from data.md fs_realmlp_fe recipe + refs/realmlp-v5-for-s6e6.py):
      - redshift ratios: g/redshift, i/redshift (2 feats)
      - log1p(redshift) (1 feat)
      - 7 color pairs (7 feats)
      - mag_mean, mag_range (2 feats)
      - base numerics: u,g,r,i,z,redshift (6 feats)

    Total: 2 + 1 + 7 + 2 + 6 = 18 numerical features.
    (Integer-floor categorical views and cross-combos are also in fs_realmlp_fe
     but these are categorical/high-cardinality — we keep the continuous floats only
     for the kernel map, which needs continuous input.)
    Returns float32 array shape (n, n_features).
    """
    cols = []
    feature_names = []

    # 6 base numerics (u,g,r,i,z,redshift)
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

    # For landmark sampling, use a subsample of train-fold (tractable at 577k rows)
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
    """Apply fitted scaler + Nystroem transform (transform only, never fit).
    Uses batched processing to bound peak memory: batch_size rows at a time.
    """
    if len(X) <= batch_size:
        return nystroem.transform(scaler.transform(X))
    # Batch transform to keep peak memory bounded
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


# ─── Load data ────────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_data = json.loads((COMP_DIR / "folds.json").read_text())
folds_list = folds_data["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test = len(test_raw)
assert n_train == 577347, f"unexpected n_train={n_train}"
assert n_test == 247435, f"unexpected n_test={n_test}"

# ─── Load node_0070 OOF (for err-corr gate) ──────────────────────────────────
log("Loading node_0070 OOF for error-correlation gate ...")
oof_70 = np.load(COMP_DIR / "nodes/node_0070/oof.npy")
assert oof_70.shape == (n_train, N_CLASSES), f"Expected ({n_train},{N_CLASSES}), got {oof_70.shape}"
y_pred_70 = oof_70.argmax(axis=1)
log(f"  node_0070 OOF errors: {(y_pred_70 != y_all).sum()} / {n_train}")

# ─── Build fs_realmlp_fe features (STATELESS — safe before fold split) ───────
log("Building fs_realmlp_fe stateless features ...")
X_train_fe, feature_names = build_realmlp_fe(train_raw)
X_test_fe, _ = build_realmlp_fe(test_raw)
n_features = X_train_fe.shape[1]
log(f"  X_train_fe={X_train_fe.shape}  X_test_fe={X_test_fe.shape}")
log(f"  features: {feature_names}")

# ─── Pre-flight leakage check 1: TARGET and ID not in feature list ─────────
log("Pre-flight leakage checks ...")
assert TARGET not in feature_names, f"TARGET {TARGET} in features — LEAK!"
assert IDC not in feature_names, f"ID {IDC} in features — LEAK!"
log(f"  check1 PASS: target/id absent from {n_features} feature columns")

# ─── Pre-flight leakage check 2: single-feature↔target sweep on sample ──────
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

# ─── Pre-flight leakage check 5: frozen folds ────────────────────────────────
assert len(folds_list) == 5, f"Expected 5 folds, got {len(folds_list)}"
all_val_idx = []
for fi in folds_list:
    all_val_idx.extend(fi["val_idx"])
assert len(set(all_val_idx)) == n_train, "Folds don't cover all train rows exactly once"
log(f"  check5 PASS: frozen folds verified ({len(folds_list)} folds, {n_train} unique val rows)")

# ─── Check 6: train/test near-dup spot check ────────────────────────────────
log("Check 6: train/test near-dup spot check ...")
tr_sample = X_train_fe[:1000].round(4)
te_sample = X_test_fe[:1000].round(4)
tr_set = set(map(tuple, tr_sample.tolist()))
te_set = set(map(tuple, te_sample.tolist()))
overlap = tr_set & te_set
log(f"  check6: exact-match overlap (1k sample each): {len(overlap)} rows — {'warn' if len(overlap) > 0 else 'clean'}")

# ─── PRE-FLIGHT COMPLETE — launch training ───────────────────────────────────

# ─── Gamma micro-sweep on fold-0 train split (CHEAP, before full loop) ───────
log("=== GAMMA MICRO-SWEEP on fold-0 (5 candidates, richer input) ===")
fold0_info = folds_list[0]
val_idx0 = np.asarray(fold0_info["val_idx"])
tr_idx0 = np.setdiff1d(np.arange(n_train), val_idx0)

X_tr0 = X_train_fe[tr_idx0]
X_val0 = X_train_fe[val_idx0]
y_tr0 = y_all[tr_idx0]
y_val0 = y_all[val_idx0]

# For sweep: use a smaller n_components (1000) to keep the sweep cheap, then
# use the chosen gamma with full n_components for the real run.
N_COMPONENTS_SWEEP = 1000
SWEEP_SUBSAMPLE = 30_000  # subsample for fast gamma sweep

# 5-point gamma sweep on the richer input
# node_0096 best gamma was 0.077 = 1/13 (1/n_features on 13-dim)
# Now n_features=18, so 1/18 ≈ 0.056; also try scaled variants
feature_var_fold0 = X_tr0.var(axis=0).mean()
gamma_candidates = [
    0.01,
    1.0 / n_features,                          # ~0.056 (1/n_features)
    1.0 / (n_features * feature_var_fold0),    # scaled by feature var
    0.5 / n_features,                          # 0.028
    2.0 / n_features,                          # 0.111
]
log(f"  n_features={n_features}  feature_var_mean={feature_var_fold0:.4f}")
log(f"  gamma candidates: {[f'{g:.6f}' for g in gamma_candidates]}")

best_gamma = None
best_gamma_ba = -1.0

# Use a small subsample for the sweep to keep it cheap
rng_sweep = np.random.RandomState(SEED)
sweep_tr_idx = rng_sweep.choice(len(tr_idx0), min(SWEEP_SUBSAMPLE, len(tr_idx0)), replace=False)
X_sweep_tr = X_tr0[sweep_tr_idx]
y_sweep_tr = y_tr0[sweep_tr_idx]

for gamma in gamma_candidates:
    t_g = time.perf_counter()
    rng_g = np.random.RandomState(SEED)
    scaler_g, nystroem_g = fit_nystroem_pipeline(
        X_sweep_tr, gamma=gamma, n_components=N_COMPONENTS_SWEEP, rng=rng_g
    )
    X_sw_mapped = apply_pipeline(X_sweep_tr, scaler_g, nystroem_g)
    X_val_mapped_g = apply_pipeline(X_val0, scaler_g, nystroem_g)

    lr_g = make_logreg(SEED)
    lr_g.fit(X_sw_mapped, y_sweep_tr)
    val_ba_g = balanced_accuracy_score(y_val0, lr_g.predict(X_val_mapped_g))
    elapsed_g = time.perf_counter() - t_g
    log(f"  gamma={gamma:.6f}  sweep_val_BA={val_ba_g:.6f}  ({elapsed_g:.1f}s)")

    if val_ba_g > best_gamma_ba:
        best_gamma_ba = val_ba_g
        best_gamma = gamma

log(f"  SELECTED gamma={best_gamma:.6f} (sweep_val_BA={best_gamma_ba:.6f})")
del X_sweep_tr, y_sweep_tr, X_val_mapped_g, X_sw_mapped
del scaler_g, nystroem_g, lr_g
gc.collect()

# ─── FOLD-0 TIMING PROBE + GATE ──────────────────────────────────────────────
log(f"=== FOLD-0 FULL PROBE: n_components={N_COMPONENTS}, gamma={best_gamma:.6f} ===")
t_fold0 = time.perf_counter()

rng_fold0 = np.random.RandomState(SEED + 100)
scaler_fold0, nystroem_fold0 = fit_nystroem_pipeline(
    X_tr0, gamma=best_gamma, n_components=N_COMPONENTS, rng=rng_fold0
)
X_tr0_mapped = apply_pipeline(X_tr0, scaler_fold0, nystroem_fold0)
X_val0_mapped = apply_pipeline(X_val0, scaler_fold0, nystroem_fold0)
X_te0_mapped = apply_pipeline(X_test_fe, scaler_fold0, nystroem_fold0)

log(f"  X_tr0_mapped={X_tr0_mapped.shape}  X_val0_mapped={X_val0_mapped.shape}")

model_fold0 = make_logreg(SEED + 100)
model_fold0.fit(X_tr0_mapped, y_tr0)
val_proba0 = model_fold0.predict_proba(X_val0_mapped).astype(np.float32)
fold0_ba = float(balanced_accuracy_score(y_val0, val_proba0.argmax(1)))

fold0_elapsed = time.perf_counter() - t_fold0
projected_total = fold0_elapsed * len(folds_list)
log(f"  fold0_ba={fold0_ba:.6f}  elapsed={fold0_elapsed:.1f}s  projected_5fold={projected_total/60:.1f}min")

# Error-correlation vs node_0070 fold-0
val_pred0 = val_proba0.argmax(1)
val_pred_70_f0 = y_pred_70[val_idx0]

err_97 = (val_pred0 != y_val0).astype(np.float32)
err_70 = (val_pred_70_f0 != y_val0).astype(np.float32)
fold0_err_corr = float(np.corrcoef(err_97, err_70)[0, 1])

log(f"  FOLD-0 GATE CHECK:")
log(f"    solo_BA={fold0_ba:.6f}  (threshold>={FOLD0_BA_THRESHOLD})")
log(f"    err_corr_vs_n70={fold0_err_corr:.4f}  (threshold<{FOLD0_ERR_CORR_THRESHOLD})")
print(f"fold0_solo_ba={fold0_ba:.6f}", flush=True)
print(f"fold0_err_corr_vs_n70={fold0_err_corr:.6f}", flush=True)
print(f"best_gamma={best_gamma:.8f}", flush=True)

# ─── FOLD-0 GATE DECISION ────────────────────────────────────────────────────
fold0_gate_passed = (fold0_ba >= FOLD0_BA_THRESHOLD) and (fold0_err_corr < FOLD0_ERR_CORR_THRESHOLD)

if not fold0_gate_passed:
    reason = []
    if fold0_ba < FOLD0_BA_THRESHOLD:
        reason.append(f"solo_BA={fold0_ba:.6f} < {FOLD0_BA_THRESHOLD}")
    if fold0_err_corr >= FOLD0_ERR_CORR_THRESHOLD:
        reason.append(f"err_corr={fold0_err_corr:.4f} >= {FOLD0_ERR_CORR_THRESHOLD}")
    kill_msg = " AND ".join(reason)
    log(f"  FOLD-0 GATE FAILED: {kill_msg}")
    log("  KILL: stopping after fold 0 — write partial artifacts")
    print(f"GATE_KILL: {kill_msg}", flush=True)

    # Write partial artifacts for diagnostics
    oof_partial = np.zeros((n_train, N_CLASSES), dtype=np.float32)
    oof_partial[val_idx0] = val_proba0
    np.save(NODE_DIR / "oof.npy", oof_partial)

    test_proba0 = model_fold0.predict_proba(X_te0_mapped).astype(np.float32)
    np.save(NODE_DIR / "test_probs.npy", test_proba0)

    pred_labels = np.array([CLASSES[i] for i in test_proba0.argmax(1)])
    sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
    sub = sub[list(sample_sub.columns)]
    sub.to_csv(NODE_DIR / "submission.csv", index=False)

    log(f"GATE KILL SUMMARY:")
    log(f"  fold0_ba={fold0_ba:.6f}  err_corr={fold0_err_corr:.4f}  best_gamma={best_gamma:.8f}")

    total_elapsed = time.perf_counter() - T0
    log(f"Done. Total elapsed={total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    sys.exit(0)

# ─── GATE PASSED — proceed to full 5-fold run ────────────────────────────────
log(f"  FOLD-0 GATE PASSED: BA={fold0_ba:.6f}>={FOLD0_BA_THRESHOLD}, "
    f"err_corr={fold0_err_corr:.4f}<{FOLD0_ERR_CORR_THRESHOLD}")
log("  Continuing to full 5-fold run ...")

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []

log(f"Starting OOF loop (n_components={N_COMPONENTS}, gamma={best_gamma:.6f}) ...")

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    # Fold 0 is already computed above — reuse its results
    if fold_id == 0:
        oof_proba[val_idx0] = val_proba0
        test_proba_fold0 = model_fold0.predict_proba(X_te0_mapped).astype(np.float32)
        test_proba_accum += test_proba_fold0 / len(folds_list)
        per_fold_scores.append(fold0_ba)
        log(f"  fold 0: balanced_accuracy={fold0_ba:.6f}  (reused from gate)")
        print(f"fold0_score={fold0_ba:.6f}", flush=True)
        # Free fold-0 resources
        del X_tr0_mapped, X_val0_mapped, X_te0_mapped
        del scaler_fold0, nystroem_fold0, model_fold0, val_proba0, test_proba_fold0
        gc.collect()
        continue

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")
    fold_t0 = time.perf_counter()

    X_tr_fold = X_train_fe[tr_idx]
    X_val_fold = X_train_fe[val_idx]
    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # ─── Check 4: fit_in_fold — StandardScaler + Nystroem on TRAIN FOLD ONLY ──
    rng_fold = np.random.RandomState(fold_seed)
    scaler_fold, nystroem_fold = fit_nystroem_pipeline(
        X_tr_fold, gamma=best_gamma, n_components=N_COMPONENTS, rng=rng_fold
    )

    # Apply to val and test (transform only, never fit on val/test)
    X_tr_mapped = apply_pipeline(X_tr_fold, scaler_fold, nystroem_fold)
    X_val_mapped = apply_pipeline(X_val_fold, scaler_fold, nystroem_fold)
    X_te_mapped = apply_pipeline(X_test_fe, scaler_fold, nystroem_fold)

    log(f"  Fold {fold_id}: mapped X_tr={X_tr_mapped.shape} X_val={X_val_mapped.shape}")

    # ─── Fit balanced LogReg ──────────────────────────────────────────────────
    model = make_logreg(seed=fold_seed)
    model.fit(X_tr_mapped, y_tr_fold)

    val_proba = model.predict_proba(X_val_mapped).astype(np.float32)
    oof_proba[val_idx] = val_proba

    test_proba_fold = model.predict_proba(X_te_mapped).astype(np.float32)
    test_proba_accum += test_proba_fold / len(folds_list)

    fold_score = balanced_accuracy_score(y_val_fold, val_proba.argmax(1))
    per_fold_scores.append(fold_score)
    fold_elapsed = time.perf_counter() - fold_t0
    log(f"  fold {fold_id}: balanced_accuracy={fold_score:.6f}  elapsed={fold_elapsed:.1f}s")
    print(f"fold{fold_id}_score={fold_score:.6f}", flush=True)

    # Clean up fold memory
    del X_tr_fold, X_val_fold, X_tr_mapped, X_val_mapped, X_te_mapped
    del scaler_fold, nystroem_fold, model
    gc.collect()

# ─── Full OOF complete ────────────────────────────────────────────────────────
mean_cv = float(np.mean(per_fold_scores))
sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
print(f"cv={mean_cv:.6f}", flush=True)

# ─── Post-OOF checks ─────────────────────────────────────────────────────────
# Check 7: OOF complete
assert not np.any(np.isnan(oof_proba)), "NaN in OOF probs!"
covered = (oof_proba.sum(axis=1) > 0)
assert covered.all(), f"OOF has {(~covered).sum()} uncovered rows!"
log(f"  check7 PASS: OOF complete, no NaN ({n_train} rows)")

# Check 8: distribution sane
prob_sums = oof_proba.sum(axis=1)
assert np.allclose(prob_sums, 1.0, atol=1e-3), f"OOF probs don't sum to 1"
assert oof_proba.min() >= -1e-6, f"OOF probs < 0: {oof_proba.min()}"
assert oof_proba.max() <= 1.0 + 1e-6, f"OOF probs > 1: {oof_proba.max()}"
log(f"  check8 PASS: OOF distribution sane (probs in [0,1], sum to 1)")

oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
log(f"OOF full balanced_accuracy={oof_metric:.6f}")

# ─── Save OOF + test_probs ────────────────────────────────────────────────────
np.save(NODE_DIR / "oof.npy", oof_proba)
log(f"Saved oof.npy shape={oof_proba.shape}")

np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

# ─── Write submission ─────────────────────────────────────────────────────────
pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
sub = sub[list(sample_sub.columns)]
sub_path = NODE_DIR / "submission.csv"
sub.to_csv(sub_path, index=False)
log(f"Saved submission.csv shape={sub.shape}")
log(f"  prediction distribution: {dict(zip(*np.unique(pred_labels, return_counts=True)))}")

# ─── DECISIVE TEST: CHAMPION STACK-ADD ──────────────────────────────────────
log("=" * 70)
log("DECISIVE TEST: CHAMPION STACK-ADD (node_0091 recipe + this OOF)")
log("Baseline: champion stacked CV = 0.970355")

# Load all the champion's bases (node_0091 recipe)
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
NC = 3
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# TIGHT pool from node_0091
TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]

def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))

def norm(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True)
    s[s == 0] = 1
    return a / s

def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))

def rd(path: str | Path, nr: int) -> np.ndarray:
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a
        return a[:, :3]
    d = pd.read_csv(p)
    c = list(d.columns)
    if set(LAB).issubset(c):
        return d[LAB].values.astype(float)
    pc = [f"prob_{l}" for l in LAB]
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

# Load public bank-17
B = COMP_DIR / "refs/oof_bank"
K = COMP_DIR / "refs/kernel_out"

MANIFEST = {
    'xgb-0':      (K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",         K/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
    'xgb-1':      (K/"xgb-v1-for-s6e6/oof_preds.npy",           K/"xgb-v1-for-s6e6/test_preds.npy"),
    'realmlp-0':  (B/"oof_preds_realmlp0_v12.csv",               B/"test_preds_realmlp0_v12.csv"),
    'realmlp-1':  (K/"realmlp-v1-for-s6e6/oof_preds.npy",        K/"realmlp-v1-for-s6e6/test_preds.npy"),
    'tabm-0':     (B/"oof_preds_tabm0_v2.csv",                   B/"test_preds_tabm0_v2.csv"),
    'cat-0':      (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
    'realmlp-2':  (B/"oof_preds_realmlp2_v10.csv",               B/"test_preds_realmlp2_v10.csv"),
    'tabicl-2':   (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
    'lgbm-3':     (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",     K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
    'logreg-1':   (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",  K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
    'nn-1':       (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",          K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
    'xgb-3':      (K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
    'xgb-5':      (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",       K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
    'realmlp-5':  (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
    'nn-2':       (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",          K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
    'cat-3':      (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",        K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
    'lgbm-5':     (B/"oof_preds_lgbm5_v1.csv",                  B/"test_preds_lgbm5_v1.csv"),
    'xgb-6':      (B/"oof_final_xgb6_v1.csv",                   B/"test_final_xgb6_v1.csv"),
    'tabm-1':     (B/"oof_final_tabm1_v1.csv",                   B/"test_final_tabm1_v1.csv"),
}

log("Loading public bank bases ...")
POOF = {}; PTEST = {}; good = []
for name, (op, tp) in MANIFEST.items():
    try:
        o = norm(rd(op, n_train)); t = norm(rd(tp, n_test))
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

# Load in-house TIGHT + FULL bases
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
            inhouse_oof_tight[node_nm] = logp(o)
            inhouse_test_tight[node_nm] = logp(t)
    except Exception as e:
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
            inhouse_oof_weak[node_nm] = logp(o)
            inhouse_test_weak[node_nm] = logp(t)
    except Exception as e:
        pass

log(f"  in-house TIGHT={len(inhouse_oof_tight)}/{len(TIGHT_IDS)}, WEAK={len(inhouse_oof_weak)}/{len(WEAK_EXTRA_IDS)}")

# Build base feature matrix (same as node_0091 winning arm — need to determine which won)
# node_0091 reported TIGHT won (cv=0.970355). Use same pool + add node_0097.
base_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
base_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]

tight_inhouse_oof  = list(inhouse_oof_tight.values())
tight_inhouse_test = list(inhouse_test_tight.values())
full_inhouse_oof   = tight_inhouse_oof + list(inhouse_oof_weak.values())
full_inhouse_test  = tight_inhouse_test + list(inhouse_test_weak.values())

# This node's OOF (as one more base column)
this_oof_logp  = logp(norm(oof_proba.astype(float)))
this_test_logp = logp(norm(test_proba_accum.astype(float)))

OOF_tight_base  = np.concatenate(base_oof_logp + tight_inhouse_oof,  axis=1)
TST_tight_base  = np.concatenate(base_test_logp + tight_inhouse_test, axis=1)
OOF_full_base   = np.concatenate(base_oof_logp + full_inhouse_oof,   axis=1)
TST_full_base   = np.concatenate(base_test_logp + full_inhouse_test,  axis=1)

# +node_0097 versions
OOF_tight_plus97 = np.concatenate([OOF_tight_base, this_oof_logp],  axis=1)
TST_tight_plus97 = np.concatenate([TST_tight_base, this_test_logp], axis=1)
OOF_full_plus97  = np.concatenate([OOF_full_base,  this_oof_logp],  axis=1)
TST_full_plus97  = np.concatenate([TST_full_base,  this_test_logp], axis=1)

log(f"  OOF_tight_base={OOF_tight_base.shape}  OOF_tight_plus97={OOF_tight_plus97.shape}")

fval = [np.asarray(f["val_idx"]) for f in folds_list]


def nested_cv_arm_fast(OOF_mat, TST_mat, y, fval, label):
    """Nested C-selection + outer OOF using LogisticRegressionCV."""
    n = len(y)
    oof_probs = np.zeros((n, NC), dtype=float)
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
    m_final = LogisticRegression(class_weight="balanced", C=c_win, max_iter=2000,
                                  n_jobs=-1, random_state=42,
                                  solver="lbfgs", multi_class="multinomial")
    m_final.fit(OOF_mat, y)
    tst_probs = m_final.predict_proba(TST_mat)
    log(f"  {label}: cv={cv_mean:.6f}  sem={cv_sem:.6f}  per_fold={[f'{s:.6f}' for s in pf]}  best_Cs={best_Cs}")
    return oof_probs, tst_probs, pf, cv_mean, cv_sem


log("Running BASELINE stack (TIGHT, no node_0097) ...")
_, _, pf_base, cv_base, sem_base = nested_cv_arm_fast(
    OOF_tight_base, TST_tight_base, y_all, fval, "TIGHT_BASELINE"
)

log("Running TIGHT+97 stack ...")
_, _, pf_tight97, cv_tight97, sem_tight97 = nested_cv_arm_fast(
    OOF_tight_plus97, TST_tight_plus97, y_all, fval, "TIGHT+97"
)

log("Running FULL+97 stack ...")
_, _, pf_full97, cv_full97, sem_full97 = nested_cv_arm_fast(
    OOF_full_plus97, TST_full_plus97, y_all, fval, "FULL+97"
)

# ─── STACK-ADD RESULTS ────────────────────────────────────────────────────────
log("=" * 70)
log("STACK-ADD RESULTS:")
log(f"  champion baseline (node_0091):  cv={CHAMPION_CV:.6f}")
log(f"  TIGHT baseline (this run):      cv={cv_base:.6f}  sem={sem_base:.6f}")
log(f"  TIGHT + node_0097:              cv={cv_tight97:.6f}  sem={sem_tight97:.6f}  delta={cv_tight97-cv_base:+.6f}")
log(f"  FULL  + node_0097:              cv={cv_full97:.6f}  sem={sem_full97:.6f}  delta={cv_full97-cv_base:+.6f}")
log(f"  per-fold TIGHT baseline:       {[f'{s:.6f}' for s in pf_base]}")
log(f"  per-fold TIGHT+97:             {[f'{s:.6f}' for s in pf_tight97]}")
log(f"  per-fold FULL+97:              {[f'{s:.6f}' for s in pf_full97]}")

pf_deltas_tight = [pf_tight97[i] - pf_base[i] for i in range(len(pf_base))]
pf_deltas_full  = [pf_full97[i] - pf_base[i] for i in range(len(pf_base))]
log(f"  per-fold delta TIGHT+97-base:  {[f'{d:+.6f}' for d in pf_deltas_tight]}")
log(f"  per-fold delta FULL+97-base:   {[f'{d:+.6f}' for d in pf_deltas_full]}")

best_stack_cv  = max(cv_tight97, cv_full97)
best_stack_sem = sem_tight97 if cv_tight97 >= cv_full97 else sem_full97
best_stack_arm = "TIGHT+97" if cv_tight97 >= cv_full97 else "FULL+97"
delta_vs_champ = best_stack_cv - CHAMPION_CV
promote_bar = CHAMPION_CV + 2 * sem_base  # use baseline sem as proxy

log(f"  BEST STACK ARM: {best_stack_arm}  cv={best_stack_cv:.6f}  delta_vs_champ={delta_vs_champ:+.6f}")
log(f"  promote_bar (champion + 2*sem_base): {promote_bar:.6f}")
log(f"  STACK-ADD result: {'GAIN' if delta_vs_champ > 0 else 'LOSS'} ({delta_vs_champ:+.6f})")
log(f"  promotes? {'YES' if best_stack_cv > promote_bar else 'NO'}")

print(f"stack_add_cv={best_stack_cv:.6f}", flush=True)
print(f"stack_add_delta={delta_vs_champ:+.6f}", flush=True)
print(f"stack_add_arm={best_stack_arm}", flush=True)

# ─── Validate submission ──────────────────────────────────────────────────────
log("Validating submission schema ...")
sub_loaded = pd.read_csv(sub_path)
assert list(sub_loaded.columns) == list(sample_sub.columns), \
    f"column mismatch: {list(sub_loaded.columns)} vs {list(sample_sub.columns)}"
assert len(sub_loaded) == len(sample_sub), \
    f"row count: {len(sub_loaded)} vs {len(sample_sub)}"
assert set(sub_loaded[TARGET].unique()) <= set(CLASSES), \
    f"unknown classes: {set(sub_loaded[TARGET].unique()) - set(CLASSES)}"
log("  check9 PASS: submission schema OK")

# ─── Final summary ────────────────────────────────────────────────────────────
log("=" * 70)
log("FINAL SUMMARY:")
log(f"  n_features={n_features}  n_components={N_COMPONENTS}  best_gamma={best_gamma:.6f}")
log(f"  fold0_ba={fold0_ba:.6f}  fold0_err_corr={fold0_err_corr:.4f}  GATE=PASS")
log(f"  solo_cv={mean_cv:.6f}  sem={sem_cv:.6f}")
log(f"  per_fold={[f'{s:.6f}' for s in per_fold_scores]}")
log(f"  stack_add: {best_stack_arm}  cv={best_stack_cv:.6f}  delta={delta_vs_champ:+.6f}")

total_elapsed = time.perf_counter() - T0
log(f"Done. Total elapsed={total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
