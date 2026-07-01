"""node_0096 — draft: Nystroem RBF random-feature linear base.

NEW MODEL CLASS: sklearn.kernel_approximation.Nystroem(kernel='rbf') →
balanced multinomial LogisticRegression. Input = ~13-dim standardized core
photometric vector (u,g,r,i,z,redshift + 7 color pairs from fs_realmlp_fe).

PIPELINE (all fit_in_fold):
  1. Build 13-dim core vector (stateless color pairs).
  2. StandardScaler fit on train-fold rows only.
  3. Nystroem RBF map (landmarks sampled from train-fold subsample) fit on
     train-fold rows only.
  4. LogisticRegression(class_weight='balanced', multi_class='multinomial')
     fit on train-fold.

DECISIVE FOLD-0 KILL GATE (copied from node_0094 pattern):
  - Solo fold-0 Balanced Accuracy >= 0.96 AND
  - Error-correlation vs node_0070 fold-0 errors < 0.65
  If either fails: write artifacts, set gate_note, STOP.

GAMMA MICRO-SWEEP (fold-0 only, on a 20k subsample):
  Try 3 gamma values around 1/n_features and 1/(n_features*std),
  pick best by fold-0 val BA, freeze for full run.

LEAK DISCIPLINE (fs_rbf_nystroem is fit_in_fold):
  - StandardScaler stats fit on TRAIN FOLD ONLY.
  - Nystroem landmarks sampled from TRAIN FOLD ONLY (subsample up to 30k).
  - gamma chosen by fold-0 train-fold micro-sweep only.
  - TARGET and ID columns absent from feature matrix (asserted).
  - Folds loaded from frozen folds.json.

Class order: GALAXY=0, QSO=1, STAR=2.
Metric: Balanced Accuracy, maximize.
Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv
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
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
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

# Color pairs from fs_realmlp_fe recipe (data.md)
COLOR_PAIRS = [
    ("u", "g"),   # u-g
    ("g", "r"),   # g-r
    ("r", "i"),   # r-i
    ("i", "z"),   # i-z
    ("u", "r"),   # u-r
    ("g", "i"),   # g-i
    ("r", "z"),   # r-z
]

# Core photometric columns (5 mags + redshift)
CORE_COLS = ["u", "g", "r", "i", "z", "redshift"]

# Nystroem config
N_COMPONENTS = 1000          # start here; enough for kernel-approx quality
NYSTROEM_SUBSAMPLE = 30_000  # landmarks sampled from this many train-fold rows
LOGREG_MAX_ITER = 1000
LOGREG_C = 1.0

# Fold-0 gate thresholds
FOLD0_BA_THRESHOLD = 0.96
FOLD0_ERR_CORR_THRESHOLD = 0.65

# Gamma micro-sweep candidates (3 values, fold-0 only)
N_FEATURES_CORE = len(CORE_COLS) + len(COLOR_PAIRS)  # 13


def build_core_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build the ~13-dim core photometric feature vector.
    Stateless: row-wise, no fitting, no target, no cross-row stats.
    Returns float32 array shape (n, 13).
    """
    cols = []
    # 6 base columns
    for c in CORE_COLS:
        cols.append(df[c].values.astype(np.float32))
    # 7 color pairs
    for a, b in COLOR_PAIRS:
        cols.append((df[a].values - df[b].values).astype(np.float32))
    return np.column_stack(cols)


def fit_nystroem_pipeline(
    X_tr: np.ndarray,
    gamma: float,
    rng: np.random.RandomState,
) -> tuple[StandardScaler, Nystroem]:
    """
    Fit StandardScaler + Nystroem on TRAIN-FOLD rows only.
    Nystroem landmarks are sampled from up to NYSTROEM_SUBSAMPLE rows.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)

    # For landmark sampling, use a subsample of train-fold
    if len(X_tr_scaled) > NYSTROEM_SUBSAMPLE:
        idx = rng.choice(len(X_tr_scaled), NYSTROEM_SUBSAMPLE, replace=False)
        X_landmark_source = X_tr_scaled[idx]
    else:
        X_landmark_source = X_tr_scaled

    nystroem = Nystroem(
        kernel="rbf",
        gamma=gamma,
        n_components=N_COMPONENTS,
        random_state=rng.randint(0, 2**31),
        n_jobs=1,
    )
    nystroem.fit(X_landmark_source)

    return scaler, nystroem


def apply_pipeline(
    X: np.ndarray,
    scaler: StandardScaler,
    nystroem: Nystroem,
) -> np.ndarray:
    """Apply fitted scaler + Nystroem transform."""
    return nystroem.transform(scaler.transform(X))


def make_logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        C=LOGREG_C,
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

# ─── Load node_0070 OOF (for err-corr gate) ──────────────────────────────────
log("Loading node_0070 OOF for error-correlation gate ...")
oof_70 = np.load(COMP_DIR / "nodes/node_0070/oof.npy")
assert oof_70.shape == (n_train, N_CLASSES), f"Expected ({n_train},{N_CLASSES}), got {oof_70.shape}"
y_pred_70 = oof_70.argmax(axis=1)
log(f"  node_0070 OOF errors: {(y_pred_70 != y_all).sum()} / {n_train}")

# ─── Pre-flight leakage check 1: TARGET and ID not in feature list ─────────
log("Pre-flight leakage checks ...")
assert TARGET not in CORE_COLS, f"TARGET {TARGET} is a core column — LEAK!"
assert IDC not in CORE_COLS, f"ID {IDC} is a core column — LEAK!"
feature_names = CORE_COLS + [f"{a}-{b}" for a, b in COLOR_PAIRS]
assert TARGET not in feature_names, f"TARGET in features — LEAK!"
assert IDC not in feature_names, f"ID in features — LEAK!"
log(f"  check1 PASS: target/id absent from {len(feature_names)} feature columns")

# ─── Build stateless core features ──────────────────────────────────────────
log("Building stateless core features ...")
X_train_core = build_core_features(train_raw)
X_test_core = build_core_features(test_raw)
log(f"  X_train_core={X_train_core.shape}  X_test_core={X_test_core.shape}")
assert X_train_core.shape[1] == N_FEATURES_CORE, f"Expected {N_FEATURES_CORE} cols, got {X_train_core.shape[1]}"

# ─── Pre-flight leakage check 2: single-feature sweep on sample ──────────────
log("Pre-flight leakage check 2: single-feature correlation sweep ...")
sample_size = min(50_000, n_train)
rng_check = np.random.RandomState(0)
sample_idx = rng_check.choice(n_train, sample_size, replace=False)
s_X = X_train_core[sample_idx]
s_y = y_all[sample_idx]
leaked_cols = []
for fi in range(s_X.shape[1]):
    x = s_X[:, fi]
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

# ─── Check 6: train/test near-dup check on sample ────────────────────────────
# For tabular photometry, exact-match is unlikely — spot-check a tiny sample
log("Check 6: train/test near-dup spot check ...")
tr_sample = X_train_core[:1000].round(4)
te_sample = X_test_core[:1000].round(4)
tr_set = set(map(tuple, tr_sample.tolist()))
te_set = set(map(tuple, te_sample.tolist()))
overlap = tr_set & te_set
log(f"  check6: exact-match overlap (1k sample each): {len(overlap)} rows — {'warn' if len(overlap) > 0 else 'clean'}")

# ─── PRE-FLIGHT COMPLETE — launch training ───────────────────────────────────

# ─── Gamma micro-sweep on fold-0 train split (quick, before full loop) ───────
log("=== GAMMA MICRO-SWEEP on fold-0 ===")
fold0_info = folds_list[0]
val_idx0 = np.asarray(fold0_info["val_idx"])
tr_idx0 = np.setdiff1d(np.arange(n_train), val_idx0)

X_tr0 = X_train_core[tr_idx0]
X_val0 = X_train_core[val_idx0]
y_tr0 = y_all[tr_idx0]
y_val0 = y_all[val_idx0]

# 3-point gamma sweep
feature_var = X_tr0.var(axis=0).mean()
gamma_candidates = [
    1.0 / N_FEATURES_CORE,
    1.0 / (N_FEATURES_CORE * feature_var + 1e-8),
    0.1 / N_FEATURES_CORE,
]
log(f"  gamma candidates: {[f'{g:.6f}' for g in gamma_candidates]}")

best_gamma = None
best_gamma_ba = -1.0

# Use a small subsample for the sweep to keep it cheap
sweep_size = min(20_000, len(tr_idx0))
rng_sweep = np.random.RandomState(SEED)
sweep_tr_idx = rng_sweep.choice(len(tr_idx0), sweep_size, replace=False)
X_sweep_tr = X_tr0[sweep_tr_idx]
y_sweep_tr = y_tr0[sweep_tr_idx]

for gamma in gamma_candidates:
    t_g = time.perf_counter()
    rng_g = np.random.RandomState(SEED)
    scaler_g, nystroem_g = fit_nystroem_pipeline(X_sweep_tr, gamma=gamma, rng=rng_g)
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
gc.collect()

# ─── OOF loop ─────────────────────────────────────────────────────────────────
oof_proba = np.zeros((n_train, N_CLASSES), dtype=np.float32)
test_proba_accum = np.zeros((n_test, N_CLASSES), dtype=np.float32)
per_fold_scores = []
fold0_gate_passed = False
fold0_err_corr = None
fold0_ba = None

log("Starting OOF loop ...")

for fi in folds_list:
    fold_id = fi["fold"]
    val_idx = np.asarray(fi["val_idx"])
    tr_idx = np.setdiff1d(np.arange(n_train), val_idx)
    fold_seed = SEED + (fold_id + 1) * 100

    log(f"Fold {fold_id}: train={len(tr_idx)} val={len(val_idx)}")
    fold_t0 = time.perf_counter()

    X_tr_fold = X_train_core[tr_idx]
    X_val_fold = X_train_core[val_idx]
    y_tr_fold = y_all[tr_idx]
    y_val_fold = y_all[val_idx]

    # ─── Check 4: fs_rbf_nystroem fit_in_fold — TRAIN FOLD ONLY ──────────────
    # Proof by code: scaler + Nystroem landmarks are fit here on tr_idx rows ONLY.
    rng_fold = np.random.RandomState(fold_seed)
    scaler_fold, nystroem_fold = fit_nystroem_pipeline(
        X_tr_fold, gamma=best_gamma, rng=rng_fold
    )

    # Apply to val and test (transform only, never fit on val/test)
    X_tr_mapped = apply_pipeline(X_tr_fold, scaler_fold, nystroem_fold)
    X_val_mapped = apply_pipeline(X_val_fold, scaler_fold, nystroem_fold)
    X_te_mapped = apply_pipeline(X_test_core, scaler_fold, nystroem_fold)

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

    # ─── FOLD-0 GATE ─────────────────────────────────────────────────────────
    if fold_id == 0:
        fold_time = fold_elapsed
        projected_total = fold_time * len(folds_list)
        log(f"  TIMING: fold0={fold_time:.1f}s  projected_5fold={projected_total:.1f}s "
            f"({projected_total/60:.1f}min)")

        # Error-correlation vs node_0070 fold-0
        val_pred_96 = val_proba.argmax(1)
        val_pred_70 = y_pred_70[val_idx]
        y_val_true = y_all[val_idx]

        err_96 = (val_pred_96 != y_val_true).astype(np.float32)
        err_70 = (val_pred_70 != y_val_true).astype(np.float32)

        fold0_err_corr = float(np.corrcoef(err_96, err_70)[0, 1])
        fold0_ba = fold_score

        log(f"  FOLD-0 GATE: solo_BA={fold0_ba:.6f} (threshold>={FOLD0_BA_THRESHOLD})")
        log(f"  FOLD-0 GATE: err_corr_vs_n70={fold0_err_corr:.4f} (threshold<{FOLD0_ERR_CORR_THRESHOLD})")

        print(f"fold0_solo_ba={fold0_ba:.6f}", flush=True)
        print(f"fold0_err_corr_vs_n70={fold0_err_corr:.6f}", flush=True)
        print(f"best_gamma={best_gamma:.8f}", flush=True)

        if fold0_ba < FOLD0_BA_THRESHOLD:
            log(f"  FOLD-0 GATE FAILED: solo BA {fold0_ba:.6f} < {FOLD0_BA_THRESHOLD}")
            log("  KILL: stopping after fold 0 (BA below weak-base tier 0.96)")
            print(f"GATE_KILL: solo_BA={fold0_ba:.6f} below threshold", flush=True)
            fold0_gate_passed = False
        elif fold0_err_corr >= FOLD0_ERR_CORR_THRESHOLD:
            log(f"  FOLD-0 GATE FAILED: err_corr {fold0_err_corr:.4f} >= {FOLD0_ERR_CORR_THRESHOLD}")
            log("  KILL: Nystroem RBF did NOT decorrelate — null result, stopping")
            print(f"GATE_KILL: err_corr={fold0_err_corr:.6f} above threshold", flush=True)
            fold0_gate_passed = False
        else:
            log(f"  FOLD-0 GATE PASSED: BA={fold0_ba:.6f}>={FOLD0_BA_THRESHOLD}, "
                f"err_corr={fold0_err_corr:.4f}<{FOLD0_ERR_CORR_THRESHOLD}")
            log("  Continuing to full 5-fold run ...")
            fold0_gate_passed = True

    # Clean up fold memory
    del X_tr_fold, X_val_fold, X_tr_mapped, X_val_mapped, X_te_mapped
    del scaler_fold, nystroem_fold, model
    gc.collect()

    if fold_id == 0 and not fold0_gate_passed:
        log("KILL DECISION: stopping after fold 0 — node is a null result, no stack use.")
        break

# ─── Post-OOF checks ─────────────────────────────────────────────────────────
if fold0_gate_passed:
    # Full 5-fold run completed
    mean_cv = float(np.mean(per_fold_scores))
    sem_cv = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
    log(f"per_fold=" + ",".join(f"{s:.6f}" for s in per_fold_scores))
    log(f"cv={mean_cv:.6f}+/-{sem_cv:.6f}")
    print(f"cv={mean_cv:.6f}", flush=True)

    # ─── Check 7: OOF complete ────────────────────────────────────────────────
    assert not np.any(np.isnan(oof_proba)), "NaN in OOF probs!"
    covered = (oof_proba.sum(axis=1) > 0)
    assert covered.all(), f"OOF has {(~covered).sum()} uncovered rows!"
    log(f"  check7 PASS: OOF complete, no NaN ({n_train} rows)")

    # ─── Check 8: distribution sane ──────────────────────────────────────────
    prob_sums = oof_proba.sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-3), f"OOF probs don't sum to 1"
    assert oof_proba.min() >= -1e-6, f"OOF probs < 0: {oof_proba.min()}"
    assert oof_proba.max() <= 1.0 + 1e-6, f"OOF probs > 1: {oof_proba.max()}"
    log(f"  check8 PASS: OOF distribution sane (probs in [0,1], sum to 1)")

    oof_metric = balanced_accuracy_score(y_all, oof_proba.argmax(1))
    log(f"OOF full balanced_accuracy={oof_metric:.6f}")

    # ─── Save OOF ────────────────────────────────────────────────────────────
    np.save(NODE_DIR / "oof.npy", oof_proba)
    log(f"Saved oof.npy shape={oof_proba.shape}")

    # ─── Save test_probs ─────────────────────────────────────────────────────
    np.save(NODE_DIR / "test_probs.npy", test_proba_accum)
    log(f"Saved test_probs.npy shape={test_proba_accum.shape}")

    # ─── Write submission ─────────────────────────────────────────────────────
    pred_labels = np.array([CLASSES[i] for i in test_proba_accum.argmax(1)])
    sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
    sub = sub[list(sample_sub.columns)]
    sub_path = NODE_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    log(f"Saved submission.csv shape={sub.shape}")
    log(f"  prediction distribution: {dict(zip(*np.unique(pred_labels, return_counts=True)))}")

else:
    # GATE KILLED — partial OOF (fold-0 only), write what we have for diagnostics
    log("GATE KILLED — writing fold-0 partial artifacts for diagnostics ...")

    # Save partial OOF (fold-0 only is filled; rest are zeros — mark as null)
    # We write the zeros array so the shape is correct but flag it as partial
    np.save(NODE_DIR / "oof.npy", oof_proba)
    log(f"Saved partial oof.npy shape={oof_proba.shape} (fold-0 only, rest zeros)")

    # Save partial test_probs (fold-0 scaled)
    test_proba_scaled = test_proba_accum * len(folds_list)  # undo the /n_folds division
    np.save(NODE_DIR / "test_probs.npy", test_proba_scaled)
    log(f"Saved partial test_probs.npy shape={test_proba_scaled.shape} (fold-0 only)")

    # Write submission from fold-0 only (diagnostic, not valid for submission)
    pred_labels = np.array([CLASSES[i] for i in test_proba_scaled.argmax(1)])
    sub = pd.DataFrame({IDC: test_raw[IDC].values, TARGET: pred_labels})
    sub = sub[list(sample_sub.columns)]
    sub_path = NODE_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)
    log(f"Saved diagnostic submission.csv (fold-0 only)")

    log(f"GATE KILL SUMMARY:")
    log(f"  fold0_solo_BA={fold0_ba:.6f} (threshold>={FOLD0_BA_THRESHOLD})")
    log(f"  fold0_err_corr_vs_n70={fold0_err_corr:.4f} (threshold<{FOLD0_ERR_CORR_THRESHOLD})")
    log(f"  best_gamma={best_gamma:.8f}")
    print(f"GATE_KILL_SUMMARY: fold0_ba={fold0_ba:.6f} err_corr={fold0_err_corr:.4f}", flush=True)

total_elapsed = time.perf_counter() - T0
log(f"Done. Total elapsed={total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
