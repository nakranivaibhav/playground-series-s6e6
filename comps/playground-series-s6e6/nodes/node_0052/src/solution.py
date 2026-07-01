"""node_0052 — Revival re-stack: CORE15 + discarded candidates sweep.

THE ONE ATOMIC CHANGE:
  Pure OOF arithmetic — no model training on raw data. Re-fit the
  balanced-LogReg meta + DE per-class threshold on CORE15 OOF augmented with
  previously-discarded bases: node_0042 (RealMLP config-B), node_0043
  (CatBoost config-B), node_0049 (binary chain), node_0050 (OvR). NOTE:
  node_0011 is already in CORE15 so it is skipped.

  Reports fold-honest stacked CV for:
    1. CORE15 alone (sanity check, should ≈ 0.969808)
    2. CORE15 + each single candidate (4 runs)
    3. Greedy forward selection of cumulative combos (best-1, best-2, ...)

  Emits oof.npy + test_probs.npy + submission.csv for the best config.

Leakage discipline:
  - No .fit() on raw data anywhere. All inputs are already-gated OOF arrays.
  - Meta LogReg is fit inside each fold (nested-CV, train-fold only).
  - DE threshold is fit on out-of-fold complement at eval time.
  - Frozen folds.json used throughout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

T0 = time.perf_counter()


def log(msg: str):
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


# ─── Constants ───────────────────────────────────────────────────────────────
TARGET = "class"
IDC = "id"
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
NC = 3

# CORE15 bases (same as node_0041 / node_0047)
BASES = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030",
    "node_0039",
]
CORE15_SET = set(BASES)

# Candidate discarded bases (skip any already in CORE15)
ALL_CANDIDATES = ["node_0042", "node_0043", "node_0049", "node_0050", "node_0011"]
CANDIDATES = [c for c in ALL_CANDIDATES if c not in CORE15_SET]
log(f"CORE15 has {len(BASES)} bases. Candidates after skipping CORE15 members: {CANDIDATES}")


# ─── Stack helpers ───────────────────────────────────────────────────────────
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


def stack_cv(OOF_mat: np.ndarray, y_all: np.ndarray, fval: list) -> tuple[float, float, list, np.ndarray]:
    """Fold-honest meta + DE threshold scoring. Returns (mean, sem, per_fold, stack_oof)."""
    n = len(y_all)
    stack_oof = np.zeros((n, NC))
    for vi in fval:
        tr = np.setdiff1d(np.arange(n), vi)
        m = fit_meta(OOF_mat[tr], y_all[tr])
        stack_oof[vi] = m.predict_proba(OOF_mat[vi])

    per_fold = []
    for vi in fval:
        other = np.setdiff1d(np.arange(n), vi)
        w = best_thr_de(stack_oof[other], y_all[other])
        pred = np.argmax(stack_oof[vi] * w, axis=1)
        per_fold.append(score_fn(y_all[vi], pred))

    mean_ = float(np.mean(per_fold))
    sem_  = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
    return mean_, sem_, per_fold, stack_oof


# ─── Load data ───────────────────────────────────────────────────────────────
log("Loading data ...")
train_raw  = pd.read_csv(COMP_DIR / "data/train.csv")
test_raw   = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
log(f"  train={train_raw.shape}  test={test_raw.shape}  folds={len(folds_list)}")

y_all  = train_raw[TARGET].map(LABEL_MAP).astype(int).values
n_train = len(train_raw)
n_test  = len(test_raw)
fval   = [np.asarray(f["val_idx"]) for f in folds_list]

nodes_dir = COMP_DIR / "nodes"


# ─── Load CORE15 OOF + test ─────────────────────────────────────────────────
log("Loading CORE15 OOF + test probs ...")
OOF_CORE  = np.concatenate([logp(np.load(nodes_dir / b / "oof.npy"))       for b in BASES], axis=1)
TEST_CORE = np.concatenate([logp(np.load(nodes_dir / b / "test_probs.npy")) for b in BASES], axis=1)
log(f"  CORE15 OOF={OOF_CORE.shape}  TEST={TEST_CORE.shape}")

# ─── Load candidate OOFs ─────────────────────────────────────────────────────
log("Loading candidate OOFs ...")
cand_oof  = {}
cand_test = {}
for c in CANDIDATES:
    cand_oof[c]  = logp(np.load(nodes_dir / c / "oof.npy"))
    cand_test[c] = logp(np.load(nodes_dir / c / "test_probs.npy"))
    log(f"  {c}: oof={cand_oof[c].shape}  test={cand_test[c].shape}")


# ─── Step 1: CORE15 baseline (sanity check) ──────────────────────────────────
log("=" * 60)
log("STEP 1: CORE15 baseline (sanity check) ...")
baseline_cv, baseline_sem, baseline_folds, baseline_stack_oof = stack_cv(OOF_CORE, y_all, fval)
log(f"  CORE15 cv={baseline_cv:.6f}  sem={baseline_sem:.6f}  folds={baseline_folds}")
print(f"CORE15_baseline cv={baseline_cv:.6f}  sem={baseline_sem:.6f}", flush=True)

EXPECTED = 0.969808
if abs(baseline_cv - EXPECTED) > 0.0005:
    log(f"SANITY CHECK FAILED: expected ~{EXPECTED}, got {baseline_cv:.6f}. Delta={baseline_cv-EXPECTED:.6f}. STOPPING.")
    sys.exit(1)
log(f"Sanity check PASSED (delta={baseline_cv - EXPECTED:.6f})")


# ─── Step 2: CORE15 + each single candidate ──────────────────────────────────
log("=" * 60)
log("STEP 2: single-candidate additions ...")
single_results = {}  # cand -> (cv, sem, per_fold)
for c in CANDIDATES:
    OOF_aug  = np.concatenate([OOF_CORE,  cand_oof[c]],  axis=1)
    cv, sem_, pf, _ = stack_cv(OOF_aug, y_all, fval)
    delta = cv - baseline_cv
    single_results[c] = (cv, sem_, pf, delta)
    log(f"  CORE15+{c}: cv={cv:.6f}  sem={sem_:.6f}  delta={delta:+.6f}")
    print(f"CORE15+{c} cv={cv:.6f}  sem={sem_:.6f}  delta={delta:+.6f}", flush=True)


# ─── Step 3: greedy forward selection ────────────────────────────────────────
log("=" * 60)
log("STEP 3: greedy forward selection ...")

# Sort candidates by single-result delta descending
sorted_cands = sorted(CANDIDATES, key=lambda c: single_results[c][3], reverse=True)
log(f"  Sorted by delta: {[(c, f'{single_results[c][3]:+.6f}') for c in sorted_cands]}")

# Only consider candidates that individually helped (delta > 0)
helpful = [c for c in sorted_cands if single_results[c][3] > 0]
log(f"  Candidates that individually helped (delta>0): {helpful}")

greedy_results = []  # (combo_list, cv, sem, per_fold)
current_bases_oof  = OOF_CORE
current_bases_test = TEST_CORE
current_combo = []
current_cv = baseline_cv

for c in helpful:
    new_oof  = np.concatenate([current_bases_oof,  cand_oof[c]],  axis=1)
    new_test = np.concatenate([current_bases_test, cand_test[c]], axis=1)
    cv, sem_, pf, _ = stack_cv(new_oof, y_all, fval)
    delta = cv - current_cv
    combo = current_combo + [c]
    greedy_results.append((list(combo), cv, sem_, pf, delta))
    log(f"  CORE15+{combo}: cv={cv:.6f}  sem={sem_:.6f}  delta_marginal={delta:+.6f}")
    print(f"CORE15+{combo} cv={cv:.6f}  sem={sem_:.6f}  delta={cv-baseline_cv:+.6f}", flush=True)
    if delta > 0:
        current_combo = combo
        current_bases_oof  = new_oof
        current_bases_test = new_test
        current_cv = cv
    else:
        log(f"    -> marginal delta non-positive, stopping greedy at {current_combo}")
        break


# ─── Find best config overall ────────────────────────────────────────────────
log("=" * 60)
log("Finding best config ...")

all_configs = [("CORE15", [], baseline_cv, baseline_sem, baseline_folds)]
for c, (cv, sem_, pf, delta) in single_results.items():
    all_configs.append((f"CORE15+{c}", [c], cv, sem_, pf))
for combo, cv, sem_, pf, _ in greedy_results:
    all_configs.append((f"CORE15+{combo}", combo, cv, sem_, pf))

# Sort by cv descending
all_configs.sort(key=lambda x: x[2], reverse=True)

best_name, best_extra, best_cv, best_sem, best_folds = all_configs[0]
log(f"Best config: {best_name}  cv={best_cv:.6f}  sem={best_sem:.6f}")

# Print sweep table
log("=" * 60)
log("FULL SWEEP TABLE:")
log(f"{'Config':<40} {'CV':>10} {'±SEM':>8} {'ΔCV':>9}")
for name, extra, cv, sem_, pf in all_configs:
    delta = cv - baseline_cv
    log(f"  {name:<38} {cv:.6f}  {sem_:.6f}  {delta:+.6f}")


# ─── Emit artifacts for the best config ──────────────────────────────────────
log("=" * 60)
log(f"Emitting artifacts for best config: {best_name} ...")

# Build the OOF + TEST for best config
best_extra_set = best_extra  # list of extra candidates to add
OOF_best  = OOF_CORE
TEST_best = TEST_CORE
for c in best_extra_set:
    OOF_best  = np.concatenate([OOF_best,  cand_oof[c]],  axis=1)
    TEST_best = np.concatenate([TEST_best, cand_test[c]], axis=1)

# Recompute to get final stack_oof
_, _, _, best_stack_oof = stack_cv(OOF_best, y_all, fval)

# Final meta on full OOF
meta_full = fit_meta(OOF_best, y_all)
w_full    = best_thr_de(best_stack_oof, y_all)
log(f"Final w=[{w_full[0]:.4f},{w_full[1]:.4f},{w_full[2]:.4f}]")

# Test predictions
stack_test_probs  = meta_full.predict_proba(TEST_best)
test_preds_idx    = np.argmax(stack_test_probs * w_full, axis=1)
test_labels       = [CLASSES[i] for i in test_preds_idx]

sub = pd.DataFrame({"id": test_raw["id"], "class": test_labels})
sub = sub[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
log(f"Saved submission.csv ({len(sub)} rows)")

# Save OOF (best_stack_oof: N x 3 probabilities)
np.save(NODE_DIR / "oof.npy",        best_stack_oof.astype("float32"))
np.save(NODE_DIR / "test_probs.npy", stack_test_probs.astype("float32"))
log(f"Saved oof.npy={best_stack_oof.shape}  test_probs.npy={stack_test_probs.shape}")

# features.txt — the stacked column names (not raw features, but base node names)
feat_names = list(BASES) + list(best_extra_set)
(NODE_SRC / "features.txt").write_text("\n".join(feat_names) + "\n")
log(f"Wrote features.txt ({len(feat_names)} base nodes)")

# Schema checks
assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
assert len(sub) == len(sample_sub), "row count mismatch"
log("Submission schema OK")

# Final CV summary
print(f"cv={best_cv:.6f}", flush=True)
log(f"cv={best_cv:.6f}  sem={best_sem:.6f}  folds={best_folds}")
log(f"Best config: {best_name}")

total = time.perf_counter() - T0
log(f"Total elapsed: {total:.1f}s ({total/60:.2f}min)")
log("Done.")
