"""node_0007 — combine: fold-honest weighted-probability blend of n6 + n4 (+ n1).

One atomic change: instead of training a model, average the class-posterior
matrices that node_0006 (champion LightGBM+research feats), node_0004 (XGBoost,
best de-correlated arm) and node_0001 (base LightGBM) already saved, under
fold-honestly chosen simplex weights.

WHY a weighted probability-average and not a meta-learner: an LR/softmax meta
optimizes logloss, which is not the competition metric — analysis showed it
regresses balanced accuracy. A convex average of calibrated posteriors keeps the
champion's calibration and only corrects boundary argmax flips where the
de-correlated XGB disagrees.

Fold-honest protocol (this is the leakage-critical part of a blend):
  for each fold f:
      w_f = argmax_{w in simplex}  balacc( y[other folds], argmax  sum_a w_a P_a[other folds] )
      score_f =                    balacc( y[fold f],     argmax  sum_a w_f,a P_a[fold f] )
The weights scoring fold f are selected WITHOUT ever seeing fold f's labels, so the
reported CV is an honest estimate of the weight-selection procedure's generalization.
Final submission weights are refit on the full OOF (test labels are never involved).

Metric = Balanced Accuracy Score = macro-average per-class recall (maximize).
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

NODE_SRC = Path(__file__).resolve().parent
NODE_DIR = NODE_SRC.parent
COMP_DIR = NODE_DIR.parent.parent

_r = NODE_SRC
while not (_r / "tools" / "leakage_scan.py").exists():
    _r = _r.parent
REPO_ROOT = _r
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TARGET, IDC, DIRECTION = "class", "id", "maximize"
RANDOM_BASELINE = 1.0 / 3.0
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]            # column order of every saved prob matrix
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
ARMS = ["node_0006", "node_0004", "node_0001"]    # all have oof.npy AND test_probs.npy
GRID_STEP = 0.05                                   # simplex resolution for the weight search
N_CLASSES = 3


def fast_balacc(y_int: np.ndarray, pred_int: np.ndarray) -> float:
    """Macro-average per-class recall (== sklearn balanced_accuracy_score, adjusted=False)."""
    recalls = []
    for c in range(N_CLASSES):
        mask = y_int == c
        denom = int(mask.sum())
        if denom:
            recalls.append(float((pred_int[mask] == c).sum()) / denom)
    return float(np.mean(recalls))


def simplex_weights(n_arms: int, step: float):
    """All weight tuples on the n_arms simplex with the given step (sum == 1)."""
    k = int(round(1.0 / step))
    out = []
    for combo in product(range(k + 1), repeat=n_arms - 1):
        if sum(combo) <= k:
            last = k - sum(combo)
            out.append(tuple(c / k for c in combo) + (last / k,))
    return out


def blend_pred(P_list, weights, rows=None):
    """argmax of the weighted probability average over the given rows (or all rows)."""
    acc = None
    for w, P in zip(weights, P_list):
        if w == 0.0:
            continue
        sub = P if rows is None else P[rows]
        acc = w * sub if acc is None else acc + w * sub
    if acc is None:                                # all-zero weights guard
        acc = sum(P if rows is None else P[rows] for P in P_list)
    return np.argmax(acc, axis=1)


# --- best weight on a set of rows; tie-break toward the most uniform vector (robustness) ---
UNIFORM = None  # set after we know n_arms


def best_weights(P_list, y_int, rows, candidates):
    best_score, best_w = -1.0, None
    for w in candidates:
        s = fast_balacc(y_int[rows], blend_pred(P_list, w, rows))
        if s > best_score + 1e-12:
            best_score, best_w = s, w
        elif abs(s - best_score) <= 1e-12:         # tie → prefer closer to uniform
            if best_w is None or _dist2(w, UNIFORM) < _dist2(best_w, UNIFORM):
                best_w = w
    return best_w, best_score


def _dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


print("Loading saved OOF + test probability matrices …")
train = pd.read_csv(COMP_DIR / "data/train.csv")
test = pd.read_csv(COMP_DIR / "data/test.csv")
sample_sub = pd.read_csv(COMP_DIR / "data/sample_submission.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
n = len(train)

P_list, T_list = [], []
for arm in ARMS:
    oof = np.load(COMP_DIR / "nodes" / arm / "oof.npy")
    tp = np.load(COMP_DIR / "nodes" / arm / "test_probs.npy")
    assert oof.shape == (n, N_CLASSES), f"{arm} oof shape {oof.shape}"
    assert tp.shape == (len(test), N_CLASSES), f"{arm} test_probs shape {tp.shape}"
    assert np.allclose(oof.sum(1), 1, atol=1e-3) and np.allclose(tp.sum(1), 1, atol=1e-3)
    P_list.append(oof)
    T_list.append(tp)
print(f"  arms = {ARMS}  (each oof {P_list[0].shape}, test {T_list[0].shape})")

y_int = train[TARGET].map(LABEL2IDX).to_numpy()
assert not np.isnan(y_int).any(), "unmapped label in train.class"

n_arms = len(ARMS)
UNIFORM = tuple([1.0 / n_arms] * n_arms)
candidates = simplex_weights(n_arms, GRID_STEP)
print(f"  weight grid: {len(candidates)} simplex points (step {GRID_STEP})")

all_idx = np.arange(n)
fold_val = [np.asarray(fi["val_idx"]) for fi in folds_list]

# ---------- FOLD-HONEST nested weight search + scoring ----------
print("Fold-honest nested weight search …")
per_fold, per_fold_w = [], []
for fi, val_idx in zip(folds_list, fold_val):
    other = np.setdiff1d(all_idx, val_idx)                 # the OTHER folds' OOF rows
    w_f, train_s = best_weights(P_list, y_int, other, candidates)
    score_f = fast_balacc(y_int[val_idx], blend_pred(P_list, w_f, val_idx))
    per_fold.append(score_f)
    per_fold_w.append(w_f)
    wtxt = ", ".join(f"{a.split('_')[1]}:{w:.2f}" for a, w in zip(ARMS, w_f))
    print(f"  fold {fi['fold']}: w=({wtxt})  -> balanced_accuracy = {score_f:.6f}")

mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}   (HONEST nested)")

# ---------- final weights on the full OOF (for the test submission) ----------
w_final, full_oof_s = best_weights(P_list, y_int, all_idx, candidates)
wtxt = ", ".join(f"{a}:{w:.3f}" for a, w in zip(ARMS, w_final))
print(f"final weights (full-OOF fit): {wtxt}   full_oof_balacc={full_oof_s:.6f}")

# ---------- reference points (transparency, not used for the decision) ----------
def honest_fixed(weights):
    s = [fast_balacc(y_int[v], blend_pred(P_list, weights, v)) for v in fold_val]
    return float(np.mean(s))

champ_only = honest_fixed((0.0, 0.0, 0.0) if False else tuple(1.0 if a == "node_0006" else 0.0 for a in ARMS))
fifty = honest_fixed(tuple({"node_0006": 0.5, "node_0004": 0.5}.get(a, 0.0) for a in ARMS))
print(f"reference: champion-only(n6)={champ_only:.6f}  n6+n4 50/50={fifty:.6f}  uniform={honest_fixed(UNIFORM):.6f}")


# ---------- emit blended OOF + test submission ----------
blend_oof = sum(w * P for w, P in zip(w_final, P_list))
np.save(NODE_DIR / "oof.npy", blend_oof)
blend_test = sum(w * T for w, T in zip(w_final, T_list))
np.save(NODE_DIR / "test_probs.npy", blend_test)
labels = np.array([LABEL_ORDER[i] for i in np.argmax(blend_test, axis=1)])
sub = pd.DataFrame({IDC: test[IDC].values, TARGET: labels})[list(sample_sub.columns)]
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"  wrote submission.csv ({len(sub)} rows), saved oof.npy + test_probs.npy")

# a blend uses no raw features — empty features.txt keeps the leakage scan honest
(NODE_SRC / "features.txt").write_text("\n")

(NODE_DIR / "metrics.md").write_text(
    f"""# node_0007 metrics
metric: Balanced Accuracy Score (maximize)
per_fold: [{', '.join(f'{s:.6f}' for s in per_fold)}]
cv: {mean_cv:.6f} ± {sem_cv:.6f}   (HONEST nested fold weights)
full_oof_balacc: {full_oof_s:.6f}   (optimistic — weights fit on all OOF)
final_weights: {{{', '.join(f'{a}:{w:.3f}' for a, w in zip(ARMS, w_final))}}}
per_fold_weights: {[tuple(round(x,2) for x in w) for w in per_fold_w]}
reference: champion_n6={champ_only:.6f}, n6+n4_50/50={fifty:.6f}, uniform={honest_fixed(UNIFORM):.6f}
change: combine — fold-honest weighted-probability-average of {ARMS}. No model retrain.
""")
print("Done.")
