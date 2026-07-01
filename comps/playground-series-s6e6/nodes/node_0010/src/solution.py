"""node_0010 — combine: fold-honest weighted blend of n6 + n4 + n1 + n9 (TabM).

Extends node_0007's 3-arm blend with the TabM arm (node_0009), which is GBDT-strength
solo (0.964215) and de-correlated from the trees (err-corr ~0.82). Diagnostic showed the
4-arm honest CV = 0.965889 ± 0.000141, beating champion node_0007 (0.965530) by +0.000359
(>2·sem), TabM earning the largest weight. node_0008 (plain MLP) is excluded (0 weight).

Same fold-honest protocol as node_0007: for each fold the simplex weights are optimized on
the OTHER folds' OOF and scored on the held-out fold (weights never see the scored fold's
labels); final test weights are refit on the full OOF. No model retrain — averages saved
posteriors. Probability-average, not a meta-learner (which optimizes logloss, not the metric).

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
if str(_r) not in sys.path:
    sys.path.insert(0, str(_r))

TARGET, IDC, DIRECTION = "class", "id", "maximize"
RANDOM_BASELINE = 1.0 / 3.0
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {lbl: i for i, lbl in enumerate(LABEL_ORDER)}
ARMS = ["node_0006", "node_0004", "node_0001", "node_0009"]   # n8 excluded (0 weight)
GRID_STEP = 0.05
N_CLASSES = 3


def fast_balacc(y_int, pred_int):
    return float(np.mean([(pred_int[y_int == c] == c).mean()
                          for c in range(N_CLASSES) if (y_int == c).any()]))


def simplex_weights(n_arms, step):
    k = int(round(1.0 / step))
    out = []
    for combo in product(range(k + 1), repeat=n_arms - 1):
        if sum(combo) <= k:
            out.append(tuple(c / k for c in combo) + ((k - sum(combo)) / k,))
    return out


def blend_pred(P_list, weights, rows=None):
    acc = None
    for w, P in zip(weights, P_list):
        if w == 0.0:
            continue
        sub = P if rows is None else P[rows]
        acc = w * sub if acc is None else acc + w * sub
    if acc is None:
        acc = sum(P if rows is None else P[rows] for P in P_list)
    return np.argmax(acc, axis=1)


def best_weights(P_list, y_int, rows, candidates, uniform):
    best_score, best_w = -1.0, None
    for w in candidates:
        s = fast_balacc(y_int[rows], blend_pred(P_list, w, rows))
        if s > best_score + 1e-12:
            best_score, best_w = s, w
        elif abs(s - best_score) <= 1e-12 and best_w is not None:
            if sum((a - b) ** 2 for a, b in zip(w, uniform)) < sum((a - b) ** 2 for a, b in zip(best_w, uniform)):
                best_w = w
    return best_w, best_score


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
    assert oof.shape == (n, N_CLASSES) and tp.shape == (len(test), N_CLASSES), f"{arm} shape"
    assert np.allclose(oof.sum(1), 1, atol=1e-3) and np.allclose(tp.sum(1), 1, atol=1e-3)
    P_list.append(oof)
    T_list.append(tp)
print(f"  arms = {ARMS}")

y_int = train[TARGET].map(LABEL2IDX).to_numpy()
n_arms = len(ARMS)
uniform = tuple([1.0 / n_arms] * n_arms)
candidates = simplex_weights(n_arms, GRID_STEP)
print(f"  weight grid: {len(candidates)} simplex points (step {GRID_STEP})")

all_idx = np.arange(n)
fold_val = [np.asarray(fi["val_idx"]) for fi in folds_list]

print("Fold-honest nested weight search …")
per_fold, per_fold_w = [], []
for fi, val_idx in zip(folds_list, fold_val):
    other = np.setdiff1d(all_idx, val_idx)
    w_f, _ = best_weights(P_list, y_int, other, candidates, uniform)
    score_f = fast_balacc(y_int[val_idx], blend_pred(P_list, w_f, val_idx))
    per_fold.append(score_f)
    per_fold_w.append(w_f)
    wtxt = ", ".join(f"{a.split('_')[1]}:{w:.2f}" for a, w in zip(ARMS, w_f))
    print(f"  fold {fi['fold']}: w=({wtxt}) -> balanced_accuracy = {score_f:.6f}")

mean_cv = float(np.mean(per_fold))
sem_cv = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
print("per_fold=" + ",".join(f"{s:.6f}" for s in per_fold))
print(f"cv={mean_cv:.6f}±{sem_cv:.6f}  (HONEST nested)")

w_final, full_oof_s = best_weights(P_list, y_int, all_idx, candidates, uniform)
print("final weights (full-OOF): " + ", ".join(f"{a}:{w:.3f}" for a, w in zip(ARMS, w_final))
      + f"  full_oof_balacc={full_oof_s:.6f}")

