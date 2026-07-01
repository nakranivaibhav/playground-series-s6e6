"""Diagnostic: does the MLP (node_0008) earn weight in the blend and lift the honest CV?

Reuses node_0007's fold-honest nested protocol. NOT a node — decides whether to build
node_0009 (a 4-arm combine) or leave node_0007 as champion.
"""
import json
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

COMP = Path(__file__).resolve().parents[3]
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LABEL_ORDER)}
ARMS = {"n1": "node_0001", "n4": "node_0004", "n6": "node_0006", "n8": "node_0008"}
N_CLASSES, STEP = 3, 0.05

y = pd.read_csv(COMP / "data/train.csv")["class"].map(L2I).to_numpy()
n = len(y)
folds = json.loads((COMP / "folds.json").read_text())["folds"]
fold_val = [np.asarray(f["val_idx"]) for f in folds]
all_idx = np.arange(n)
P = {k: np.load(COMP / "nodes" / v / "oof.npy") for k, v in ARMS.items()}


def fast_balacc(yt, yp):
    return float(np.mean([(yp[yt == c] == c).mean() for c in range(N_CLASSES) if (yt == c).any()]))


def simplex(k, step):
    m = int(round(1 / step))
    out = []
    for combo in product(range(m + 1), repeat=k - 1):
        if sum(combo) <= m:
            out.append(tuple(c / m for c in combo) + ((m - sum(combo)) / m,))
    return out


def blend_pred(keys, w, rows):
    acc = sum(wi * P[k][rows] for wi, k in zip(w, keys) if wi)
    return np.argmax(acc, axis=1)


def honest_cv(keys):
    cand = simplex(len(keys), STEP)
    uniform = tuple([1 / len(keys)] * len(keys))
    scores, weights = [], []
    for val in fold_val:
        other = np.setdiff1d(all_idx, val)
        best_s, best_w = -1, None
        for w in cand:
            s = fast_balacc(y[other], blend_pred(keys, w, other))
            if s > best_s + 1e-12 or (abs(s - best_s) <= 1e-12 and best_w is not None
                                       and sum((a-b)**2 for a, b in zip(w, uniform)) < sum((a-b)**2 for a, b in zip(best_w, uniform))):
                best_s, best_w = s, w
        scores.append(fast_balacc(y[val], blend_pred(keys, best_w, val)))
        weights.append(best_w)
    # full-OOF weights
    best_s, best_w = -1, None
    for w in cand:
        s = fast_balacc(y, blend_pred(keys, w, all_idx))
        if s > best_s + 1e-12:
            best_s, best_w = s, w
    return float(np.mean(scores)), float(np.std(scores, ddof=1)/np.sqrt(len(scores))), best_w, weights


# error-correlation matrix (per-row "is wrong" indicators)
print("=== error-correlation matrix (corr of per-row error indicators) ===")
err = {k: (np.argmax(P[k], 1) != y).astype(float) for k in ARMS}
ks = list(ARMS)
print("      " + "  ".join(f"{k:>6}" for k in ks))
for a in ks:
    row = "  ".join(f"{np.corrcoef(err[a], err[b])[0,1]:6.3f}" for b in ks)
    print(f"{a:>4}  {row}")
print("solo balacc:", {k: round(fast_balacc(y, np.argmax(P[k], 1)), 6) for k in ks})

print("\n=== fold-honest nested CV by arm-set ===")
for keys in [["n6"], ["n6", "n4", "n1"], ["n6", "n4", "n8"], ["n6", "n8"],
             ["n6", "n4", "n1", "n8"]]:
    cv, sem, wfull, _ = honest_cv(keys)
    wtxt = ", ".join(f"{k}:{w:.2f}" for k, w in zip(keys, wfull))
    print(f"  {'+'.join(keys):<18} cv={cv:.6f}±{sem:.6f}  final_w=({wtxt})")
print("\n(node_0007 champion = n6+n4+n1 = 0.965530; build node_0009 only if a set with n8 beats it beyond sem)")
