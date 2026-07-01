"""Diagnostic: does TabM (node_0009) lift the blend past champion node_0007 (0.965530)?

Same fold-honest nested protocol as node_0007. Includes all arms (n1,n4,n6,n8,n9).
Decides whether to build node_0010 (a combine including TabM).
"""
import json
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

COMP = Path(__file__).resolve().parents[3]
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LABEL_ORDER)}
ARMS = {"n1": "node_0001", "n4": "node_0004", "n6": "node_0006",
        "n8": "node_0008", "n9": "node_0009"}
N_CLASSES = 3

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


def honest_cv(keys, step):
    cand = simplex(len(keys), step)
    uni = tuple([1 / len(keys)] * len(keys))
    scores = []
    for val in fold_val:
        other = np.setdiff1d(all_idx, val)
        bs, bw = -1, None
        for w in cand:
            s = fast_balacc(y[other], blend_pred(keys, w, other))
            if s > bs + 1e-12 or (abs(s - bs) <= 1e-12 and bw is not None
                                  and sum((a-b)**2 for a, b in zip(w, uni)) < sum((a-b)**2 for a, b in zip(bw, uni))):
                bs, bw = s, w
        scores.append(fast_balacc(y[val], blend_pred(keys, bw, val)))
    bs, bw = -1, None
    for w in cand:
        s = fast_balacc(y, blend_pred(keys, w, all_idx))
        if s > bs + 1e-12:
            bs, bw = s, w
    return float(np.mean(scores)), float(np.std(scores, ddof=1)/np.sqrt(len(scores))), bw


print("=== error-correlation matrix (per-row error indicators) ===")
err = {k: (np.argmax(P[k], 1) != y).astype(float) for k in ARMS}
ks = list(ARMS)
print("      " + "  ".join(f"{k:>6}" for k in ks))
for a in ks:
    print(f"{a:>4}  " + "  ".join(f"{np.corrcoef(err[a], err[b])[0,1]:6.3f}" for b in ks))
print("solo balacc:", {k: round(fast_balacc(y, np.argmax(P[k], 1)), 6) for k in ks})

print("\n=== fold-honest nested CV by arm-set  (champion node_0007 = n6+n4+n1 = 0.965530) ===")
SETS = [(["n6"], 0.05), (["n6", "n4", "n1"], 0.05), (["n9"], 0.05),
        (["n6", "n9"], 0.05), (["n6", "n4", "n9"], 0.05),
        (["n6", "n4", "n1", "n9"], 0.05), (["n6", "n4", "n1", "n8", "n9"], 0.10)]
champ = 0.965530
for keys, step in SETS:
    cv, sem, wfull = honest_cv(keys, step)
    wtxt = ", ".join(f"{k}:{w:.2f}" for k, w in zip(keys, wfull))
    delta = cv - champ
    flag = ""
    if cv > champ:
        flag = f"  BEATS champ by {delta:+.6f}" + ("  (>2·sem!)" if delta > 2*sem else "  (within noise)")
    print(f"  {'+'.join(keys):<22}(step {step}) cv={cv:.6f}±{sem:.6f}  w=({wtxt}){flag}")
print("\n(build node_0010 only if a set with n9 beats 0.965530 beyond its sem)")
