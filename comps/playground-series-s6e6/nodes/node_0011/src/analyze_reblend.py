"""Re-blend with the full-feature GBDT arms (n11 XGB-full, n12 CatBoost-full).

Does swapping base-XGB (n4) -> full-XGB (n11), or adding CatBoost-full (n12), beat the
current champion node_0010 (n6+n4+n1+n9 = 0.965889)? Same fold-honest nested protocol.
"""
import json
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd

COMP = Path(__file__).resolve().parents[3]
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LABEL_ORDER)}
ARMS = {"n1": "node_0001", "n4": "node_0004", "n6": "node_0006", "n9": "node_0009",
        "n11": "node_0011", "n12": "node_0012"}
N_CLASSES = 3
CHAMP = 0.965889

y = pd.read_csv(COMP / "data/train.csv")["class"].map(L2I).to_numpy()
n = len(y)
folds = json.loads((COMP / "folds.json").read_text())["folds"]
fold_val = [np.asarray(f["val_idx"]) for f in folds]
all_idx = np.arange(n)
P = {k: np.load(COMP / "nodes" / v / "oof.npy") for k, v in ARMS.items()}


def fast_balacc(yt, yp):
    return float(np.mean([(yp[yt == c] == c).mean() for c in range(N_CLASSES) if (yt == c).any()]))


def simplex(k, step):
    m = int(round(1 / step)); out = []
    for combo in product(range(m + 1), repeat=k - 1):
        if sum(combo) <= m:
            out.append(tuple(c / m for c in combo) + ((m - sum(combo)) / m,))
    return out


def blend_pred(keys, w, rows):
    acc = sum(wi * P[k][rows] for wi, k in zip(w, keys) if wi)
    return np.argmax(acc, axis=1)


def honest_cv(keys, step):
    cand = simplex(len(keys), step); uni = tuple([1/len(keys)]*len(keys))
    sc = []
    for val in fold_val:
        other = np.setdiff1d(all_idx, val); bs, bw = -1, None
        for w in cand:
            s = fast_balacc(y[other], blend_pred(keys, w, other))
            if s > bs + 1e-12 or (abs(s-bs) <= 1e-12 and bw is not None
                                  and sum((a-b)**2 for a,b in zip(w,uni)) < sum((a-b)**2 for a,b in zip(bw,uni))):
                bs, bw = s, w
        sc.append(fast_balacc(y[val], blend_pred(keys, bw, val)))
    bs, bw = -1, None
    for w in cand:
        s = fast_balacc(y, blend_pred(keys, w, all_idx))
        if s > bs + 1e-12: bs, bw = s, w
    return float(np.mean(sc)), float(np.std(sc, ddof=1)/np.sqrt(len(sc))), bw


print("=== error-corr (per-row error indicators) ===")
err = {k: (np.argmax(P[k], 1) != y).astype(float) for k in ARMS}
ks = list(ARMS)
print("       " + "  ".join(f"{k:>5}" for k in ks))
for a in ks:
    print(f"{a:>5}  " + "  ".join(f"{np.corrcoef(err[a], err[b])[0,1]:5.2f}" for b in ks))
print("solo:", {k: round(fast_balacc(y, np.argmax(P[k], 1)), 6) for k in ks})

print(f"\n=== fold-honest nested CV  (champion node_0010 = n6+n4+n1+n9 = {CHAMP}) ===")
SETS = [(["n6","n4","n1","n9"], 0.05), (["n6","n11","n1","n9"], 0.05),
        (["n6","n11","n9"], 0.05), (["n6","n11","n4","n1","n9"], 0.10),
        (["n6","n11","n1","n9","n12"], 0.10), (["n6","n11","n4","n1","n9","n12"], 0.10)]
for keys, step in SETS:
    cv, sem, wf = honest_cv(keys, step)
    wtxt = ", ".join(f"{k}:{w:.2f}" for k, w in zip(keys, wf) if w > 0)
    d = cv - CHAMP
    flag = (f"  BEATS by {d:+.6f}" + ("  (>2sem!)" if d > 2*sem else "  (within noise)")) if cv > CHAMP else f"  ({d:+.6f})"
    print(f"  {'+'.join(keys):<26}(s{step}) cv={cv:.6f}±{sem:.6f}  w=({wtxt}){flag}")
print("\n(build node_0013 only if a set beats 0.965889 beyond its sem)")
