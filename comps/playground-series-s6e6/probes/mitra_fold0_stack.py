"""Quick 'is Mitra worth it?' test using ONLY fold-0 OOF (honest: every model
predicted fold-0 val after training on fold-0 train). Internal StratifiedKFold(5)
over the 115470 fold-0 rows: fit an L2 LogReg meta on clipped log-probs WITH vs
WITHOUT Mitra, compare held-out balanced accuracy.

Two views:
  A) incremental over the CHAMPION (n091 as a single super-base) ± Mitra
  B) a small base bank {n070,n033,n063,n040} ± Mitra (closer to real stacking)

Caveat: fold-0-only proxy with an internal split — directional, noisier than the
full 5-fold restack, but a fast go/no-go.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

COMP = Path("comps/playground-series-s6e6")
CLASSES = ["GALAXY", "QSO", "STAR"]
y_all = pd.read_csv(COMP / "data/train.csv")["class"].map({c: i for i, c in enumerate(CLASSES)}).values
idx = np.load(COMP / "nodes/node_0134/oof_fold0_idx.npy")
y = y_all[idx]
mitra = np.load(COMP / "nodes/node_0134/oof_fold0.npy")


def lp(p):  # clipped log-probs, as the real combiner uses
    return np.log(np.clip(p, 1e-6, 1.0))


def stack_cv(blocks, y, seed=42, C=0.003):
    """Nested 5-fold LogReg over horizontally-stacked log-prob blocks → OOF BA."""
    X = np.hstack([lp(b) for b in blocks])
    oof = np.zeros((len(y), 3))
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])
    return balanced_accuracy_score(y, oof.argmax(1))


def load(node):
    return np.load(COMP / f"nodes/{node}/oof.npy")[idx]

n091 = load("node_0091")
bank = [load(n) for n in ["node_0070", "node_0033", "node_0063", "node_0040"]]

print(f"fold-0 rows: {len(y)}   (internal 5-fold meta, C=0.003, balanced LogReg)\n")
print(f"  Mitra solo BA            : {balanced_accuracy_score(y, mitra.argmax(1)):.6f}")
print(f"  champion n091 solo BA    : {balanced_accuracy_score(y, n091.argmax(1)):.6f}\n")

# average over a few seeds to damp split noise
def avg(blocks, seeds=(1, 2, 3, 4, 5)):
    return np.mean([stack_cv(blocks, y, seed=s) for s in seeds])

a0 = avg([n091]);              a1 = avg([n091, mitra])
b0 = avg(bank);                b1 = avg(bank + [mitra])
print("A) incremental over CHAMPION (n091 as super-base):")
print(f"   n091            : {a0:.6f}")
print(f"   n091 + Mitra    : {a1:.6f}    delta = {a1-a0:+.6f}")
print("\nB) small bank {n070,n033,n063,n040}:")
print(f"   bank            : {b0:.6f}")
print(f"   bank + Mitra    : {b1:.6f}    delta = {b1-b0:+.6f}")
print("\nverdict: clear +delta on BOTH → worth the full 5-fold; flat/neg → wall holds, kill.")
