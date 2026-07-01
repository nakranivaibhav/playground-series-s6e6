"""PROBE (not a node): does a fold-honest per-class weight/threshold lift the champion
node_0091's balanced accuracy over plain argmax? Optimizes 3 per-class multipliers via
differential evolution on the OOF rows of the OTHER folds, evaluates on the held-out fold.
Cheap diagnostic on existing OOF — escalate to a registered node ONLY if it shows a real lift."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import differential_evolution

COMP = Path(__file__).resolve().parents[1]
L2I = {"GALAXY": 0, "QSO": 1, "STAR": 2}

oof = np.load(COMP / "nodes/node_0091/oof.npy").astype(np.float64)  # (n,3) stacked probs
y = pd.read_csv(COMP / "data/train.csv")["class"].map(L2I).to_numpy()
folds = json.loads((COMP / "folds.json").read_text())

vis = [np.asarray(f["val_idx"], dtype=int) for f in folds["folds"]]
print(f"folds: {len(vis)}  sizes={[len(v) for v in vis]}  total={sum(len(v) for v in vis)} (n={len(y)})")

def ba_w(probs, yy, w):
    return balanced_accuracy_score(yy, (probs * w).argmax(1))

# plain-argmax baseline (should reproduce ~0.970355)
plain = np.mean([balanced_accuracy_score(y[vi], oof[vi].argmax(1)) for vi in vis])
print(f"plain argmax CV = {plain:.6f}")

# fold-honest per-class weight optimization
thr_scores, ws = [], []
for k, vi in enumerate(vis):
    other = np.concatenate([vis[j] for j in range(len(vis)) if j != k])
    # maximize BA on `other` (the 4 training folds) over 3 per-class weights in [0.3, 3.0]
    res = differential_evolution(
        lambda w: -ba_w(oof[other], y[other], w),
        bounds=[(0.3, 3.0)] * 3, seed=0, maxiter=60, tol=1e-7, polish=True, updating="deferred",
    )
    w = res.x / res.x.sum() * 3  # normalize scale (argmax is scale-invariant per-row anyway)
    held = ba_w(oof[vi], y[vi], w)
    base = balanced_accuracy_score(y[vi], oof[vi].argmax(1))
    thr_scores.append(held); ws.append(np.round(w, 3))
    print(f"  fold {k}: argmax={base:.6f}  weighted={held:.6f}  delta={held-base:+.6f}  w={np.round(w,3)}")

thr_cv = float(np.mean(thr_scores))
print(f"\nfold-honest weighted CV = {thr_cv:.6f}   delta vs argmax = {thr_cv - plain:+.6f}")
print(f"champion baseline 0.970355 · 2·sem ≈ 0.000498 (promote bar)")
print("VERDICT:", "LIFTS (escalate to node + LB probe)" if thr_cv - plain > 0.0002
      else "WASH/HURT (threshold lever closed on n091; one journal line)")
