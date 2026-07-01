"""Does a balanced-LogReg meta beat simple averaging for the single RealMLP?

KEY POINT: the N fold-models are the SAME architecture on ~97%-overlapping data,
so they're ~0.99 correlated. A LogReg over N near-identical columns just recovers
equal weights = averaging — no gain there. The ONLY non-degenerate version is a
1-base balanced-LogReg *calibration*: map the RealMLP's 3 softmax probs -> a
learned, class-balanced decision boundary (exactly what n091's meta does per base).
That can help BALANCED ACCURACY by re-weighting toward the rare classes, or wash.

This tests it FAST on the existing 5-fold OOF (instant) before committing 45 min to
re-running 30-fold: fit a balanced multinomial LogReg on the OOF log-probs, evaluated
fold-honestly (outer CV), vs the raw argmax. If it helps here, it'll help at 30-fold.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
oof = np.load(COMP / "nodes/node_0140/oof.npy").astype(float)      # 5-fold RealMLP OOF (577347 x 3)
y = pd.read_csv(COMP / "data/train.csv")["class"].map({"GALAXY":0,"QSO":1,"STAR":2}).values
logp = np.log(np.clip(oof, 1e-7, 1.0))

raw_ba = balanced_accuracy_score(y, oof.argmax(1))
print(f"raw RealMLP argmax (5-fold OOF) BA = {raw_ba:.6f}")

# balanced LogReg meta, evaluated fold-honestly via an outer 5-fold over the OOF rows
for C in (0.1, 1.0, 10.0):
    meta = np.zeros_like(oof)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(logp, y):
        m = LogisticRegression(class_weight="balanced", C=C, max_iter=2000,
                               solver="lbfgs", multi_class="multinomial", n_jobs=-1)
        m.fit(logp[tr], y[tr])
        meta[va] = m.predict_proba(logp[va])
    ba = balanced_accuracy_score(y, meta.argmax(1))
    print(f"balanced-LogReg meta (C={C:>4}) BA = {ba:.6f}   delta vs raw = {ba-raw_ba:+.6f}")

# also: simple per-class threshold via the balanced LogReg fit on ALL oof (upper-bound-ish)
print("\n(If delta is ~0 or negative, a LogReg meta on the single model is a wash -> "
      "averaging is already optimal; the 30-fold avg submission stands.)")
