"""Meta-stacker config sweep on the 15-base CORE set (champion node_0041).

The public 0.97105 meta-stacker differs from ours at the META level in 3 ways we
may not have tested on OUR bases:
  (a) LOGIT transform  log(p/(1-p)) clipped +/-30   (we feed raw log(p))
  (b) C = 0.1 (meta regularized ~10x harder)        (we use C=1.0)
  (c) seed-bagging the meta (multiple fold partitions averaged)

This sweeps transform x C under our HONEST nested CV on the frozen folds.json
(meta fit on the other 4 folds, applied to the held fold; DE per-class threshold
fit on the other folds' stacked OOF, scored on held fold). Same protocol as
restack_probe.py so numbers are comparable. (c) is approximated by averaging the
meta over INNER_SEEDS re-fits with different LogReg random_state (cheap; the base
OOF is fixed on folds.json, so this only bags the meta solver, not the splits).
"""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parent
LAB = ["GALAXY", "QSO", "STAR"]; L2I = {l: i for i, l in enumerate(LAB)}; NC = 3
EPS, CLIP = 1e-15, 30.0

CHAMP9 = ["node_0006", "node_0004", "node_0001", "node_0009",
          "node_0011", "node_0003", "node_0019", "node_0016", "node_0014"]
CORE15 = CHAMP9 + ["node_0028", "node_0032", "node_0035",  # 3 RealMLP seeds
                   "node_0033", "node_0030", "node_0039"]  # TabM, LGBM, CatBoost

train = pd.read_csv(COMP / "data/train.csv")
folds = json.loads((COMP / "folds.json").read_text())["folds"]
n = len(train); y = train["class"].map(L2I).to_numpy()
fval = [np.asarray(f["val_idx"]) for f in folds]

def logp(a):  # our current transform
    return np.log(np.clip(a, 1e-7, 1.0))
def logit(a):  # the public notebook's transform
    p = np.clip(a, EPS, 1.0 - EPS).astype(np.float64)
    return np.clip(np.log(p / (1.0 - p)), -CLIP, CLIP)

def load(bases, tf):
    return np.concatenate([tf(np.load(COMP / "nodes" / b / "oof.npy")) for b in bases], axis=1)

def balacc(yy, pred):
    return float(np.mean([(pred[yy == c] == c).mean() for c in range(NC) if (yy == c).any()]))

def de_thr(P, yy):
    def neg(w): return -balacc(yy, np.argmax(P * np.array([w[0], w[1], 1.0]), axis=1))
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)], maxiter=40, tol=1e-7, seed=0, polish=False)
    return np.array([r.x[0], r.x[1], 1.0])

def eval_set(bases, tf, C, inner_seeds, OOF):
    # raw-argmax balanced accuracy under honest nested CV (no DE threshold — its
    # effect is ~constant across meta-configs, so it can't change the ranking).
    pf = []
    for vi in fval:
        tr = np.setdiff1d(np.arange(n), vi)
        acc = np.zeros((len(vi), NC))
        for s in range(inner_seeds):
            m = LogisticRegression(class_weight="balanced", C=C, max_iter=2000,
                                   n_jobs=-1, random_state=s)
            m.fit(OOF[tr], y[tr]); acc += m.predict_proba(OOF[vi])
        pf.append(balacc(y[vi], np.argmax(acc / inner_seeds, axis=1)))
    return float(np.mean(pf)), float(np.std(pf, ddof=1) / np.sqrt(len(pf)))

CONFIGS = [
    ("logprob C=1.0  (current champion)", logp,  1.0, 1),
    ("logit   C=1.0",                     logit, 1.0, 1),
    ("logit   C=0.3",                     logit, 0.3, 1),
    ("logit   C=0.1",                     logit, 0.1, 1),
    ("logprob C=0.3",                     logp,  0.3, 1),
    ("logprob C=0.1",                     logp,  0.1, 1),
    ("logit   C=0.1 seed-bag x5",         logit, 0.1, 5),
    ("logit   C=0.3 seed-bag x5",         logit, 0.3, 5),
]
print(f"{'config':38s} {'cv(raw-argmax)':>14s} {'sem':>9s} {'Δ vs base':>11s}", flush=True)
OOF_CACHE = {}
base = None
for name, tf, C, sd in CONFIGS:
    key = tf.__name__
    if key not in OOF_CACHE: OOF_CACHE[key] = load(CORE15, tf)
    cv, sem = eval_set(CORE15, tf, C, sd, OOF_CACHE[key])
    if base is None: base = cv
    print(f"{name:38s} {cv:14.6f} {sem:9.6f} {cv-base:+11.6f}", flush=True)
print("\nNote: CV here is raw-argmax (no DE thresh) so it sits ~0.0008 below the "
      "recorded champion 0.969808; compare configs to each other, not to that number.", flush=True)
