"""Does the 0.97105 notebook's META-CONFIG help our existing bases?

Our node_0020 stack: log-prob features, LogReg(C=1, balanced), 5-fold, DE threshold.
Notebook (LB 0.97105): LOGIT features, LogReg(C=0.1, balanced+STAR-boost, multinomial),
10-fold × 5-seed, Nelder-Mead 3-multiplier threshold.

This isolates the META-CONFIG (not the bases — same OOFs both ways). Bases are the real
lever, but this tells us the best stacker settings to lock in for the final.
"""
from __future__ import annotations
import json, warnings, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import minimize, differential_evolution
warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parent
LAB = ["GALAXY", "QSO", "STAR"]; L2I = {l: i for i, l in enumerate(LAB)}; NC = 3
EPS = 1e-15; CLIP = 30.0

# our current champion base set + the strongest extras we've built
BASES = ["node_0006","node_0004","node_0001","node_0009","node_0011","node_0003",
         "node_0019","node_0016","node_0014","node_0023","node_0026"]

train = pd.read_csv(COMP/"data/train.csv")
y = train["class"].map(L2I).to_numpy(); n = len(y)

def logp(a): return np.log(np.clip(a, 1e-7, 1.0))
def logit(a):
    a = np.clip(a, EPS, 1-EPS).astype(np.float64)
    return np.clip(np.log(a/(1-a)), -CLIP, CLIP).astype(np.float32)

raw = [np.load(COMP/"nodes"/b/"oof.npy") for b in BASES]
OOF_logp  = np.concatenate([logp(r)  for r in raw], axis=1)
OOF_logit = np.concatenate([logit(r) for r in raw], axis=1)

def balacc(yy, pred):
    return float(np.mean([(pred[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))

def stacked_oof(feat, C, n_folds, seeds, star_boost):
    acc = np.zeros((n, NC))
    for seed in seeds:
        skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
        oof = np.zeros((n, NC))
        for tr, va in skf.split(feat, y):
            cw = dict(zip(*[np.unique(y[tr]), compute_class_weight("balanced", classes=np.unique(y[tr]), y=y[tr])]))
            cw[2] *= star_boost
            m = LogisticRegression(C=C, class_weight=cw, max_iter=1000, multi_class="multinomial", n_jobs=-1)
            m.fit(feat[tr], y[tr]); oof[va] = m.predict_proba(feat[va])
        acc += oof
    return acc/len(seeds)

def de_thr(P, yy):
    r = differential_evolution(lambda w: -balacc(yy, np.argmax(P*np.array([w[0],w[1],1.0]),1)),
                               [(0.1,5),(0.1,5)], maxiter=40, seed=0, polish=False)
    return np.array([r.x[0], r.x[1], 1.0])
def nm_thr(P, yy):
    r = minimize(lambda w: -balacc(yy, np.argmax(P*w,1)), [1.,1.,1.], method="Nelder-Mead",
                 options={"maxiter":1000})
    return r.x

def evaluate(name, feat, C, n_folds, seeds, star_boost, thr):
    so = stacked_oof(feat, C, n_folds, seeds, star_boost)
    base = balacc(y, np.argmax(so,1))
    w = (de_thr if thr=="de" else nm_thr)(so, y)
    tuned = balacc(y, np.argmax(so*w,1))
    print(f"{name:42s} base={base:.6f}  +thr({thr})={tuned:.6f}")
    return tuned

print(f"bases={len(BASES)}  features={OOF_logp.shape[1]}\n")
print("=== our config vs notebook config (same bases) ===")
evaluate("OURS: log-prob C=1 5f×1 DE",        OOF_logp,  1.0, 5,  [42],              1.0, "de")
evaluate("logit C=0.1 5f×1 DE",               OOF_logit, 0.1, 5,  [42],              1.0, "de")
evaluate("logit C=0.1 10f×1 NM",              OOF_logit, 0.1, 10, [42],              1.0, "nm")
evaluate("NOTEBOOK: logit C=0.1 10f×5 NM",    OOF_logit, 0.1, 10, [42,63,55555,37,47],1.0,"nm")
evaluate("NOTEBOOK+STARx1.03",                OOF_logit, 0.1, 10, [42,63,55555,37,47],1.03,"nm")
print("\nchamp node_0020 recorded cv 0.966627 (its own 9-base set).")
