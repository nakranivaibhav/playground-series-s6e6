"""Re-fit the node_0020 balanced-LogReg STACK with the new base columns.

Champion (node_0020) stack = 9 bases + DE threshold, cv 0.966627.
New bases this round: node_0021 RealMLP, node_0022 TabPFN-3, node_0023 CatBoost-retune.
Question: does adding any/all of them lift the honest stacked-OOF balanced accuracy?

All fold-honest: meta (balanced multinomial LogReg on log-prob features) fit on the
other 4 folds, applied to the held fold; DE per-class threshold fit on the other
folds' stacked OOF, scored on the held fold. Same recipe as node_0020.
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

CHAMP9 = ["node_0006", "node_0004", "node_0001", "node_0009",
          "node_0011", "node_0003", "node_0019", "node_0016", "node_0014"]
NEW = {"rmlp1": "node_0028",   # 0.96907 RealMLP seed-1 (breakthrough)
       "rmlp2": "node_0032",   # 0.96912 RealMLP seed-2
       "rmlp3": "node_0035",   # 0.96897 RealMLP seed-3
       "tabm":  "node_0033",   # 0.96805 TabM on rich FE (strong de-correlated NN)
       "lgbm":  "node_0030",   # 0.96695 LightGBM on rich FE
       "xgb":   "node_0031",   # 0.96624 XGBoost on rich FE
       "tabicl":"node_0026"}   # 0.9590 TabICL

train = pd.read_csv(COMP / "data/train.csv")
folds = json.loads((COMP / "folds.json").read_text())["folds"]
n = len(train); y = train["class"].map(L2I).to_numpy()
fval = [np.asarray(f["val_idx"]) for f in folds]

def logp(a): return np.log(np.clip(a, 1e-7, 1.0))
def load(bases): return np.concatenate([logp(np.load(COMP/"nodes"/b/"oof.npy")) for b in bases], axis=1)

def balacc(yy, pred):
    return float(np.mean([(pred[yy == c] == c).mean() for c in range(NC) if (yy == c).any()]))

def fit_meta(X, yy):
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(X, yy); return m

def de_thr(P, yy):
    def neg(w): return -balacc(yy, np.argmax(P*np.array([w[0], w[1], 1.0]), axis=1))
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)], maxiter=40, tol=1e-7, seed=0, polish=False)
    return np.array([r.x[0], r.x[1], 1.0])

def eval_set(bases):
    OOF = load(bases)
    stack = np.zeros((n, NC))
    for vi in fval:
        tr = np.setdiff1d(np.arange(n), vi)
        stack[vi] = fit_meta(OOF[tr], y[tr]).predict_proba(OOF[vi])
    # honest DE threshold (fit on other folds' stacked oof, scored on held fold)
    pf = []
    for vi in fval:
        oth = np.setdiff1d(np.arange(n), vi)
        w = de_thr(stack[oth], y[oth])
        pf.append(balacc(y[vi], np.argmax(stack[vi]*w, axis=1)))
    return float(np.mean(pf)), float(np.std(pf, ddof=1)/np.sqrt(len(pf)))

CH = CHAMP9
N = NEW
RMLP3 = [N["rmlp1"], N["rmlp2"], N["rmlp3"]]
CHAMP15 = CH + RMLP3 + [N["tabm"], N["lgbm"], "node_0039"]   # champion node_0041 (0.969808)
VARIANTS = {
    "champion node_0041 (15 bases)":     CHAMP15,
    "+realmlp_B(n42)":                   CHAMP15 + ["node_0042"],
    "+catboost_B(n43)":                  CHAMP15 + ["node_0043"],
    "+catB +realmlpB":                   CHAMP15 + ["node_0043", "node_0042"],
}

print(f"{'base set':28s} {'honest cv':>11s} {'sem':>9s} {'Δ vs champ':>11s}")
base_cv = None
for name, bases in VARIANTS.items():
    cv, sem = eval_set(bases)
    if base_cv is None: base_cv = cv
    d = cv - base_cv
    flag = "  <-- champ" if name.startswith("champ") else (f"  +{d:.6f}" if d > 0 else f"  {d:.6f}")
    print(f"{name:28s} {cv:11.6f} {sem:9.6f} {d:+11.6f}{flag}")
print("\nchamp = node_0020 recorded cv 0.966627; promote a variant only if Δ > ~2·sem (~0.00045).")
