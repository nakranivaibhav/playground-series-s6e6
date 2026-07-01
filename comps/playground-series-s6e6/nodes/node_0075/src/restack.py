"""node_0075 restack: bank17+bag and bank17+bag+n67.

Runs after oof.npy and test_probs.npy are produced by solution.py.
Requires pub_oof.npy and pub_test.npy in /tmp (produced by a1_full_merge.py).

Outputs printed to stdout; no files written (probes only).
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")

C = Path("comps/playground-series-s6e6")
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
NC = 3

tr = pd.read_csv(C / "data/train.csv")
te = pd.read_csv(C / "data/test.csv")
n = len(tr); nt = len(te)
y = tr["class"].map(L2I).to_numpy()
folds = json.loads((C / "folds.json").read_text())["folds"]
fval = [np.asarray(f["val_idx"]) for f in folds]

def logp(a): return np.log(np.clip(a, 1e-7, 1.0))

# Load bank17 OOFs
pub_oof_d = np.load("/tmp/pub_oof.npy", allow_pickle=True).item()
pub_test_d = np.load("/tmp/pub_test.npy", allow_pickle=True).item()
good = json.loads(open("/tmp/pub_good.json").read())
pub_oof = [logp(pub_oof_d[k]) for k in good]
pub_test = [logp(pub_test_d[k]) for k in good]

# Load node_0075 bag OOF/test
n75_oof = logp(np.load(C / "nodes/node_0075/oof.npy"))
n75_test = logp(np.load(C / "nodes/node_0075/test_probs.npy"))

# Load node_0067 OOF/test
n67_oof = logp(np.load(C / "nodes/node_0067/oof.npy"))
n67_test = logp(np.load(C / "nodes/node_0067/test_probs.npy"))

def balacc(yy, pred): return float(np.mean([(pred[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))

def de_thr(P, yy):
    f = lambda w: -balacc(yy, np.argmax(P * np.array([w[0], w[1], 1.0]), 1))
    r = differential_evolution(f, [(0.1, 5.0), (0.1, 5.0)], maxiter=40, tol=1e-7, seed=0, polish=False)
    return np.array([r.x[0], r.x[1], 1.0])

def eval_cv(cols):
    OOF = np.concatenate(cols, 1)
    stack = np.zeros((n, NC))
    for vi in fval:
        trr = np.setdiff1d(np.arange(n), vi)
        stack[vi] = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1).fit(OOF[trr], y[trr]).predict_proba(OOF[vi])
    pf = []
    for vi in fval:
        oth = np.setdiff1d(np.arange(n), vi)
        w = de_thr(stack[oth], y[oth])
        pf.append(balacc(y[vi], np.argmax(stack[vi] * w, 1)))
    return float(np.mean(pf)), float(np.std(pf, ddof=1) / np.sqrt(len(pf)))

print("Running restack probes...")
configs = {
    "bank17": pub_oof,
    "bank17+n75bag": pub_oof + [n75_oof],
    "bank17+n75bag+n67": pub_oof + [n75_oof, n67_oof],
}
for name, cols in configs.items():
    cv, sem = eval_cv(cols)
    print(f"{name:24s}  cv={cv:.6f}  sem={sem:.6f}")
