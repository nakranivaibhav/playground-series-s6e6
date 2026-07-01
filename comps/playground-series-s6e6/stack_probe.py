"""Balanced multinomial LogReg STACKER + threshold calibration (DE vs GPU grid).

Recipe from the Kaggle discussion (Chris Deotte / Siddhesh Sathe variance note):
  - features = log(clip(p,1e-7,1)) of each base model's OOF, concatenated
  - meta = LogisticRegression(class_weight='balanced', multinomial)
  - then per-class threshold w: argmax(prob * w), w found by DE or a fine grid
  - ALL fold-honest: meta fit on the other 4 folds, applied to the held fold;
    threshold fit on the other folds' stacked OOF, scored on the held fold.

Compares: stack-only / stack+GPU-grid-threshold / stack+DE-threshold,
vs champion blend 0.965889 and node_0017 (prob-avg + grid threshold) 0.966084.
"""
from __future__ import annotations
import json, time, warnings
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAB = ["GALAXY", "QSO", "STAR"]; L2I = {l: i for i, l in enumerate(LAB)}
NC = 3

# base zoo for the stack — strong + diverse (stacking can use more than a prob-avg blend)
BASES = ["node_0006", "node_0004", "node_0001", "node_0009",   # champion arms
         "node_0011", "node_0003", "node_0019", "node_0016", "node_0014"]  # extra diversity

train = pd.read_csv(COMP / "data/train.csv")
folds = json.loads((COMP / "folds.json").read_text())["folds"]
n = len(train)
y = train["class"].map(L2I).to_numpy()
fval = [np.asarray(f["val_idx"]) for f in folds]

def logp(a): return np.log(np.clip(a, 1e-7, 1.0))
OOF = np.concatenate([logp(np.load(COMP / "nodes" / b / "oof.npy")) for b in BASES], axis=1)  # (N, 3*B)
print(f"stack features: {OOF.shape[1]} cols from {len(BASES)} bases  device={DEV}")

def balacc(yy, pred):
    return float(np.mean([(pred[yy == c] == c).mean() for c in range(NC) if (yy == c).any()]))

def fit_meta(Xtr, ytr):
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr); return m

# ---- step A: honest 5-fold stacked OOF (meta fit on other 4 folds) ---------
stack_oof = np.zeros((n, NC))
for vi in fval:
    tr = np.setdiff1d(np.arange(n), vi)
    m = fit_meta(OOF[tr], y[tr])
    stack_oof[vi] = m.predict_proba(OOF[vi])

# ---- threshold optimizers (on a given prob matrix / row subset) ------------
THR = torch.tensor([[g, q, 1.0]
                    for g in [round(0.4+0.02*k,2) for k in range(81)]   # 0.40..2.00 step .02
                    for q in [round(0.4+0.02*k,2) for k in range(81)]],
                   dtype=torch.float32, device=DEV)                      # (6561,3)
yT = torch.tensor(y, device=DEV)
probT = torch.tensor(stack_oof, dtype=torch.float32, device=DEV)

def balacc_T(pred, yy):
    a = [(pred[:, yy == c] == c).float().mean(1) for c in range(NC) if (yy == c).any()]
    return torch.stack(a, 0).mean(0)

def best_thr_gpu(rows, chunk=256):
    pr = probT[rows]; yr = yT[rows]
    sc = torch.empty(THR.shape[0], device=DEV)
    for s in range(0, THR.shape[0], chunk):
        preds = (pr.unsqueeze(0) * THR[s:s+chunk].unsqueeze(1)).argmax(-1)   # (c, Ns)
        sc[s:s+chunk] = balacc_T(preds, yr)
    return THR[int(sc.argmax())].cpu().numpy()

def best_thr_de(rows):
    P = stack_oof[rows]; yr = y[rows]
    def neg(w):
        return -balacc(yr, np.argmax(P * np.array([w[0], w[1], 1.0]), axis=1))
    r = differential_evolution(neg, [(0.1, 5.0), (0.1, 5.0)], maxiter=40, tol=1e-7,
                               seed=0, polish=False)
    return np.array([r.x[0], r.x[1], 1.0])

# ---- step B: fold-honest scoring of each variant --------------------------
def honest(score_fn):
    pf = []
    for vi in fval:
        other = np.setdiff1d(np.arange(n), vi)
        pf.append(score_fn(vi, other))
    return float(np.mean(pf)), float(np.std(pf, ddof=1) / np.sqrt(len(pf)))

t0 = time.time()
cv_plain, sem_plain = honest(lambda vi, oth: balacc(y[vi], np.argmax(stack_oof[vi], 1)))
cv_gpu, sem_gpu = honest(lambda vi, oth: balacc(y[vi], np.argmax(stack_oof[vi] * best_thr_gpu(oth), 1)))
t_gpu = time.time()
cv_de, sem_de = honest(lambda vi, oth: balacc(y[vi], np.argmax(stack_oof[vi] * best_thr_de(oth), 1)))
t_de = time.time()

print(f"\n{'variant':34s} {'honest cv':>10s} {'sem':>9s}")
print(f"{'champion prob-avg blend':34s} {0.965889:10.6f} {0.000141:9.6f}   (node_0010)")
print(f"{'node_0017 (prob-avg + grid thr)':34s} {0.966084:10.6f} {0.000177:9.6f}")
print(f"{'STACK only (balanced logreg)':34s} {cv_plain:10.6f} {sem_plain:9.6f}")
print(f"{'STACK + GPU-grid threshold':34s} {cv_gpu:10.6f} {sem_gpu:9.6f}   ({t_gpu-t0:.1f}s grid)")
print(f"{'STACK + DE threshold':34s} {cv_de:10.6f} {sem_de:9.6f}   ({t_de-t_gpu:.1f}s DE)")
print(f"\nGPU-grid vs DE threshold cv diff = {cv_gpu - cv_de:+.6f}")
