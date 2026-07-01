"""GPU blend search v2 — tests the new arms (n18 tgt-enc, n19 bagged-TabM) and
stacks a fold-honest per-class threshold tune on top of each blend.

Fully nested & fold-honest: for each held-out fold, the blend weights AND the
3-class threshold vector are both fit on the OTHER 4 folds, then scored on the
held fold. Reports plain-blend CV and threshold-tuned CV per config.
"""
from __future__ import annotations
import json, time
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd, torch

COMP = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAB = ["GALAXY", "QSO", "STAR"]; L2I = {l: i for i, l in enumerate(LAB)}
NC = 3; STEP = 0.05; CHUNK = 64

POOL = ["node_0006", "node_0004", "node_0001", "node_0009", "node_0018", "node_0019"]
train = pd.read_csv(COMP / "data/train.csv")
folds = json.loads((COMP / "folds.json").read_text())["folds"]
n = len(train)
y = torch.tensor(train["class"].map(L2I).to_numpy(), device=DEV)
allidx = torch.arange(n, device=DEV)
fval = [torch.tensor(np.asarray(f["val_idx"]), device=DEV) for f in folds]
OOF = {a: torch.tensor(np.load(COMP / "nodes" / a / "oof.npy"), dtype=torch.float32, device=DEV) for a in POOL}

# per-class threshold grid: fix STAR=1.0, vary GALAXY/QSO in [0.7,1.3] step .05
twv = [round(0.7 + .05 * k, 2) for k in range(13)]
THR = torch.tensor([[g, q, 1.0] for g in twv for q in twv], dtype=torch.float32, device=DEV)  # (T,3)


def simplex(k_arms, step):
    k = int(round(1 / step)); out = []
    for c in product(range(k + 1), repeat=k_arms - 1):
        if sum(c) <= k:
            out.append(tuple(x / k for x in c) + ((k - sum(c)) / k,))
    return out


def balacc(preds, yy):
    a = [(preds[:, yy == c] == c).float().mean(1) for c in range(NC) if (yy == c).any()]
    return torch.stack(a, 0).mean(0)


def best_blend_w(P, W, rows):
    Ps = P[:, rows, :]; out = torch.empty(W.shape[0], device=DEV)
    for s in range(0, W.shape[0], CHUNK):
        bl = torch.einsum("ca,and->cnd", W[s:s+CHUNK], Ps).argmax(-1)
        out[s:s+CHUNK] = balacc(bl, y[rows])
    return W[int(out.argmax())]


def blended_probs(P, w):
    return torch.einsum("a,and->nd", w, P)            # (N,3)


def best_thr(prob, rows):
    # prob (N,3); pick threshold maximizing balacc on rows
    pr = prob[rows]                                    # (Ns,3)
    preds = (pr.unsqueeze(0) * THR.unsqueeze(1)).argmax(-1)   # (T,Ns)
    sc = balacc(preds, y[rows])
    return THR[int(sc.argmax())]


def honest(arms, thr=False):
    P = torch.stack([OOF[a] for a in arms])            # (A,N,3)
    W = torch.tensor(simplex(len(arms), STEP), dtype=torch.float32, device=DEV)
    pf = []
    for vi in fval:
        m = torch.ones(n, dtype=torch.bool, device=DEV); m[vi] = False
        other = allidx[m]
        w = best_blend_w(P, W, other)
        prob = blended_probs(P, w)
        if thr:
            t = best_thr(prob, other)
            pred = (prob[vi] * t).argmax(-1)
        else:
            pred = prob[vi].argmax(-1)
        pf.append(float(balacc(pred[None, :], y[vi])[0].item()))
    cv = float(np.mean(pf)); sem = float(np.std(pf, ddof=1) / np.sqrt(len(pf)))
    return cv, sem


CFG = {
    "champion n6/n4/n1/n9":      ["node_0006","node_0004","node_0001","node_0009"],
    "swap n9->n19 (bagTabM)":    ["node_0006","node_0004","node_0001","node_0019"],
    "+n19 (5-arm)":              ["node_0006","node_0004","node_0001","node_0009","node_0019"],
    "+n18 (tgt-enc)":            ["node_0006","node_0004","node_0001","node_0009","node_0018"],
    "swap n9->n19 +n18":         ["node_0006","node_0004","node_0001","node_0019","node_0018"],
    "all 6":                     POOL,
}

print(f"device={DEV}  threshold grid {THR.shape[0]} pts")
print(f"{'config':26s} {'blend cv':>10s} {'+thresh cv':>11s} {'Δthr vs champ':>13s}  sec", flush=True)
champ = None
for name, arms in CFG.items():
    t0 = time.time()
    cv, _ = honest(arms, thr=False)
    cvt, semt = honest(arms, thr=True)
    if DEV == "cuda": torch.cuda.synchronize()
    if champ is None: champ = cv      # champion plain-blend baseline
    print(f"{name:26s} {cv:10.6f} {cvt:11.6f} {cvt-champ:+13.6f}  {time.time()-t0:.1f}", flush=True)
print(f"\n(baseline champion plain-blend cv = {champ:.6f}; node_0010 official = 0.965889)")
