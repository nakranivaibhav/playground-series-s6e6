"""GPU-vectorized blend search — same fold-honest nested protocol as blend_probe.py,
but the whole weight grid is evaluated as batched CUDA tensor ops instead of a Python
for-loop over candidates.

The idea: the blend is just  argmax_k ( sum_a w_a * P[a] )  scored by balanced accuracy.
That's pure linear algebra, so we stack ALL ~1000 candidate weight-vectors into one
tensor and let the GPU score them in a handful of batched ops. No per-candidate Python.

Correctness target: must reproduce blend_probe.py (CPU) to the 6th decimal.
"""
from __future__ import annotations
import json, time
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd, torch

COMP_DIR = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {l: i for i, l in enumerate(LABEL_ORDER)}
N_CLASSES = 3
GRID_STEP = 0.10
CAND_CHUNK = 64          # candidates scored per GPU batch (caps peak VRAM)

ALL_ARMS = ["node_0006", "node_0004", "node_0001", "node_0009",
            "node_0014", "node_0015", "node_0016"]

# ---- load everything once, onto the GPU -----------------------------------
train = pd.read_csv(COMP_DIR / "data/train.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
n = len(train)
y_full = torch.tensor(train["class"].map(LABEL2IDX).to_numpy(), device=DEV)   # (N,)
all_idx = torch.arange(n, device=DEV)
fold_val = [torch.tensor(np.asarray(fi["val_idx"]), device=DEV) for fi in folds_list]

# OOF stack: (A_all, N, 3) float32 on GPU — load once, index arms/rows as views.
OOF = torch.stack([
    torch.tensor(np.load(COMP_DIR / "nodes" / a / "oof.npy"), dtype=torch.float32)
    for a in ALL_ARMS
]).to(DEV)
ARM_IDX = {a: i for i, a in enumerate(ALL_ARMS)}
print(f"device={DEV}  OOF stack {tuple(OOF.shape)}  ({OOF.element_size()*OOF.nelement()/1e9:.2f} GB)")


def simplex_weights(k_arms, step):
    k = int(round(1.0 / step))
    out = []
    for combo in product(range(k + 1), repeat=k_arms - 1):
        if sum(combo) <= k:
            out.append(tuple(c / k for c in combo) + ((k - sum(combo)) / k,))
    return out


def balacc_batch(preds, y):
    """preds (C, Ns) int, y (Ns,) int -> (C,) balanced accuracy, fully vectorized."""
    accs = []
    for c in range(N_CLASSES):
        m = (y == c)
        if m.any():
            accs.append((preds[:, m] == c).float().mean(dim=1))   # (C,)
    return torch.stack(accs, dim=0).mean(dim=0)                    # (C,)


def score_candidates(P, W, y, rows):
    """P (A,N,3) gpu, W (C,A) gpu, y (N,) gpu, rows (Ns,) gpu -> scores (C,) gpu.
    Chunk over candidates so we never materialize the full (C,Ns,3) blend at once."""
    Psub = P[:, rows, :]                       # (A, Ns, 3)  — view-ish gather
    ysub = y[rows]                             # (Ns,)
    out = torch.empty(W.shape[0], device=DEV)
    for s in range(0, W.shape[0], CAND_CHUNK):
        Wc = W[s:s + CAND_CHUNK]               # (c, A)
        # blended[c, n, d] = sum_a Wc[c,a] * Psub[a, n, d]
        blended = torch.einsum("ca,and->cnd", Wc, Psub)   # (c, Ns, 3)
        preds = blended.argmax(dim=-1)                     # (c, Ns)
        out[s:s + CAND_CHUNK] = balacc_batch(preds, ysub)
    return out


def best_weights(P, W, y, rows):
    scores = score_candidates(P, W, y, rows)
    i = int(torch.argmax(scores).item())
    return W[i], float(scores[i].item())


def honest_cv(arms):
    P = OOF[[ARM_IDX[a] for a in arms]]                    # (A, N, 3) gpu view
    W = torch.tensor(simplex_weights(len(arms), GRID_STEP), dtype=torch.float32, device=DEV)
    per_fold = []
    for val_idx in fold_val:
        mask = torch.ones(n, dtype=torch.bool, device=DEV); mask[val_idx] = False
        other = all_idx[mask]
        w_f, _ = best_weights(P, W, y_full, other)
        s = balacc_batch(
            torch.einsum("a,and->nd", w_f, P[:, val_idx, :]).argmax(-1)[None, :],
            y_full[val_idx],
        )[0]
        per_fold.append(float(s.item()))
    w_full, _ = best_weights(P, W, y_full, all_idx)
    cv = float(np.mean(per_fold)); sem = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
    return cv, sem, w_full


CONFIGS = {
    "champion n6/n4/n1/n9":  ["node_0006", "node_0004", "node_0001", "node_0009"],
    "+n14 (FT-T)":           ["node_0006", "node_0004", "node_0001", "node_0009", "node_0014"],
    "+n15 (DART)":           ["node_0006", "node_0004", "node_0001", "node_0009", "node_0015"],
    "+n16 (TabM-wt)":        ["node_0006", "node_0004", "node_0001", "node_0009", "node_0016"],
    "all 7":                 ALL_ARMS,            # the config that TIMED OUT on CPU
}

print(f"{'config':24s}  {'honest cv':>10s}  {'sem':>8s}  {'Δ':>9s}  {'sec':>6s}  weights")
champ = None
for name, arms in CONFIGS.items():
    t0 = time.time()
    cv, sem, w = honest_cv(arms)
    torch.cuda.synchronize() if DEV == "cuda" else None
    dt = time.time() - t0
    if champ is None: champ = cv
    wtxt = ", ".join(f"{a.split('_')[1]}:{wi:.1f}" for a, wi in zip(arms, w.tolist()) if wi > 0)
    print(f"{name:24s}  {cv:10.6f}  {sem:8.6f}  {cv-champ:+9.6f}  {dt:6.2f}  {wtxt}", flush=True)
