"""Exploratory blend probe — fold-honest nested weight search (same protocol as
node_0010) over several arm sets, to see if any new arm (n14 FT-Transformer,
n15 DART, n16 TabM-weighted) lifts the champion 4-arm blend (0.965889).
Coarser 0.1 grid for speed; no shuffled control (removed from system)."""
from __future__ import annotations
import json, sys
from itertools import product
from pathlib import Path
import numpy as np, pandas as pd

COMP_DIR = Path(__file__).resolve().parent
LABEL_ORDER = ["GALAXY", "QSO", "STAR"]
LABEL2IDX = {l: i for i, l in enumerate(LABEL_ORDER)}
N_CLASSES = 3
GRID_STEP = 0.10

ALL_ARMS = ["node_0006", "node_0004", "node_0001", "node_0009",
            "node_0014", "node_0015", "node_0016"]

train = pd.read_csv(COMP_DIR / "data/train.csv")
folds_list = json.loads((COMP_DIR / "folds.json").read_text())["folds"]
n = len(train)
y_int = train["class"].map(LABEL2IDX).to_numpy()
all_idx = np.arange(n)
fold_val = [np.asarray(fi["val_idx"]) for fi in folds_list]

OOF = {a: np.load(COMP_DIR / "nodes" / a / "oof.npy") for a in ALL_ARMS}
for a, P in OOF.items():
    assert P.shape == (n, N_CLASSES), f"{a} {P.shape}"


def fast_balacc(y, pred):
    return float(np.mean([(pred[y == c] == c).mean() for c in range(N_CLASSES) if (y == c).any()]))


def simplex_weights(k_arms, step):
    k = int(round(1.0 / step))
    out = []
    for combo in product(range(k + 1), repeat=k_arms - 1):
        if sum(combo) <= k:
            out.append(tuple(c / k for c in combo) + ((k - sum(combo)) / k,))
    return out


def blend_pred(P_list, w, rows):
    acc = None
    for wi, P in zip(w, P_list):
        if wi == 0.0:
            continue
        acc = wi * P[rows] if acc is None else acc + wi * P[rows]
    if acc is None:
        acc = sum(P[rows] for P in P_list)
    return np.argmax(acc, axis=1)


def best_weights(P_list, y, rows, cands):
    bs, bw = -1.0, None
    for w in cands:
        s = fast_balacc(y[rows], blend_pred(P_list, w, rows))
        if s > bs + 1e-12:
            bs, bw = s, w
    return bw, bs


def honest_cv(arms):
    P_list = [OOF[a] for a in arms]
    cands = simplex_weights(len(arms), GRID_STEP)
    per_fold = []
    for val_idx in fold_val:
        other = np.setdiff1d(all_idx, val_idx)
        w_f, _ = best_weights(P_list, y_int, other, cands)
        per_fold.append(fast_balacc(y_int[val_idx], blend_pred(P_list, w_f, val_idx)))
    w_full, full_s = best_weights(P_list, y_int, all_idx, cands)
    cv = float(np.mean(per_fold))
    sem = float(np.std(per_fold, ddof=1) / np.sqrt(len(per_fold)))
    return cv, sem, w_full, full_s


CONFIGS = {
    "champion n6/n4/n1/n9":        ["node_0006", "node_0004", "node_0001", "node_0009"],
    "+n14 (FT-T)":                 ["node_0006", "node_0004", "node_0001", "node_0009", "node_0014"],
    "+n15 (DART)":                 ["node_0006", "node_0004", "node_0001", "node_0009", "node_0015"],
    "+n16 (TabM-wt)":              ["node_0006", "node_0004", "node_0001", "node_0009", "node_0016"],
    "swap n9->n16":                ["node_0006", "node_0004", "node_0001", "node_0016"],
}

print(f"{'config':28s}  {'honest cv':>10s}  {'sem':>8s}  {'Δ vs champ':>10s}  weights", flush=True)
champ = None
for name, arms in CONFIGS.items():
    cv, sem, w, _ = honest_cv(arms)
    if champ is None:
        champ = cv
    wtxt = ", ".join(f"{a.split('_')[1]}:{wi:.1f}" for a, wi in zip(arms, w) if wi > 0)
    print(f"{name:28s}  {cv:10.6f}  {sem:8.6f}  {cv-champ:+10.6f}  {wtxt}", flush=True)
