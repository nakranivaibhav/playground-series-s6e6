#!/usr/bin/env python3
"""ensemble_of_ensembles.py — does stacking our STACKS beat the champion?

Tests the "ensemble of ensembles" idea directly on existing OOF: take our top
stack nodes (each already a meta over the 63-base bank) and combine THEM two ways
— (a) simple probability average, (b) a fresh nested-fold LogReg meta over their
clipped log-probs (the same combiner form as the champion). Compare to champion
n091 via balanced accuracy + a paired bootstrap. Read-only, no training.

Prior: all these stacks draw from the SAME bank on the SAME frozen folds, so they
are near-copies (n091 ⊇ n070's bank). Expect a wash — but measure it.

Run from repo root:
  uv run --no-sync python comps/playground-series-s6e6/probes/ensemble_of_ensembles.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

COMP = Path(__file__).resolve().parent.parent
CHAMP = "node_0091"
# our strongest stack/ensemble nodes (each is itself a meta over the bank):
STACKS = ["node_0091", "node_0070", "node_0116", "node_0063", "node_0041", "node_0040"]


def load(nid):
    p = COMP / "nodes" / nid / "oof.npy"
    return np.load(p).astype(np.float64) if p.exists() else None


def main():
    y = pd.read_csv(COMP / "data" / "train.csv", usecols=["class"])["class"].astype(str)
    classes = sorted(y.unique()); y = y.map({c: i for i, c in enumerate(classes)}).to_numpy()
    folds = json.load(open(COMP / "folds.json"))["folds"]
    foldid = np.empty(len(y), int)
    for f in folds:
        foldid[np.asarray(f["val_idx"], int)] = f["fold"]

    champ = load(CHAMP)
    ba_champ = balanced_accuracy_score(y, champ.argmax(1))
    print(f"champion {CHAMP} BA = {ba_champ:.6f}\n")

    oofs = {n: load(n) for n in STACKS if load(n) is not None}
    print(f"ensembles combined ({len(oofs)}): {list(oofs)}")
    for n, o in oofs.items():
        print(f"   {n}: solo BA {balanced_accuracy_score(y, o.argmax(1)):.6f}")

    # (a) simple probability average of the stacks
    avg = np.mean([o for o in oofs.values()], axis=0)
    ba_avg = balanced_accuracy_score(y, avg.argmax(1))

    # (b) nested-fold LogReg meta over the stacks' clipped log-probs (champion's form)
    X = np.hstack([np.log(np.clip(o, 1e-6, 1)) for o in oofs.values()])
    oof_meta = np.zeros((len(y), len(classes)))
    for f in range(len(folds)):
        tr, va = foldid != f, foldid == f
        clf = LogisticRegression(C=0.003, max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        oof_meta[va] = clf.predict_proba(X[va])
    ba_meta = balanced_accuracy_score(y, oof_meta.argmax(1))

    # paired bootstrap of the better combiner vs champion
    best_name, best = ("avg", avg.argmax(1)) if ba_avg >= ba_meta else ("logreg-meta", oof_meta.argmax(1))
    cp = champ.argmax(1)
    rng = np.random.default_rng(42); n = len(y); B = 3000
    wins = 0
    for _ in range(B):
        idx = rng.integers(0, n, n); yk = y[idx]
        if balanced_accuracy_score(yk, best[idx]) > balanced_accuracy_score(yk, cp[idx]):
            wins += 1

    print(f"\n--- ENSEMBLE-OF-ENSEMBLES results ---")
    print(f"  (a) simple prob-average of {len(oofs)} stacks : BA {ba_avg:.6f}   ({ba_avg-ba_champ:+.6f} vs champion)")
    print(f"  (b) LogReg meta over the stacks            : BA {ba_meta:.6f}   ({ba_meta-ba_champ:+.6f} vs champion)")
    print(f"  best = {best_name}; paired bootstrap P(better than champion) = {wins/B:.3f}  (promote needs >=0.90)")
    print(f"\n  VERDICT: {'BEATS champion — investigate!' if wins/B>=0.90 and max(ba_avg,ba_meta)>ba_champ else 'WASH — stacking our own stacks adds nothing (they share the same bank/folds).'}")


if __name__ == "__main__":
    main()
