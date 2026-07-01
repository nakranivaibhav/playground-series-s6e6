#!/usr/bin/env python3
"""finals_robustness.py — pick the 2 finals submissions on evidence, not vibes.

Kaggle scores your PRIVATE result as the BETTER of your 2 selected submissions on
the private slice. So a second pick has value only as a VARIANCE HEDGE: it helps on
the draws where it beats the champion. The right finals pair maximizes
E[max(BA_champion, BA_hedge)] under private-draw variance, NOT each sub's mean.

We have no private labels, so we estimate draw variance by resampling private-test-
sized slices (n_test rows, with replacement) from the OOF predictions on train, and
recomputing each candidate's Balanced Accuracy per slice. For every candidate hedge X
we report E[max(n091, X)] and the hedge gain over submitting the champion alone.

Read-only, no training. Run from repo root:
  uv run --no-sync python comps/playground-series-s6e6/probes/finals_robustness.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

COMP = Path(__file__).resolve().parent.parent
CHAMP = "node_0091"
# all finals-grade candidates (any could be slot-1 OR slot-2). n129 = highest CV.
CANDIDATES = ["node_0091", "node_0129", "node_0116", "node_0104", "node_0084",
              "node_0070", "node_0117"]
# known public LB for sanity (None = not probed):
LB = {"node_0091": 0.97121, "node_0129": 0.97118, "node_0117": 0.97003}
N_BOOT = 2000
SEED = 42


def load_pred(nid):
    p = COMP / "nodes" / nid / "oof.npy"
    return np.load(p).argmax(1) if p.exists() else None


def main():
    y = pd.read_csv(COMP / "data" / "train.csv", usecols=["class"])["class"].astype(str)
    classes = sorted(y.unique())
    y = y.map({c: i for i, c in enumerate(classes)}).to_numpy()
    n = len(y)
    n_test = json.load(open(COMP / "spec.md".replace("spec.md", "folds.json")))["n_rows"]  # fallback
    # private test size from spec
    try:
        spec = (COMP / "spec.md").read_text()
        import re
        n_test = int(re.search(r"n_test_rows:\s*(\d+)", spec).group(1))
    except Exception:
        n_test = 247435

    cands = {c: load_pred(c) for c in CANDIDATES if load_pred(c) is not None}
    print(f"candidates ({len(cands)}): {list(cands)}")
    print(f"resample slice size = n_test = {n_test:,}  | B = {N_BOOT}  | train rows = {n:,}\n")

    rng = np.random.default_rng(SEED)
    ba = {k: np.empty(N_BOOT) for k in cands}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n_test)
        yk = y[idx]
        for k, p in cands.items():
            ba[k][b] = balanced_accuracy_score(yk, p[idx])

    # solo expected private BA (the slot-1 ranking) + known LB
    print("  SOLO (slot-1 ranking by expected private BA):")
    solo = sorted(cands, key=lambda k: ba[k].mean(), reverse=True)
    for k in solo:
        lb = LB.get(k); lbs = f"LB {lb}" if lb else "LB n/a"
        print(f"    {k:<12}  E[BA]={ba[k].mean():.6f}  sd={ba[k].std():.6f}  {lbs}")

    # ALL PAIRS, ranked by E[max] (Kaggle scores best-of-your-2 on private)
    import itertools
    pairs = []
    for a, b in itertools.combinations(cands, 2):
        emax = np.maximum(ba[a], ba[b]).mean()
        pairs.append((a, b, emax, ba[a].mean(), ba[b].mean()))
    pairs.sort(key=lambda r: r[2], reverse=True)
    print("\n  ALL FINALS PAIRS by E[max] (best-of-2 private):")
    print("    pair                                E[max]      vs best-solo")
    best_solo = max(ba[k].mean() for k in cands)
    for a, b, emax, ma, mb in pairs[:8]:
        print(f"    {a} + {b:<12}   {emax:.6f}   {emax-best_solo:+.6f}")

    ba_, bb, emax, ma, mb = pairs[0]
    print(f"\n  BEST FINALS PAIR: {ba_} + {bb}  (E[max]={emax:.6f})")
    print("  NOTE: gaps are within draw-noise (MEMORY: private draw-dominated). LB is the only true "
          "out-of-sample signal — n129 highest CV (0.970410) but LB 0.97118 ≈ n091 0.97121, so they are "
          "STATISTICALLY TIED. A pair of two co-equal, slightly-decorrelated stacks maximizes E[max].")
    out = COMP / "probes" / "finals_robustness.csv"
    pd.DataFrame(pairs, columns=["a", "b", "E_max", "a_solo_E_BA", "b_solo_E_BA"]).to_csv(out, index=False)
    print(f"  saved -> {out.relative_to(COMP.parent.parent)}")


if __name__ == "__main__":
    main()
