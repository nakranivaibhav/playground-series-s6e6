#!/usr/bin/env python3
"""capturability_check.py — is the complementary signal CAPTURABLE or just diffuse?

The sweep showed ~12 nodes fix a holdout-significant block of rows the champion
misses, but all wash on global BA because each model's fixes are coupled to breaks
elsewhere. Question: is that signal CONCENTRATED in the low-z GALAXY<->STAR zone
(capturable by a region-gated correction) or diffuse (uncapturable)?

Leak-free test: a region-gated correction rule with ONE tunable integer K
(how many complementary models must agree to override the champion), in the low-z
band only. Tune K on the WORKING folds (0-3); apply to the inviolable HOLDOUT
(fold 4) and measure the BA delta. A working-only gain that does NOT survive on the
holdout is the n0047 mirage; a gain that survives on the untouched holdout is real
recoverable headroom and justifies building the principled region-interacted meta.

Read-only. Run from repo root:
  uv run --no-sync python comps/playground-series-s6e6/probes/capturability_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

COMP = Path(__file__).resolve().parent.parent
CHAMP = "node_0091"
# complementary set: holdout-significant nodes that are still reasonably strong
# (BA >= 0.966) so a vote among them is not dominated by weak models.
COMP_NODES = ["node_0118", "node_0030", "node_0060", "node_0085",
              "node_0094", "node_0099"]
GAL, QSO, STAR = 0, 1, 2  # alphabetical oof col order


def main():
    df = pd.read_csv(COMP / "data" / "train.csv", usecols=["class", "redshift"])
    classes = sorted(df["class"].astype(str).unique())
    y = df["class"].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    rs = df["redshift"].to_numpy()
    n = len(y)

    hold = np.zeros(n, bool)
    hold[np.asarray(json.load(open(COMP / "folds.json"))["folds"][-1]["val_idx"], int)] = True
    work = ~hold

    champ = np.load(COMP / "nodes" / CHAMP / "oof.npy").argmax(1)
    votes = np.stack([np.load(COMP / "nodes" / nd / "oof.npy").argmax(1)
                      for nd in COMP_NODES], axis=1)  # (n, n_models)

    # low-z GALAXY<->STAR boundary zone (where n118's fixes concentrated)
    zone = (rs > 0.0025) & (rs < 0.15)
    print(f"low-z zone rows: {int(zone.sum()):,} ({zone.mean():.1%})  | "
          f"complementary models: {COMP_NODES}")

    base_work = balanced_accuracy_score(y[work], champ[work])
    base_hold = balanced_accuracy_score(y[hold], champ[hold])
    print(f"\nchampion BA  working={base_work:.6f}  holdout={base_hold:.6f}")

    # Correction rule: in-zone, restrict to GALAXY<->STAR confusions. If >=K of the
    # complementary models agree on a single GALAXY-or-STAR label that differs from
    # the champion, override to it. K tuned on WORKING only.
    def apply_rule(K, mask):
        out = champ.copy()
        in_play = mask & zone & np.isin(champ, [GAL, STAR])
        # count votes for GALAXY and STAR among complementary models
        vg = (votes == GAL).sum(1)
        vs = (votes == STAR).sum(1)
        to_gal = in_play & (champ == STAR) & (vg >= K) & (vg > vs)
        to_star = in_play & (champ == GAL) & (vs >= K) & (vs > vg)
        out[to_gal] = GAL
        out[to_star] = STAR
        return out, int(to_gal.sum() + to_star.sum())

    print("\n  K   workBA      dWork     #switch(work)")
    best = (None, -1)
    for K in range(1, len(COMP_NODES) + 1):
        pred, _ = apply_rule(K, work)
        ba = balanced_accuracy_score(y[work], pred[work])
        nsw = apply_rule(K, work)[1]
        d = ba - base_work
        print(f"  {K}   {ba:.6f}  {d:+.6f}   {nsw:,}")
        if ba > best[1]:
            best = (K, ba)

    K = best[0]
    print(f"\n  best K on WORKING = {K}")
    # apply that K to the HOLDOUT (never used to tune)
    pred_h, nsw_h = apply_rule(K, hold)
    ba_h = balanced_accuracy_score(y[hold], pred_h[hold])
    print(f"  HOLDOUT: champion {base_hold:.6f} -> corrected {ba_h:.6f}  "
          f"delta {ba_h - base_hold:+.6f}  ({nsw_h:,} switches)")
    verdict = ("CAPTURABLE — survives on the untouched holdout; build the "
               "region-interacted meta" if ba_h > base_hold else
               "NOT capturable by this simple gate — holdout did not improve; "
               "signal is coupled/diffuse, needs a smarter conditioner or is a wash")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
