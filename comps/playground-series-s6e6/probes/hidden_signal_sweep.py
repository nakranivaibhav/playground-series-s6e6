#!/usr/bin/env python3
"""hidden_signal_sweep.py — mine the whole bank for gains the scalar BA hides.

For every valid node with an oof.npy, compare it to the champion at the PREDICTION
level (not the scalar): net fixes (champ-wrong -> node-right minus the reverse),
the McNemar significance of that imbalance, the same restricted to the inviolable
holdout fold, and a paired-bootstrap P(node > champ on global BA). Rank by the
holdout-confirmed significant complementary signal — i.e. nodes that are flat-or-
worse on global BA yet genuinely fix a block of rows the champion misses, which a
region/class-gated combine could capture (cf. n118: -0.004 BA but p=4e-53 on a
4098-row GALAXY/low-z fix-block).

Read-only, no training. Run from repo root:
  uv run --no-sync python comps/playground-series-s6e6/probes/hidden_signal_sweep.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

try:
    from scipy.stats import binomtest
    def _mcnemar_p(nf, nb):
        nd = nf + nb
        return binomtest(nf, nd, 0.5, alternative="two-sided").pvalue if nd else 1.0
except Exception:
    def _mcnemar_p(nf, nb):
        nd = nf + nb
        if not nd:
            return 1.0
        z = (nf - nd / 2) / (0.5 * nd ** 0.5)
        return float(2 * (1 - 0.5 * (1 + abs(z) / (abs(z) + 1))))

COMP = Path(__file__).resolve().parent.parent
CHAMP = "node_0091"
BOOT = 1500
SEED = 42


def main():
    y = (pd.read_csv(COMP / "data" / "train.csv", usecols=["class"])["class"]
         .astype(str))
    classes = sorted(y.unique())
    y = y.map({c: i for i, c in enumerate(classes)}).to_numpy()
    n = len(y)

    hold = np.zeros(n, bool)
    folds = json.load(open(COMP / "folds.json"))["folds"]
    hold[np.asarray(folds[-1]["val_idx"], int)] = True

    champ = np.load(COMP / "nodes" / CHAMP / "oof.npy").argmax(1)
    champ_ok = champ == y
    ba_champ = balanced_accuracy_score(y, champ)
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, n, n) for _ in range(BOOT)]

    rows = []
    for nd_dir in sorted((COMP / "nodes").glob("node_*")):
        nid = nd_dir.name
        if nid == CHAMP:
            continue
        oof_p = nd_dir / "oof.npy"
        if not oof_p.exists():
            continue
        try:
            oof = np.load(oof_p)
        except Exception:
            continue
        if oof.shape != (n, len(classes)):
            continue
        # only consider nodes that actually scored (valid/champion); skip dead-by-status
        md = (nd_dir / "node.md").read_text(errors="ignore")
        status = (re.search(r"^status:\s*(\w+)", md, re.M) or [None, "?"])[1]
        if status in ("dead", "buggy", "proposed", "running"):
            continue

        p = oof.argmax(1)
        ok = p == y
        fixes = (~champ_ok) & ok
        breaks = champ_ok & (~ok)
        nf, nb = int(fixes.sum()), int(breaks.sum())
        nf_h = int((fixes & hold).sum())
        nb_h = int((breaks & hold).sum())
        ba = balanced_accuracy_score(y, p)
        # paired bootstrap P(node>champ) on global BA, shared resamples
        wins = 0
        for idx in boot_idx:
            yk = y[idx]
            if balanced_accuracy_score(yk, p[idx]) > balanced_accuracy_score(yk, champ[idx]):
                wins += 1
        rows.append(dict(
            node=nid, status=status, ba=round(ba, 6), dBA=round(ba - ba_champ, 6),
            net_fix=nf - nb, mcnemar_p=_mcnemar_p(nf, nb),
            net_fix_holdout=nf_h - nb_h, mcnemar_p_hold=_mcnemar_p(nf_h, nb_h),
            P_better=round(wins / BOOT, 3),
        ))

    df = pd.DataFrame(rows)
    # "hidden signal" = significant net fixes that ALSO hold on holdout, regardless
    # of global dBA sign. Rank by holdout net-fix among the holdout-significant ones.
    df["holdout_sig"] = (df.mcnemar_p_hold < 0.05) & (df.net_fix_holdout > 0)
    df = df.sort_values(["holdout_sig", "net_fix_holdout"], ascending=[False, False])
    pd.set_option("display.width", 200, "display.max_rows", 200)
    cols = ["node", "status", "ba", "dBA", "net_fix", "mcnemar_p",
            "net_fix_holdout", "mcnemar_p_hold", "P_better", "holdout_sig"]
    print(f"champion {CHAMP}  BA={ba_champ:.6f}  | nodes compared={len(df)}  "
          f"| holdout-significant complementary nodes={int(df.holdout_sig.sum())}\n")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    out = COMP / "probes" / "hidden_signal_sweep.csv"
    df[cols].to_csv(out, index=False)
    print(f"\nsaved -> {out.relative_to(COMP.parent.parent)}")


if __name__ == "__main__":
    main()
