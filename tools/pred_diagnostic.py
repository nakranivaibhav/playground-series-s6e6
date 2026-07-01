#!/usr/bin/env python3
"""pred_diagnostic.py — structural comparison of two OOF prediction banks.

Why this exists
---------------
Balanced Accuracy (and most single scalars) is a *macro-average of per-class
recall* — it collapses a 3x3 error structure into one number. Two models with the
SAME BA can have completely different error geometry (e.g. +QSO recall / -GALAXY
recall nets to ~0 BA change). On a saturated ensemble, the gains that survive are
small and *localized*, so a scalar gate ("beats champion by > 2*sem") is blind to
exactly the signal we want to keep and combine.

This tool reads two nodes' `oof.npy` (n_rows x n_classes, rows aligned to the
frozen folds) and the true labels, and reports WHAT CHANGED and WHETHER IT IS REAL:

  1. per-class recall + precision for both, and the DELTA (the headline BA hides);
  2. full confusion matrices + the delta matrix;
  3. flip analysis — rows the candidate FIXES (champ-wrong -> cand-right) vs BREAKS
     (champ-right -> cand-wrong), by true class and by an optional region column
     (e.g. redshift band), with a McNemar exact test on the discordant pairs;
  4. a PAIRED BOOTSTRAP of the BA difference (fold-independent, far finer than 5
     coarse fold-means) — P(candidate > champion) and a CI. This is the honest
     quantitative gate that lets us drop below 2*sem WITHOUT reopening the n0047
     CV-mirage door: it tests "is this difference real?" directly on the rows.

It retrains NOTHING. Generic across competitions: pass any numeric `--region-col`
that exists in the labels CSV and it is auto-binned.

Usage
-----
  uv run --no-sync python tools/pred_diagnostic.py \
      --comp comps/playground-series-s6e6 \
      --champion node_0091 --candidate node_0116 \
      --region-col redshift --bootstrap 3000

Exit code is always 0; this is a read-only diagnostic, not a gate that fails CI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

try:
    from scipy.stats import binomtest  # scipy ships with sklearn
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - fallback if scipy missing
    _HAVE_SCIPY = False


# ----------------------------------------------------------------------------- IO
def _load_oof(comp: Path, node: str) -> np.ndarray:
    p = comp / "nodes" / node / "oof.npy"
    if not p.exists():
        raise SystemExit(f"missing OOF for {node}: {p}")
    a = np.load(p)
    if a.ndim != 2:
        raise SystemExit(f"{p} is not 2-D (got shape {a.shape})")
    return a.astype(np.float64)


def _load_labels(comp: Path, target_col: str, region_col: str | None):
    train = comp / "data" / "train.csv"
    if not train.exists():
        raise SystemExit(f"missing labels: {train}")
    want = [target_col] + ([region_col] if region_col else [])
    df = pd.read_csv(train, usecols=lambda c: c in want)
    classes = sorted(df[target_col].astype(str).unique())
    code = {c: i for i, c in enumerate(classes)}
    y = df[target_col].astype(str).map(code).to_numpy()
    region = df[region_col].to_numpy() if region_col else None
    return y, classes, region


def _holdout_mask(comp: Path, n_rows: int) -> np.ndarray:
    """True where a row is in the inviolable holdout (the last frozen fold)."""
    fp = comp / "folds.json"
    mask = np.zeros(n_rows, dtype=bool)
    if not fp.exists():
        return mask
    folds = json.load(open(fp)).get("folds", [])
    if not folds:
        return mask
    idx = np.asarray(folds[-1].get("val_idx", []), dtype=int)
    mask[idx] = True
    return mask


# ------------------------------------------------------------------- core metrics
def _per_class(y_true, y_pred, classes):
    """Return per-class recall and precision dicts."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    recall, precision = {}, {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        recall[c] = tp / cm[i, :].sum() if cm[i, :].sum() else float("nan")
        precision[c] = tp / cm[:, i].sum() if cm[:, i].sum() else float("nan")
    return cm, recall, precision


def _fmt_signed(x: float, nd: int = 4) -> str:
    return f"{x:+.{nd}f}"


def _section(title: str) -> str:
    return f"\n{'=' * 78}\n{title}\n{'=' * 78}"


def _report_split(name, y, pa, pb, classes, champ, cand):
    """Text report for one row-subset (all / working / holdout)."""
    out = [_section(f"SUBSET: {name}  (n={len(y):,})")]
    ba_a = balanced_accuracy_score(y, pa)
    ba_b = balanced_accuracy_score(y, pb)
    out.append(f"  Balanced Accuracy   {champ}: {ba_a:.6f}   {cand}: {ba_b:.6f}   "
               f"delta: {_fmt_signed(ba_b - ba_a, 6)}")

    cm_a, rec_a, pre_a = _per_class(y, pa, classes)
    cm_b, rec_b, pre_b = _per_class(y, pb, classes)

    out.append("\n  PER-CLASS RECALL  (the average BA hides) — champ / cand / delta")
    for c in classes:
        out.append(f"    {c:<7} {rec_a[c]:.4f} / {rec_b[c]:.4f} / "
                   f"{_fmt_signed(rec_b[c] - rec_a[c])}")
    out.append("  PER-CLASS PRECISION — champ / cand / delta")
    for c in classes:
        out.append(f"    {c:<7} {pre_a[c]:.4f} / {pre_b[c]:.4f} / "
                   f"{_fmt_signed(pre_b[c] - pre_a[c])}")

    out.append("\n  CONFUSION DELTA (cand - champ), rows=true, cols=pred  [%s]"
               % ", ".join(classes))
    d = cm_b - cm_a
    for i, c in enumerate(classes):
        row = "  ".join(f"{d[i, j]:+6d}" for j in range(len(classes)))
        out.append(f"    {c:<7} {row}")
    return "\n".join(out), (ba_a, ba_b)


def _flip_analysis(y, pa, pb, classes, region, region_col):
    out = [_section("FLIP ANALYSIS — candidate vs champion (paired, per row)")]
    a_ok = pa == y
    b_ok = pb == y
    fixes = (~a_ok) & b_ok      # champ wrong -> cand right
    breaks = a_ok & (~b_ok)     # champ right -> cand wrong
    nf, nb = int(fixes.sum()), int(breaks.sum())
    out.append(f"  FIXES  (champ wrong -> cand right): {nf:,}")
    out.append(f"  BREAKS (champ right -> cand wrong): {nb:,}")
    out.append(f"  NET (fixes - breaks):               {nf - nb:+,}")

    # McNemar exact test on discordant pairs: under H0 fixes ~ Binom(nf+nb, 0.5)
    nd = nf + nb
    if nd:
        if _HAVE_SCIPY:
            p = binomtest(nf, nd, 0.5, alternative="two-sided").pvalue
            ptxt = f"{p:.3g}"
        else:
            # normal approx
            z = (nf - nd / 2) / (0.5 * nd ** 0.5)
            ptxt = f"~{2 * (1 - 0.5 * (1 + abs(z) / (abs(z) + 1))):.3g} (approx)"
        out.append(f"  McNemar exact p (is the fix/break imbalance real?): {ptxt}")

    out.append("\n  NET FIX BY TRUE CLASS  (positive = candidate helps this class)")
    for i, c in enumerate(classes):
        m = y == i
        out.append(f"    {c:<7} fixes {int(fixes[m].sum()):>6,}  "
                   f"breaks {int(breaks[m].sum()):>6,}  "
                   f"net {int(fixes[m].sum()) - int(breaks[m].sum()):+,}")

    if region is not None:
        out.append(f"\n  NET FIX BY {region_col.upper()} BAND")
        edges = [-np.inf, 0.0025, 0.15, 0.5, 1.0, 2.0, np.inf]
        names = ["~0 (STAR-like)", "0.0025-0.15 (low-z)", "0.15-0.5",
                 "0.5-1.0", "1.0-2.0", ">2 (QSO-like)"]
        band = np.digitize(region, edges[1:-1])
        for b, nm in enumerate(names):
            m = band == b
            if not m.any():
                continue
            out.append(f"    {nm:<22} fixes {int(fixes[m].sum()):>6,}  "
                       f"breaks {int(breaks[m].sum()):>6,}  "
                       f"net {int(fixes[m].sum()) - int(breaks[m].sum()):+,}")
    return "\n".join(out), (fixes, breaks)


def _paired_bootstrap(y, pa, pb, n_boot, seed):
    out = [_section(f"PAIRED BOOTSTRAP of BA difference (B={n_boot:,})")]
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    base = balanced_accuracy_score(y, pb) - balanced_accuracy_score(y, pa)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        yk = y[idx]
        diffs[k] = (balanced_accuracy_score(yk, pb[idx])
                    - balanced_accuracy_score(yk, pa[idx]))
    p_better = float((diffs > 0).mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out.append(f"  observed BA delta (cand - champ): {_fmt_signed(base, 6)}")
    out.append(f"  bootstrap mean delta:            {_fmt_signed(diffs.mean(), 6)}")
    out.append(f"  95% CI:                          [{_fmt_signed(lo,6)}, {_fmt_signed(hi,6)}]")
    out.append(f"  P(candidate > champion):         {p_better:.3f}")
    verdict = ("REAL gain (>=0.90 confidence)" if p_better >= 0.90 else
               "REAL loss (<=0.10)" if p_better <= 0.10 else
               "INDISTINGUISHABLE on global BA — check the per-class / region "
               "structure above for a localized gain a stack can use")
    out.append(f"  VERDICT: {verdict}")
    return "\n".join(out), p_better


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comp", required=True, type=Path, help="comp dir (comps/<slug>)")
    ap.add_argument("--champion", required=True, help="baseline node id (e.g. node_0091)")
    ap.add_argument("--candidate", required=True, help="node id to compare")
    ap.add_argument("--target-col", default="class")
    ap.add_argument("--region-col", default=None,
                    help="optional numeric col in train.csv to bin (e.g. redshift)")
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    comp = args.comp
    oof_a = _load_oof(comp, args.champion)
    oof_b = _load_oof(comp, args.candidate)
    if oof_a.shape != oof_b.shape:
        raise SystemExit(f"shape mismatch {oof_a.shape} vs {oof_b.shape}")
    y, classes, region = _load_labels(comp, args.target_col, args.region_col)
    if len(y) != len(oof_a):
        raise SystemExit(f"label/oof length mismatch {len(y)} vs {len(oof_a)}")

    pa = oof_a.argmax(1)
    pb = oof_b.argmax(1)
    hold = _holdout_mask(comp, len(y))

    print(f"comp={comp.name}  champion={args.champion}  candidate={args.candidate}")
    print(f"classes (oof col order) = {classes}  | holdout rows = {int(hold.sum()):,}")

    # Report on all rows, then working (folds 0-3) and holdout (fold 4) separately:
    # a gain that holds on BOTH the working set and the untouched holdout is the
    # trustworthy kind (n0047's mirage would show on working-CV but not holdout).
    for name, m in [("ALL rows", np.ones(len(y), bool)),
                    ("WORKING (folds 0-3)", ~hold),
                    ("HOLDOUT (fold 4, inviolable)", hold)]:
        if not m.any():
            continue
        txt, _ = _report_split(name, y[m], pa[m], pb[m], classes,
                               args.champion, args.candidate)
        print(txt)

    txt, _ = _flip_analysis(y, pa, pb, classes, region, args.region_col)
    print(txt)
    txt, _ = _paired_bootstrap(y, pa, pb, args.bootstrap, args.seed)
    print(txt)


if __name__ == "__main__":
    main()
