# Validation — comps/playground-series-s6e6  (frozen 2026-06-05T11:23Z)
official metric: Balanced Accuracy Score  (direction: maximize)

## Scheme
**StratifiedKFold · 5 splits · seed 42 (split-only) · n_rows 577,347 → folds.json**

Why it matches the official grade: Kaggle scores submissions with **balanced
accuracy** (the macro-average of per-class recall), and the three classes are
imbalanced (GALAXY 65.4% / QSO 20.3% / STAR 14.3%). We therefore **stratify on
`class`** so every fold preserves that exact class mix (verified: each fold is
0.654 / 0.203 / 0.143) — otherwise a fold could under-represent STAR/QSO and make
the per-class recall estimate noisy, which is precisely the quantity the metric
averages. Local CV is computed with the **same** `sklearn.metrics.balanced_accuracy_score`
and the **same** maximize direction, so our private grading mirrors Kaggle's.

EDA established there is **no time column, no group key, and `id` is randomly
ordered w.r.t. the label** (class flips on ~51% of consecutive ids), so neither
TimeSeriesSplit nor GroupKFold applies — a plain stratified random split is the
leak-correct choice. No `--scheme` override was used; this is the tool's
auto-pick for a classification task with no group/time. `id` is **dropped from
features** (never a predictor).

## Holdout (inviolable — never fit on)
- rows: **fold 4's val set — 115,469 rows (20.0%)**, positional indices spanning
  4…577345 (the exact list is `folds.json → folds[-1].val_idx`; load it to
  assert-exclude rather than copying 115k integers here).
- derivation: the last fold of the frozen 5-way stratified split (a clean,
  class-balanced 20% slice — not random-on-top, so it's reproducible from the seed).
- rule: these rows are **excluded from every `.fit()`** — model fit, encoders,
  scalers, target encoders, any feature statistic — forever. Working CV uses the
  other 4 folds (folds 0–3). The holdout is a one-shot final honesty check before
  the deadline, not a tuning set.

## Frozen contract
scheme + n_splits + seed are **immutable**; `folds.json` is the source of truth.
A node that re-splits with a different seed or scheme is **auto-rejected**, no
matter its CV. The split seed (42) controls ONLY the split, nothing else
(model seeds are separate).

## Decision gate — structural, not scalar (adopted 2026-06-17T11:43Z, director-directed)
5-fold frozen scheme is UNCHANGED. What changed is the *gate*: the raw 2·sem-on-BA
scalar is a screen, not the arbiter. After every node, run
`tools/pred_diagnostic.py --comp comps/playground-series-s6e6 --champion <champ> --candidate <node> --region-col redshift`
and judge on: (a) **paired bootstrap** P(cand>champ) ≥ 0.90, and/or (b) a
**McNemar-significant block of fixes** concentrated in a class/region that **also
holds on the fold-4 holdout**. Rationale (journal 2026-06-17T11:43Z): BA is a macro-recall average
and hid a p=4e-53, 4098-row GALAXY/low-z gain in n118 (filed -0.004 "worse"). The
n0047 mirage guard stands — working-CV-only gains die; sub-2·sem promotions are
LB-probe-gated before counting. Per-class recalls + flip summary go in each node record.
