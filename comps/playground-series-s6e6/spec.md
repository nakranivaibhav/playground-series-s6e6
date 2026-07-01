# spec — Predicting Stellar Class (Playground Series S6E6)

We classify each astronomical observation as one of three object types — **GALAXY**,
**QSO** (quasar), or **STAR** — from photometric and spectral measurements (sky
coordinates, the u/g/r/i/z magnitudes, redshift, and two engineered categorical
features). The data is fully tabular: 577,347 labelled training rows and 247,435
unlabelled test rows, with **no missing values**. A submission is a CSV with two
columns, `id` and `class`, where `class` is the **predicted label** (not a
probability). Scoring is **Balanced Accuracy Score** — the macro-average of the
per-class recall (sklearn `balanced_accuracy_score`), so each of the three classes
counts equally despite the heavy imbalance (GALAXY ≈65%, QSO ≈20%, STAR ≈14%);
getting the two minority classes right matters as much as the majority. Up to 10
submissions/day; the final deadline is 2026-06-30 23:59 UTC.

## machine
```yaml
slug: playground-series-s6e6
title: Predicting Stellar Class
task_type: classification_multiclass
metric: Balanced Accuracy Score
metric_direction: maximize
id_col: id
target_col: class
target_cols: []
group_key: null
time_col: null
submission_columns: [id, class]
sample_submission: comps/playground-series-s6e6/data/sample_submission.csv
n_test_rows: 247435
daily_submission_limit: 10
deadline: 2026-06-30
```

## notes (non-spec, for downstream stages)
- classes: GALAXY (377,480) · QSO (117,143) · STAR (82,724) — imbalanced; **balanced
  accuracy weights each class equally**, so class-weighting / threshold tuning matters.
- features: `alpha`,`delta` (sky coords) · `u`,`g`,`r`,`i`,`z` (photometric mags) ·
  `redshift` (numeric, strong physical separator: STAR≈0, GALAXY low, QSO high) ·
  `spectral_type` (categorical, 4 levels: M, A/F, G/K, O/B) ·
  `galaxy_population` (categorical, 2 levels: Red_Sequence, Blue_Cloud).
- no nulls in train or test; train/test id ranges are disjoint (0–577346 / 577347–824781).
- public LB ≈ 20% of test, private ≈ 80%.
```
