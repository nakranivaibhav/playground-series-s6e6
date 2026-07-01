# eda — Predicting Stellar Class (playground-series-s6e6)

## 20-second summary
- 577,347 train / 247,435 test rows, 9 features + `id`, target `class` ∈ {GALAXY, QSO, STAR}.
- **Spotless data:** zero missing values, no `-9999`/`99` sentinels, no duplicate rows, no train↔test overlap. Cleaning is therefore minimal — the work is *features*, not repair.
- **`redshift` is the dominant physical separator** (STAR≈0, GALAXY low, QSO high); alone a shallow tree hits 0.84 balanced accuracy, with the 5 magnitudes 0.93. A full GBDT should land ~0.95+.
- **No leakage hazards.** No time column, no group key, `id` is randomly ordered w.r.t. the label → the CV scheme is **plain StratifiedKFold** (stratify on `class` because the metric is balanced accuracy and classes are imbalanced).

## Shape & schema
- train `(577347, 12)`, test `(247435, 11)`; the only train-only column is the target `class` (as expected).
- numeric: `alpha`,`delta` (sky coordinates, deg), `u`,`g`,`r`,`i`,`z` (SDSS photometric magnitudes), `redshift`.
- categorical: `spectral_type` (4 levels: M, A/F, G/K, O/B), `galaxy_population` (2 levels: Red_Sequence, Blue_Cloud).
- `id` is a pure row identifier (int64, monotonic, range 0–577346 train / 577347–824781 test, disjoint).

## Target
- GALAXY 377,480 (65.4%) · QSO 117,143 (20.3%) · STAR 82,724 (14.3%) — **imbalanced**.
- Metric is **balanced accuracy = mean of per-class recall**, so the 14% STAR and 20% QSO classes each count as much as the 65% GALAXY class. Implication: **class weighting / `class_weight='balanced'` (or per-class threshold tuning on predicted probabilities) is where the score lives**, not raw accuracy. A model that nails GALAXY but confuses QSO↔STAR will score poorly.

## Missingness
- None. `train.isna().sum().sum() == 0`, `test == 0`. No imputation needed; nothing to drop.

## Distributions & outliers
- `redshift` by class (median): STAR 0.057, GALAXY 0.482, QSO 1.799 — clear ordering with overlap in the GALAXY↔STAR low-z region and GALAXY↔QSO mid-z region (that overlap is where the contest is decided).
- A few thousand STAR/GALAXY rows have `redshift ≤ 0` (physically a tiny blueshift / measurement noise) — **not** a sentinel; leave as-is, trees handle it.
- Magnitudes are well-behaved (≈12–28). Train has **3 rows with `u < 10`** (one slightly negative) — rare physical outliers, test's `u` min is 13.9. Trees are robust to these; **not clipped globally** (clipping would be a fold-local modelling choice if ever needed).
- `alpha`∈[0,360], `delta`∈[-18,79] — sky position; likely weak signal but kept.

## Categorical signal (row-normalized P(class | level))
- `spectral_type`: **M → 95% GALAXY**, **O/B → 71% QSO**, A/F → mostly QSO/STAR, G/K → mostly GALAXY. Strongly informative, not deterministic.
- `galaxy_population`: **Red_Sequence → 90% GALAXY**, Blue_Cloud → mixed (42% QSO). Informative.
- These read like engineered/leaky-by-design helper columns but they are present **identically in test**, so they are legitimate features, not leakage.

## Leakage hazards → CV verdict
| hazard | finding | verdict |
|---|---|---|
| id / order leakage | `id` monotonic but class flips on ~51% of consecutive rows (≈ random order); id range disjoint train/test | **drop `id` from features**; not used as a signal. No order leakage. |
| time leakage | no date/time column | none → no TimeSeriesSplit |
| group leakage | no group/session/object-id key; no feature repeats as a group | none → no GroupKFold |
| train↔test duplicates | 0 exact feature-row matches, 0 within-set dup rows | none |
| CV-too-good tripwire | best single-feature shallow tree = 0.84, all-magnitudes = 0.93 (realistic, not ~1.0) | no perfect separator; no tripwire |

**→ `/kaggle-validate` should freeze `StratifiedKFold` (stratify on `class`), shuffle on, fixed seed.** Carve a stratified holdout.

## Proposed cleaning / feature steps (each = atomic, tested where stateless)
Coded + unit-tested now in `src/clean.py` (`src/test_clean.py`, 8 tests green):
1. `cast_categoricals` — cast `spectral_type`, `galaxy_population` to `category` dtype (native GBDT handling). *stateless.*
2. `add_color_features` — astronomy color indices `u-g, g-r, r-i, i-z, u-z` (band differences; canonical star/galaxy/QSO discriminators, brightness-invariant). *stateless, fit-inside-fold-safe by construction (proven by test).*
3. `feature_columns` — feature selector that excludes `id` and `class` (id-leakage guard).

Deferred to modelling nodes (NOT global — fit inside fold only if they learn anything):
- target/categorical *encoding* for the XGBoost branch (one-hot or ordinal) — fold-local.
- optional redshift transforms (`log1p` of shifted redshift) and magnitude ratios — try as feature nodes.
- class-weighting / per-class threshold tuning on OOF probabilities — a modelling node, the main score lever under balanced accuracy.
