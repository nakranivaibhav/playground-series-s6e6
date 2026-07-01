---
id: node_0083
desc: External SDSS DR17 coord-label augmentation
op: draft
parents: [root]
uses_data: [fs_realmlp_fe, fs_sdss_labels]
family: gbdt
status: dead
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.966623
gates: {schema_ok: false, oof_full: false, no_nan: false, dist_sane: false, leak_clean: false, cv_too_good: false, passed: false}
gate_note: "DEAD before training: full 5M SDSS DR17 catalog downloaded and crossmatched — label agreement at all tolerances (0.001–0.1 deg) is ~50%, worse than the 65% GALAXY base rate. The synthetic generator does NOT preserve real class labels at sky positions; real SDSS labels carry zero signal beyond noise. Hypothesis invalidated empirically."
leak: null
lb: null
submitted: null
created: 2026-06-13T11:56Z
decided: 2026-06-13T14:34Z
---

## plan
built on:   root (new data-augmentation draft); copies node_0030 (LightGBM richFE) for the pipeline.
change:     Build fs_sdss_labels (fit_in_fold): coordinate-match train+test rows to a downloaded real SDSS DR17 spectroscopic catalog by (alpha,delta) within a small tol → attach the REAL spectroscopic class as an extra GBDT feature/sample-weight prior; train a LightGBM-richFE base on it. node_0060 found the generator reuses 32-38% of real SDSS coords but only used provenance FLAGS — this uses the real LABELS.
hypothesis: The real SDSS DR17 spectroscopic label for the 32-38% coordinate-matched rows is a higher-fidelity target signal than the synthetic label, lifting recall on the confused minority classes.
target:     balanced accuracy maximize; solo > node_0060 0.966623 AND must pass mandatory LB-gate before promote.

Data-centric well (favored). node_0060 (LightGBM+SDSS17 provenance flags, journal
2026-06-11) PROVED the synthetic generator reuses real SDSS17 (alpha,delta) values for
32-38% of rows — but only added binary match FLAGS, which overfit. The unexploited
signal: for matched rows, the REAL spectroscopic label (GALAXY/QSO/STAR) is recoverable
from the public SDSS DR17 catalog and is far cleaner than the noisy synthetic label.

Download a SDSS DR17 specObj/photoObj slice covering this sky region (CasJobs/SkyServer
crossmatch, or a Kaggle-hosted SDSS catalog if internet-restricted — surface to human if
download blocked). fs_sdss_labels is fit_in_fold ONLY because the match-confidence/coverage
stats must be computed train-fold-only; the matched real label itself is external
(stateless) but treat as fit_in_fold to be safe.

MANDATORY LB-gate before any promote (node_0047 mirage rule, round_plan: label-touching
changes lift CV then can crash LB — train/test share the SAME generator noise per AQuA, so
a 'cleaner' real label may DISAGREE with what the metric scores).

Copy node_0030 (LightGBM richFE) for the pipeline. Read node_0060/node.md for the
coord-match recipe already written, and journal 2026-06-11 (C5) + round_plan C5/C11 lines.

well: data

## post-build findings

Downloaded full SDSS DR17 SpecObj catalog (5,112,724 rows) via SkyServer REST API in 36-degree RA batches.
Saved to comps/playground-series-s6e6/data/sdss17_full/ (specobj_full.csv + specobj_extra.csv).
Built cKDTree and crossmatched competition train rows against catalog at tolerances 0.001–0.1 degrees:

| tolerance | matches | agreement with train label |
|-----------|---------|---------------------------|
| 0.001°    | 1,160 (0.2%)  | 55.7% |
| 0.01°     | 87,314 (15.1%) | 49.9% |
| 0.05°     | 543,578 (94.2%) | 49.9% |
| 0.1°      | 575,485 (99.7%) | 50.1% |

Agreement ~50% at all scales (below the 65% GALAXY base rate). This means the synthetic
generator assigns classes INDEPENDENTLY of real sky positions — the real SDSS class at a
nearby coordinate is random noise relative to the competition label. The core hypothesis
is empirically false. NODE KILLED before training (zero GPU/CPU wasted on leaky/worthless
features). The full SDSS catalog is available at data/sdss17_full/ for future investigations.
