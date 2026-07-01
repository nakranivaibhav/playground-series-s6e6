---
id: node_0109
desc: LightGBM on rich flux-space FE
op: draft
parents: [root]
uses_data: [fs_flux_rich]
family: gbdt
status: buggy
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.935634]
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "GATE TRIPPED: fold-0 BA=0.9356 < 0.965 threshold; run stopped after fold 0. GBDT also caps at ~0.940 on fs_flux_rich — same ceiling as NNs. Flux information ceiling confirmed, not NN-conditioning artifact."
leak: clean
lb: null
submitted: null
created: 2026-06-15T11:09Z
decided: 2026-06-15
---

## plan
built on:   root — the rebuttal to my own "flux structurally closed" finding. All 3 flux nodes
            (n103/n107/n108) capped at BA 0.940 — but ALL THREE used TabM/RealMLP NNs, which choke on
            heavy-tailed ill-conditioned flux ratios. The 0.940 ceiling was an NN-CONDITIONING artifact,
            NOT a flux-information ceiling.
change:     Train a LightGBM base on fs_flux_rich (the rich linear-flux representation, NOT log-colors),
            copying node_0030's LGBM-on-rich-FE hyperparameters. A GBDT needs no conditioning — trees
            split on raw flux-ratio thresholds natively; and a flux ratio f_b/f_b' is NOT a monotone
            transform of a single feature (it is bivariate), so it is genuinely new split geometry vs
            log-color trees (a tree IS invariant to single-band monotone transforms, but NOT to ratios).
hypothesis: a LightGBM on fs_flux_rich recovers tier BA (≥0.965) where the NNs capped at 0.940, while
            inheriting the flux representation's ~0.48 err-corr decorrelation → the first
            strong-AND-decorrelated base, which lifts the n091 stack.
target:     BA maximize · GATE fold-0 solo BA ≥ 0.965 AND err-corr vs node_0070 bank < 0.65; if both pass
            → full OOF + restack onto n091 must beat 0.970355 by >2·sem to promote.

HOW (TIGHT — single base, NO full-pool loader; do NOT read node_0091's solution.py):
- cp nodes/node_0030/src/solution.py → nodes/node_0109/src/solution.py as the LightGBM-on-rich-FE
  skeleton (its LGBM params, frozen-fold loop, OOF/test/submission writing, class handling). The ONLY
  change is the feature matrix → fs_flux_rich.
- PORT the fs_flux_rich builder VERBATIM from nodes/node_0107/src/solution.py (function add_flux_rich_features,
  ~line 111: 5 linear fluxes f_b=10^(−0.4(mag_b−mag_mean)), all pairwise flux ratios, unit-sum SED simplex,
  flux aggregates, flux×redshift, raw redshift + log1p, + the 2 categoricals native to LightGBM). It is
  stateless — copy it into this node's own src/.
- GATE ORDER: fold-0 only first; solo BA + err-corr vs nodes/node_0070/oof.npy (577347,3). BA<0.965 OR
  err-corr≥0.65 → STOP, record. Else all 5 folds + full CV/sem/folds + mean err-corr.
- Outputs nodes/node_0109/{oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, train.log}.
  Self-gate (kaggle-leakage): fs_flux_rich stateless; folds frozen; OOF full/no-NaN/each-row-once; dist
  sane; schema vs sample_submission. Write gates + cv/sem/folds + leak + err-corr (gate_note); stage: built.
  Do NOT submit. `uv run` (lightgbm already a dep). CPU minutes — marker
  DONE=/tmp/playground-series-s6e6_node_0109.done if long.

## notes
well=outside. If this clears tier AND <0.65 err-corr, draft a combine next round to stack-add onto n091
(genuinely new base). If it ALSO caps at ~0.940, the flux ceiling is information-level not NN-conditioning
and the flux avenue is closed for ALL model families (decisive either way).
