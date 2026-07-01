---
id: node_0100
desc: FWLS region-interacted meta-stack
op: improve
parents: [node_0091]
uses_data: [fs_fwls_region]
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969815
sem: 0.000192
folds: [0.970495, 0.969836, 0.969390, 0.969514, 0.969838]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "NULL — FWLS region-interacted meta CV 0.969815 vs champion LogReg 0.970355, lift −0.000540 (all 5 folds worse vs the apples-to-apples sub=80000 sanity baseline 0.970054 → −0.000239). Region interactions (945 cols vs 189) add no signal: the bases are individually strong across all redshift regions, so region-adaptive base weighting recovers nothing — same verdict as the n091 per-class-threshold probe (WASH). Built under orchestrator marker-takeover after the developer agent overflowed context at 289 tool uses; process (pid 64490) finished cleanly. Region-adaptive-meta axis now CLOSED."
leak: clean
lb: null
submitted: null
created: 2026-06-14T13:38Z
decided: 2026-06-15
---

## plan
built on:   champion node_0091 (balanced multinomial LogReg over the FULL OOF pool @C=0.003). Same
            pool, same meta family — ONE change to the meta's INPUT.
change:     Feature-Weighted Linear Stacking (FWLS, Sill et al. 2009): augment the meta feature matrix
            with INTERACTIONS between each base's class-prob columns and a small set of REDSHIFT-REGION
            indicators (~3-4 redshift quantile bins). This lets the linear meta learn region-ADAPTIVE
            base weights (e.g. trust the GBDT bases at high-z, the NN bases at low-z) instead of one
            global weight per base. NOT a bigger meta (n80 closed that) — a region-conditional LINEAR
            meta, a different axis.
hypothesis: if different bases are systematically better in different redshift regions, the global
            LogReg leaves that on the table; FWLS region-interactions recover it. HONEST CAVEAT / low-EV:
            the bases are individually strong across all regions, and the coarse region-adaptation of
            per-class thresholds already washed (n091-threshold probe); region-interacted stacking is a
            richer version but likely also sub-2sem. The value is closing the FWLS axis cleanly.
target:     Balanced Accuracy maximize. Promote ONLY if fold-honest stacked CV beats champion 0.970355
            by > 2·sem (≈0.970851). Region interactions can inflate the meta's parameter count → watch
            cv_too_good (eyeball per-fold); an apparent win needs an LB probe before any finals move.

HOW (TIGHT — avoid re-deriving the loader; node_0099 overflowed doing that):
- cp nodes/node_0099/src/solution.py → nodes/node_0100/src/solution.py as the STARTING POINT. It is a
  WORKING full-pool loader: it builds the LogReg feature matrix `OOF_mat` (full-pool clipped log-probs),
  loads train.csv (redshift available), has nested_cv_arm_logreg(OOF_mat,...), logp()/norm()/score_fn(),
  frozen-fold loop. DO NOT read node_0091's solution.py or re-derive the loader.
- The ONLY change: before the LogReg arm, AUGMENT the feature matrix. Build region indicators R =
  one-hot of redshift quantile bin (q=3 or 4; bin EDGES computed on TRAIN-FOLD rows only = fit_in_fold;
  apply to val+test). New meta input = concat[ OOF_mat , OOF_mat ⊗ R_columns ] (i.e. each base-prob
  column multiplied by each region indicator). Keep the SAME nested-C LogReg arm, the SAME fold-honest
  loop, the SAME FULL pool. Run the FULL arm (and TIGHT if cheap).
- A/B: reproduce the plain-OOF_mat LogReg baseline (assert ≈0.970355), then report the FWLS-augmented
  CV, per-fold deltas, sem, and whether it clears 0.970355 + 2·sem.
- Produce nodes/node_0100/{oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, train.log}.
  Self-gate (kaggle-leakage): region bin-edges fit train-fold-only (fit_in_fold); meta fold-honest;
  folds frozen; OOF full/no-NaN; dist sane; schema; cv_too_good. Write gate booleans + cv/sem/folds +
  leak; stage: built. Do NOT submit. `uv run`. CPU minutes — background with marker
  (DONE=/tmp/playground-series-s6e6_node_0100.done) ONLY if long.

## notes
fs_fwls_region (fit_in_fold): redshift quantile-bin one-hot, edges train-fold-only, used only to form
meta interaction terms (not a base feature). If FWLS washes, the region-adaptive-meta axis is closed too.
