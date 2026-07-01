---
id: node_0107
desc: TabM on RICH flux-space FE
op: draft
parents: [root]
uses_data: [fs_flux_rich]
family: nn
status: buggy
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.940558]
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "GATE 1 tripped: fold-0 BA=0.9406 < 0.965; err-corr=0.486 (same as n103 — decorrelation preserved but strength insufficient with 49 num+2 cat feats; n0033 uses much richer fit_in_fold FE; flux-space may need fit_in_fold target-encode or more interactions to recover BA)"
leak: clean
lb: null
submitted: null
created: 2026-06-15T08:24Z
decided: 2026-06-15
---

## plan
built on:   root — DIRECT follow-up to node_0103. n103 (TabM on bare fs_flux: 21 feats, no cats)
            cheap-killed at BA 0.940 BUT returned err-corr 0.485 vs the bank — the MOST DECORRELATED
            base we have ever built (RBF 0.53, FT-T 0.53, everything else 0.70-0.79). Decorrelation is
            PROVEN; n103 died only on STRENGTH because the feature set was starved. Copy node_0033's
            TabM-on-fs_realmlp_fe recipe VERBATIM (`nodes/node_0033/src/solution.py`, CV 0.968053).
change:     Replace the feature matrix with fs_flux_rich = the flux-space ANALOGUE of fs_realmlp_fe,
            keeping the richness that n103 dropped: linear fluxes f_b=10^(−0.4(mag_b−mag_mean)) for
            b∈{u,g,r,i,z}; ALL pairwise flux ratios f_b/f_b'; unit-sum SED simplex; flux aggregates
            (flux mean, flux range/spread, brightest/faintest-band one-hot); flux×redshift and
            flux-ratio×redshift interactions; raw redshift + log1p(redshift); AND the two engineered
            categoricals (spectral_type, galaxy_population — native/PLR-embedded as in n033). NO raw
            magnitudes, NO log colors (keep the flux geometry pure so decorrelation survives).
hypothesis: the flux representation gave TabM genuinely decorrelated errors (0.485) but too little signal
            at 21 feats; restoring the rich-FE breadth (aggregates + interactions + categoricals) recovers
            the ~2.5pp BA to clear tier (≥0.965) WHILE keeping err-corr low — the strong-AND-decorrelated
            base the saturated bank has lacked since FT-Transformer.
target:     BA maximize · GATE 1 fold-0 solo BA ≥ 0.965; GATE 2 fold-0 err-corr vs node_0070 < 0.65
            (n103 hit 0.485, so expect well under — but adding categoricals/raw-z-interactions may pull
            it UP toward the bank; watch it); if both pass, full OOF + stack-add to n091 must beat 0.970355.

HOW (TIGHT — single base, NO full-pool loader; do NOT read node_0091's solution.py):
- cp nodes/node_0033/src/solution.py → nodes/node_0107/src/solution.py as the I/O + TabM skeleton.
  Keep its frozen-fold loop, OOF/test/submission writing, class-balanced training, PLR embeddings,
  and its categorical handling VERBATIM. The ONLY change is the NUMERIC feature build → fs_flux_rich.
- Build fs_flux_rich in this node's own src/ (stateless; row-wise deterministic, no fit/target/cross-row).
  Mirror the fs_realmlp_fe aggregate/interaction structure but in flux space. KEEP the two categoricals
  exactly as n033 feeds them.
- GATE ORDER: run fold-0 ONLY first; compute solo BA and err-corr vs nodes/node_0070/oof.npy (577347,3).
  If BA < 0.965 OR err-corr ≥ 0.65 → STOP, record, do not run remaining folds. Else run all 5 folds,
  report full CV/sem/folds + final mean err-corr, and whether err-corr < 0.65 (stack-add bar).
- If it clears both gates AND solo is tier-competitive, ALSO report a quick stack-add: does adding this
  base's OOF to a simple bank+this LogReg beat the bank-alone? (rough signal; full stack-add can be a
  follow-up combine node). Produce nodes/node_0107/{oof.npy (577347,3), test_probs.npy (247435,3),
  submission.csv, train.log}. Self-gate (kaggle-leakage): fs_flux_rich stateless (no .fit on features);
  folds frozen; OOF full/no-NaN/each-row-once; dist sane; schema vs sample_submission
  (uv run tools/validate_submission.py --submission <p> --sample <s>); cv_too_good eyeball. Write gate
  booleans + cv/sem/folds + leak + err-corr (gate_note); stage: built. Do NOT submit. `uv run`. GPU —
  background with marker DONE=/tmp/playground-series-s6e6_node_0107.done, touch on completion.

## notes
well=data. fs_flux_rich = stateless. The decisive question is BA-recovery, not decorrelation (already
proven at 0.485). If this clears tier AND stays <0.65, draft a combine node next to stack-add it onto n091.
If it clears tier but err-corr jumps ≥0.65 (categoricals/raw-z pulling it back onto the bank manifold),
that localizes the decorrelation to the pure-flux-ratio geometry — then retry with categoricals dropped.
