---
id: node_0108
desc: TabM on arcsinh-flux (luptitude) FE
op: draft
parents: [root]
uses_data: [fs_luptitude]
family: nn
status: buggy
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.940626]
baseline_cv: 0.970355
gates: {schema_ok: false, oof_full: false, no_nan: false, dist_sane: false,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "GATE 1 TRIPPED: fold-0 BA=0.9406 < 0.965. err-corr=0.479 (gate 2 OK). Luptitudes hit IDENTICAL ceiling as n103/n107 (all three 0.940) — flux-space representation lever is STRUCTURALLY CLOSED regardless of conditioning transform."
leak: clean
lb: null
submitted: null
created: 2026-06-15T08:32Z
decided: 2026-06-15
---

## plan
built on:   root — third flux-representation probe, resolving the n103/n107 conditioning diagnosis.
            n103 (bare flux) and n107 (rich flux FE) BOTH hit fold-0 BA EXACTLY 0.940 with err-corr 0.486:
            the flux ceiling is CONDITIONING, not feature count — raw flux ratios are heavy-tailed and
            ill-conditioned for a NN, and the transform that conditions them (log) collapses them back to
            log-colors (corr 0.78, n056). Copy node_0033's TabM recipe VERBATIM (`nodes/node_0033/src`).
change:     Use ARCSINH-FLUX (luptitude / Lupton 1999) features instead of raw flux or log-color:
            mu_b = arcsinh(f_b / (2b)) where f_b = 10^(−0.4(mag_b−mag_mean)) and b is a small softening
            per band. Luptitudes interpolate: ≈ linear-flux for faint sources, ≈ magnitude (log) for
            bright — a WELL-CONDITIONED transform that keeps flux-space nonlinearity. Features: the 5
            luptitudes, all pairwise luptitude differences (the luptitude analogue of colors), a couple
            of luptitude aggregates (mean/range), + raw redshift + log1p(redshift) + the two categoricals
            (as n033). This is the flux analogue of fs_realmlp_fe but in arcsinh space.
hypothesis: if the n103/n107 ceiling is purely conditioning, luptitudes (well-conditioned flux geometry)
            recover BA to tier (≥0.965) while keeping err-corr BELOW the bank (<0.65, between the pure-flux
            0.486 and the magnitude 0.78). DECISIVE either way: a win = first strong+decorrelated base since
            FT-T; a corr collapse to ~0.78 = arcsinh≈magnitude, flux lever STRUCTURALLY CLOSED.
target:     BA maximize · GATE 1 fold-0 solo BA ≥ 0.965; GATE 2 fold-0 err-corr vs node_0070 < 0.65;
            both pass → full OOF + stack-add to n091 must beat 0.970355.

HOW (TIGHT — single base, NO full-pool loader; do NOT read node_0091's solution.py):
- cp nodes/node_0033/src/solution.py → nodes/node_0108/src/solution.py as the TabM/PLR/I/O skeleton;
  ONLY the numeric feature build changes → fs_luptitude. Keep categoricals fed as n033 does.
- fs_luptitude (stateless): per band b, mu_b = arcsinh( f_b / (2*soft_b) ) with f_b=10^(−0.4(mag_b−mag_mean));
  pick soft_b as a fixed small constant (e.g. 1e-2 of the band's typical flux, or simply softening=0.01 on
  the mag_mean-centered flux — choose ONE fixed constant, document it; it is NOT fit on data → stateless).
  Then: 5 luptitudes, all 10 pairwise luptitude differences, luptitude mean + range, raw redshift,
  log1p(redshift). NO raw magnitudes, NO log-colors.
- GATE ORDER: fold-0 only first; solo BA + err-corr vs nodes/node_0070/oof.npy (577347,3). BA<0.965 OR
  err-corr≥0.65 → STOP, record. Else all 5 folds + full CV/sem/folds + mean err-corr.
- Outputs nodes/node_0108/{oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, train.log}.
  Self-gate (kaggle-leakage): fs_luptitude stateless (softening is a fixed constant, NOT fit); folds frozen;
  OOF full/no-NaN/each-row-once; dist sane; schema vs sample_submission. Write gates + cv/sem/folds + leak +
  err-corr (gate_note); stage: built. Do NOT submit. `uv run`. GPU — marker
  DONE=/tmp/playground-series-s6e6_node_0108.done, touch on completion.

## notes
well=data. fs_luptitude = stateless. This is the DECISIVE flux probe: it resolves whether the flux-
decorrelation lever (proven at 0.486) can ever be made strong, or is structurally capped (RBF pattern).
If luptitudes ALSO fail (corr→0.78 or BA<0.965), close the flux/representation-decorrelation avenue and
go back to the proposer for genuinely new territory.
