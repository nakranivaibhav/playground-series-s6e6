---
id: node_0103
desc: TabM on linear flux-space features
op: draft
parents: [root]
uses_data: [fs_flux]
family: nn
status: buggy
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.939960]
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "CHEAP-KILL tripped: fold-0 BA=0.9400 < 0.965 threshold. Decorrelation OK (err-corr=0.485 < 0.75). Linear flux-space gave TabM much weaker signal than log-color space — only 21 features vs rich FE; no cats either."
leak: clean
lb: null
submitted: null
created: 2026-06-15T07:06Z
decided: 2026-06-15
---

## plan
built on:   root — fresh draft; copy node_0033's TabM-on-richFE training loop VERBATIM
            (`nodes/node_0033/src/solution.py`, strongest decorrelated NN base CV 0.968053).
            The ONLY change is the feature matrix → fs_flux.
change:     Train TabM on a NEW input representation: linear FLUX space, not magnitude space.
            fs_flux (stateless): f_b = 10^(−0.4·(mag_b − mag_mean)) for b∈{u,g,r,i,z}; pairwise flux
            RATIOS f_b/f_b'; the flux vector normalized to unit sum (SED-shape simplex); + raw redshift.
            NO magnitudes, NO log colors.
hypothesis: linear flux-ratio geometry gives TabM different decision boundaries than log-color space,
            decorrelating errors (<0.65) while staying at base tier (≥0.965).
target:     BA maximize · solo ≥0.965 AND err-corr vs node_0070 OOF < 0.65; stack-add beats 0.970355.

HOW: copy n033's loop; swap features for fs_flux. CRITICAL kill gate — a flux ratio f_b/f_b' = exp of a
log-color, so it carries the SAME information as a color (cf. n056 1D-CNN re-derived color slopes, corr
0.78). If fold-0 err-corr vs node_0070 ≥ 0.75, this is just re-encoded color space → KILL immediately.
Decorrelation must come from TabM splitting the linear-ratio surface + unit-sum simplex differently, not
new info. CHEAP-KILL: fold-0 first; solo BA < 0.965 → STOP. Else full 5 folds + err-corr + stack-add test.
Outputs: oof.npy (577347,3), test_probs.npy (247435,3), submission.csv, train.log. fs_flux = stateless.

## notes
well=data. GPU. fs_flux: row-wise deterministic, no fit/target/cross-row → stateless leak class.
