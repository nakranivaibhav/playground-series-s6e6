---
id: node_0028
desc: RealMLP reference recipe (FE+PBLD)
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969065
sem: 0.000278
folds: [0.970072, 0.968456, 0.968899, 0.968716, 0.969180]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good tripwire fired (relative gain=0.9536 > 0.9 threshold) — expected for a dataset where the best models score ~0.970; consistent with public reference recipe ~0.96973."
leak: clean
lb: null
submitted: null
created: 2026-06-07T12:18Z
decided: null
tags: [nn, realmlp, pbld, reference-recipe, feature-engineering, draft]
---

## plan
built on:   root (new draft — NOT an improve on node_0024, which was bare pytabkit-TD on
            bare fs_research and capped at 0.949). Template src to COPY from node_0024/src
            (the RealMLP fold-honest OOF/test_probs scaffold + frozen folds.json loop stays),
            but the FE and model spec are largely REPLACED with the public reference recipe
            (refs/realmlp-v5-for-s6e6.py, refs/ps-s6-e6-realmlp-pytorch.py).
change:     port the PROVEN public RealMLP recipe in two parts.
            (a) HEAVY stateless FE → new feature-set fs_realmlp_fe: redshift ratios
            (g/redshift, i/redshift), log1p(redshift), all 7 color pairs (u-g, g-r, r-i, i-z,
            u-r, g-i, r-z), mag_mean, mag_range, integer-floor categorical views of every base
            numeric, category cross-combos. All row-wise deterministic → STATELESS.
            (b) fully-specified RealMLP: n_ens=8 internal ensemble, hidden [512,512,512],
            PBLD periodic numerical embeddings, front ScalingLayer, expm4t dropout schedule
            (~0.044), GELU; robust preprocessing median_center → robust_scale(IQR) →
            smooth_clip → l2_normalize (the model's own median/IQR fit is the only fit-in-fold
            part). Fold-honest OOF over the FROZEN folds.json → oof.npy (577347×3) +
            test_probs.npy (247435×3). GPU, serialized.
hypothesis: our 0.949 (node_0024) was missing the heavy FE + a fully-specified model, NOT a
            ceiling. The reference recipe (single RealMLP ~0.96973 in public notebooks) should
            reach ~0.965-0.970 here and finally supply the stack a STRONG, de-correlated NN base
            — the strength the prior NN zoo (0.943-0.959) lacked.
target:     BA maximize; solo ≥ ~0.965 (vs node_0024 0.949). Only valuable if it also lifts the
            re-stack vs champion node_0020 (0.966627) — re-run restack_probe.py to confirm.
