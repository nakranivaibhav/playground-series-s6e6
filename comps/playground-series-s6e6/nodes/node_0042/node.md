---
id: node_0042
desc: RealMLP config-B (de-corr arch)
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969110
sem: 0.000323
folds: [0.970302, 0.968394, 0.968915, 0.968769, 0.969172]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "cv_too_good warn is expected — jump from baseline 0.333 to 0.969 triggers tripwire for known-good architecture; not a blocker"
leak: clean
lb: null
submitted: null
created: 2026-06-07T16:42Z
decided: null
tags: [nn, realmlp, config-b, de-correlation, draft]
---

## plan
built on:   root (new draft — a 2nd, architecturally-DIFFERENT RealMLP config, NOT a
            seed-bag partner of n28/n32/n35). Template src to COPY from node_0028/src (the
            reference RealMLP recipe: fs_realmlp_fe FE, PBLD embeddings, ScalingLayer, robust
            preprocessing, fold-honest OOF over the FROZEN folds.json → oof.npy 577347×3 +
            test_probs.npy 247435×3). Keep the FE and the OOF/test-probs scaffold byte-identical;
            change ONLY the RealMLP architecture hyperparameters.
change:     swap the RealMLP architecture to a DIFFERENT config (not just a seed) for de-correlation:
            change the hidden dims away from n28's [512,512,512] — e.g. WIDER [768,768,768] or
            DEEPER [512]*4 — AND nudge the dropout (~0.08 vs n28's expm4t ~0.044), and/or change
            the PBLD periodic-embedding dim. Pick one coherent alternative architecture (n_ens
            kept). Everything else (fs_realmlp_fe FE, robust preprocessing median/IQR fit-in-fold,
            GPU, frozen folds) stays identical so the only moved variable is the architecture.
hypothesis: a config-diverse RealMLP makes errors that differ from the n28/n32/n35 seed-bag (same
            arch, only seeds differ → near-identical error structure). A genuinely different
            architecture should be more de-correlated and give the meta-learner fresh signal.
target:     BA maximize · solo ≥ 0.967 (near n28 0.969065). Valuable only if it LIFTS the re-stack
            vs champion node_0041 (0.969808) — re-run restack_probe.py to confirm.
