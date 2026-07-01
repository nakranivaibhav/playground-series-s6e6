---
id: node_0043
desc: CatBoost config-B (de-corr hypers)
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: gbdt
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966209
sem: 0.000205
folds: [0.966754, 0.966206, 0.966349, 0.966247, 0.965488]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good is a warn (severity=warn): implausible jump from baseline 0.333 is expected for a well-tuned GBDT model; same pattern as n39. Not a blocker."
leak: clean
lb: null
submitted: null
created: 2026-06-07T16:42Z
decided: null
tags: [gbdt, catboost, config-b, de-correlation, memory-safe, draft]
---

## plan
built on:   root (new draft — a 2nd, hyperparameter-DIFFERENT CatBoost config, NOT a seed of
            n39). Template src to COPY from node_0039/src (the memory-fixed CatBoost: fs_realmlp_fe
            FE, native categorical handling, balanced class weights, early-stopping, fold-honest
            OOF over the FROZEN folds.json → oof.npy 577347×3 + test_probs.npy 247435×3). Keep the
            FE, native cats, balanced, early-stopping, and OOF scaffold byte-identical; change ONLY
            the CatBoost hyperparameters.
change:     swap to a DIFFERENT CatBoost config for de-correlation, distinct from n39's
            depth=6/border_count=128: e.g. depth=7 with border_count=96 (memory-safe), a different
            learning_rate / l2_leaf_reg / bootstrap, and/or grow_policy=Lossguide. Pick one coherent
            alternative hyperparameter set. MEMORY-SAFE is mandatory — keep peak RSS <20 GB (n34
            OOM'd at 29.5 GB with depth=8/border_count=254; n39 ran at 3.3 GB with depth=6/border=128,
            so depth=7/border=96 must be validated to stay well under 20 GB). CPU.
hypothesis: a different-hyperparameter CatBoost (different depth/regularization/grow policy) makes
            errors de-correlated from n39, giving the meta-learner extra GBDT signal beyond the
            single n39 CatBoost arm.
target:     BA maximize · solo ≥ 0.966 (near n39 0.967723). Valuable only if de-correlated from n39
            AND it lifts the re-stack vs champion node_0041 (0.969808) — re-run restack_probe.py.
