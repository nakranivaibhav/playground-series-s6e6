---
id: node_0027
desc: TabPFN-v3 multiclass base
op: improve
parents: [node_0025]
uses_data: [fs_research]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.940930
sem: 0.000403
folds: [0.939590, 0.941452, 0.940957, 0.940669, 0.941981]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warn expected — 0.9409 >> 0.333 random baseline; consistent across all TabPFN nodes on this well-separable 3-class task. Not implausible."
leak: clean
lb: null
submitted: null
created: 2026-06-07T11:12Z
decided: 2026-06-07T11:47Z
tags: [nn, tabpfn, tabpfn-v3, multiclass, foundation-model, in-context, cuda, diversity-arm, improve]
---

## plan
built on:   node_0025 (TabPFN-2.5 large-samples, the fast fixed subsample-ensemble). Keep
            EVERYTHING byte-identical: the fs_research load, frozen folds.json loop, fold-honest
            OOF (contexts drawn ONLY from each fold's TRAIN indices, label-free), full-train
            context for test, oof.npy (577347×3) + test_probs.npy (247435×3), and the critical
            batching fix — encode the context ONCE and run ONE predict call per subsample, K=8.
            Template src copied from node_0025/src.
change:     ONE atomic change — swap the checkpoint. Replace the TabPFN-2.5 checkpoint
            `tabpfn-v2.5-classifier-v2.5_large-samples.ckpt` with the TabPFN-v3 multiclass
            checkpoint from HF repo `Prior-Labs/tabpfn_3`, file
            `tabpfn-v3-classifier-v3_20260417_multiclass.ckpt` (a v3 variant trained
            specifically for multiclass — our task is 3-class). Only the `model_path` / ckpt
            file changes; context size (50k), K=8 subsamples, RNG seeding, OOF/test interface,
            and the encode-once + one-call batching all stay identical.
hypothesis: the newer v3 multiclass-specialized foundation model may be stronger and/or more
            de-correlated than v2.5 — the last untested foundation-model base.
target:     BA maximize; valued for de-correlation in the stack; drop if solo < 0.945 AND it
            doesn't help the re-stack.
