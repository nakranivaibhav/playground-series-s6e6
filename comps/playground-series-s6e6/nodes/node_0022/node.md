---
id: node_0022
desc: TabPFN-3 base (subsample ensemble)
op: draft
parents: [root]
uses_data: [fs_research]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.942629
sem: 0.000350
folds: [0.941581, 0.942662, 0.942153, 0.943456, 0.943296]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warn (0.943 vs baseline 0.333) — same pattern as other nodes; eyeball before first submit"
leak: clean
lb: null
submitted: null
created: 2026-06-07T06:06Z
decided: 2026-06-07T07:02Z
tags: [nn, tabpfn, in-context, cuda, diversity-arm, draft]
---

## plan
built on:   root — new NN base family. Template src copied from node_0009 (NN harness:
            fs_research load, frozen folds.json loop, fold-honest OOF + test_probs interface).
            Keep the data-loading/fold scaffolding; swap only the model + prediction routine.
change:     add `tabpfn` (`uv add tabpfn`). 247k rows >> TabPFN context, so per train-fold
            predict val by ensembling K≈8 class-stratified subsamples of ≤10k rows drawn ONLY
            from that fold's TRAIN indices (label-free context = fit_in_fold), averaging the
            softmax outputs. Test predicted the same way (subsamples from train fold). Fix the
            subsample RNG seed per fold for reproducibility. Produce oof.npy + test_probs.npy.
            GPU, runs serialized AFTER node_0021.
hypothesis: an in-context Bayesian predictor has an error pattern unlike the trees / TabM /
            RealMLP → a de-correlated stack column.
target:     BA maximize, solo ≥ ~0.960, leak-clean; drop if solo < 0.955.
