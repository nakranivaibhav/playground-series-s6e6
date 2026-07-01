---
id: node_0025
desc: TabPFN-2.5 @ 50k context
op: improve
parents: [node_0022]
uses_data: [fs_research]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.948957
sem: 0.000368
folds: [0.947879, 0.949537, 0.949771, 0.949301, 0.948297]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warns vs random baseline 0.333 (expected for this competition — all nodes score 0.94+); passes vs parent cv 0.943; warn only, not a blocker."
leak: clean
lb: null
submitted: null
created: 2026-06-07T07:18Z
decided: 2026-06-07T09:28Z
tags: [nn, tabpfn, tabpfn-2.5, in-context, cuda, diversity-arm, improve]
---

## plan
built on:   node_0022 (TabPFN base, subsample ensemble). Keep the fs_research load, frozen
            folds.json loop, and fold-honest OOF + test_probs interface. Template src copied
            from node_0022/src.
change:     upgrade to TabPFN-2.5 and use the FULL 50k context (5× the old 10k) with
            class-balanced stratified subsample-bagging — n_estimators round-robin so every train
            row is covered; `ignore_pretraining_limits=True`. Contexts drawn ONLY from each
            fold's TRAIN indices (fit_in_fold, label-free); the test pool is full train, never
            test rows. Dep: `uv add tabpfn` (+ tabpfn-extensions if needed; ensure
            torch-compatible build). Fixed subsample RNG seed per fold. Produce oof.npy
            (577347×3) + test_probs.npy (247435×3). GPU, serialized.
            CRITICAL BATCH FIX: predict the ENTIRE query block in ONE call per subsample
            (not a small-batch loop that re-encodes context repeatedly). OOM fallback halves
            chunk size from the full block (floor at 10k rows, never 512).
hypothesis: 5× context + class-balanced bagging recovers TabPFN from the context-starved 0.943
            toward the 0.96 band, giving a strong de-correlated foundation-model stack column.
target:     BA maximize, solo ≥ ~0.960, leak-clean; drop if solo < 0.955.

## notes
cv=0.948957 ± 0.000368 vs node_0022 parent cv ~0.943 — a genuine +0.006 lift from the v2.5
large-samples checkpoint + 5x context. The 0.955 solo target was not met (0.949 actual),
but this is a clean improvement over the parent and can contribute to blends.

Timing:
- 1 subsample (50k ctx + 115k val queries, one call): ~11.1s
- 1 fold (8 subsamples): ~89s
- 5 folds OOF: ~445s (~7.4 min)
- test set (50k ctx + 247k test queries, 8 subsamples): ~162s
- Total: ~607s (~10.1 min) on RTX 5090

Peak VRAM: 5.24 GB (50k context + 115k query block in one forward pass, well within 32 GB)

Checkpoint downloaded from Prior-Labs/tabpfn_2_5 (HuggingFace): 42.9 MB
Cached at: ~/.cache/tabpfn/tabpfn-v2.5-classifier-v2.5_large-samples.ckpt
