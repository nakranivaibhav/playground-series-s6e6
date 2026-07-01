---
id: node_0016
desc: TabM with balanced class weighting
op: improve
parents: [node_0009]
uses_data: [fs_colors, fs_research]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.964378
sem: 0.000178
folds: [0.964244, 0.963890, 0.964826, 0.964180, 0.964751]
baseline_cv: 0.964215
shuffled_cv: 0.33392
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: "outlier-fold warn: only 2/5 folds improved; fold-1 gain (+0.00109) dominates, 3 folds regressed — CV delta is fragile"
leak: clean
lb: null
submitted: null
created: 2026-06-06T08:49Z
decided: 2026-06-06T12:06Z
tags: [nn, tabm, class-weight, balanced-loss, cuda]
---

## plan
built on:   node_0009 (TabM, library) — byte-identical feature-prep, model config, folds,
            and OOF/test plumbing; only the training loss weighting changes.
change:     ONE atomic change — unweighted cross-entropy → balanced-weighted cross-entropy.
            Weight the loss by inverse class frequency computed on the TRAIN FOLD only
            (fit-inside-fold), fed as per-sample weights into the cross-entropy. Same
            fs_research feats, same 5 frozen folds. Save oof.npy + test_probs.npy.
hypothesis: Balanced Accuracy weights the 3 classes equally despite the 65/20/14 imbalance;
            class-weighting raises minority-class recall and lifts TabM solo above 0.964215.
target:     Balanced Accuracy (maximize) · solo cv > node_0009 (0.964215) beyond sem, and the
            re-weighted TabM as a blend arm beats node_0010.
