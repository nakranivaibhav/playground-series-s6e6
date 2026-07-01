---
id: node_0062
desc: swap-noise DAE rep + MLP head (C9)
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: scored
metric: Balanced Accuracy Score
direction: maximize
cv: 0.958002
sem: null
folds: [0.957848, 0.958532, 0.958170, 0.957079, 0.958380]
baseline_cv: null
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-10
decided: 2026-06-10
tags: [data-centric, representation, DAE, self-supervised, null]
---

## plan
built on:   root (new representation family).
change:     swap-noise denoising autoencoder fit per-train-fold (no labels/test), concat hidden reps -> balanced MLP head -> OOF+test.
hypothesis: a learned denoising manifold is a genuinely new representation family that could de-correlate from supervised bases.
target:     clear ~0.964 standalone to earn a restack; beat champion via restack.

## notes
Solo 0.958002 << 0.964 floor. No restack (weak base hurts saturated stack). Rich FE is 26-dim with no within-row manifold → DAE re-encodes known features at lower fidelity. Self-supervised representation family CLOSED. Fold-honest, leak-clean.
