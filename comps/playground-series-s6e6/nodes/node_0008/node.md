---
id: node_0008
desc: MLP (CUDA) on engineered feats
op: draft
parents: [root]
family: nn
uses_data: [fs_colors, fs_research]
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.954969
sem: 0.000315
folds: [0.955425, 0.955669, 0.954312, 0.954113, 0.955325]
baseline_cv: 0.333333
shuffled_cv: 0.33333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-05T14:30Z
decided: 2026-06-05T14:42Z
tags: [nn, mlp, pytorch, cuda, diversity-arm, blend-no-help]
---

## plan
built on:   root — first non-GBDT family (the methodology pivot after the GBDT lineage
            tapped out). Reuses the same leak-safe engineered features as node_0006.
change:     a PyTorch MLP trained on CUDA (RTX 5090). Inputs: 22 continuous feats
            standardized FIT-INSIDE-FOLD (mean/std from the train fold only), 4 binary
            flags as-is, 2 fixed-category bins one-hot (stateless → leak-safe). Arch
            [256,128,64] + BatchNorm + Dropout(0.2), class-weighted cross-entropy (balanced),
            Adam, early-stop on an internal 10% split (never the official val fold).
hypothesis: an MLP makes structurally different errors from GBDTs; even at lower solo CV it
            can be a de-correlated arm that lifts the blend (node_0007).
target:     primarily a diversity arm for the blend; solo CV need only be respectable
            (a single GBDT is ~0.9646). Real test = does it improve the combine OOF.

## notes
HONEST NEGATIVE (for the blend). Solo CV 0.954969 ± 0.000315 — clean, leak-free (shuffled
0.33333), trained on CUDA (RTX 5090). It IS the most de-correlated arm: error-corr 0.71–0.73
vs the GBDTs (which sit at 0.85–0.87 with each other). BUT its diversity does not overcome
the ~0.010 solo gap: in the fold-honest nested search it earns only 0.05 weight, and the
4-arm blend n6+n4+n1+n8 = 0.965526 ± 0.000186 — a hair BELOW champion node_0007 (0.965530),
not beyond sem. So node_0009 was NOT built; node_0007 (n6+n4+n1) stays champion. Kept as a
valid, logged-but-unused arm. Lesson: a diverse arm only helps a blend if it's also strong
enough solo — corr alone isn't sufficient. To make an NN useful here it would need to close
most of the accuracy gap (e.g. TabM / a tuned deeper net), which is speculative.
Diagnostic: src/analyze_blend.py · blend_analysis.log.
