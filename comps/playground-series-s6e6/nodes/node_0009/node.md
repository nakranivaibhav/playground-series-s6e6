---
id: node_0009
desc: TabM (library) on engineered feats
op: draft
parents: [root]
family: nn
uses_data: [fs_colors, fs_research]
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.964215
sem: 0.000374
folds: [0.964402, 0.962802, 0.965039, 0.964408, 0.964424]
baseline_cv: 0.333333
shuffled_cv: 0.33333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-05T14:53Z
decided: null
tags: [nn, tabm, library, rtdl-num-embeddings, cuda, diversity-arm]
---

## plan
built on:   root — second non-GBDT draft. node_0008 (plain MLP) was the most de-correlated
            arm but too weak solo (0.9550) to lift the blend. TabM is a much stronger tabular
            NN; the hypothesis is it closes most of the accuracy gap so its diversity pays.
change:     use the OFFICIAL `tabm` library (ICLR 2025, parameter-efficient deep ensemble,
            k=32) + `rtdl_num_embeddings` target-aware PiecewiseLinearEmbeddings — NOT a
            hand-rolled net (per CLAUDE.md rule 8). 26 numerical feats (22 cont standardized
            fit-inside-fold + 4 flags) via x_num; 2 categorical bins via cat_cardinalities=[4,2]
            (native categorical embeddings, no one-hot). Bins from compute_bins() on the TRAIN
            FOLD only (fit-inside-fold). TabM.make() paper-tuned defaults. CUDA (RTX 5090).
hypothesis: TabM ≈ GBDT-level solo (~0.963-0.965) AND de-correlated → finally lifts the blend.
target:     solo CV high enough (≥ ~0.962) that adding it beats champion node_0007 (0.965530);
            real test = does the 4/5-arm combine OOF beat 0.965530 beyond sem.

## notes
SUCCESS as a blend arm. Solo CV 0.964215 ± 0.000374 — GBDT-strength (vs node_0008 MLP 0.9550),
clean (shuffled 0.33333), via the official `tabm` library (k=32 ensemble + PiecewiseLinear
embeddings). De-correlated: err-corr ~0.82 vs the trees (they sit at 0.85–0.87). In the
fold-honest search it earns the LARGEST weight (0.35) and lifts the blend to 0.965889
(node_0010, new champion, +0.000359 vs node_0007 > 2·sem). The diversity-AND-strength combo
node_0008 lacked. Diagnostic: src/analyze_blend.py · blend_analysis.log.
