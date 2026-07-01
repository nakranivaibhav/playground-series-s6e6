---
id: node_0014
desc: FT-Transformer 2nd strong NN arm
op: draft
parents: [root]
uses_data: [fs_colors, fs_research]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.957317
sem: 0.000369
folds: [0.957882, 0.957887, 0.956939, 0.956032, 0.957847]
baseline_cv: 0.333333
shuffled_cv: 0.33333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warn: large jump from 0.333 baseline expected for this task; human-eyeball before submit"
leak: clean
lb: null
submitted: null
created: 2026-06-06T08:49Z
decided: 2026-06-06T12:06Z
tags: [nn, ft-transformer, rtdl, cuda, diversity-arm]
---

## plan
built on:   root — a SECOND strong tabular-NN draft, structurally distinct from the TabM
            arm (node_0009). Copy node_0009's feature-prep and CV/OOF plumbing byte-identical;
            only the model class changes (TabM → FT-Transformer).
change:     swap the model to an FT-Transformer (rtdl `FTTransformer`, library-first per
            rule 8) on the SAME inputs as node_0009: 26 numerical feats (standardized
            fit-inside-fold) + 2 native categorical bins (cat_cardinalities, no one-hot).
            Paper-default depth/heads/d_token, CUDA (RTX 5090), same frozen 5 folds. Save
            oof.npy + test_probs.npy. Reuse fs_research; no new feature-set.
hypothesis: an attention-based tabular NN reaches GBDT/TabM-level solo yet errs de-correlated
            from BOTH the trees and TabM → a 5th de-correlated blend arm that lifts node_0010.
target:     Balanced Accuracy (maximize) · solo cv ≥ ~0.962 AND added to the n6/n4/n1/n9 blend
            it beats node_0010 (0.965889) beyond sem.
