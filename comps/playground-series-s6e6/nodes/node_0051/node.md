---
id: node_0051
desc: FT-Transformer on fs_realmlp_fe
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966948
sem: 0.000157
folds: [0.966415, 0.967294, 0.967222, 0.966837, 0.966970]
baseline_cv: 0.969808
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-09
decided: null
tags: [nn, ft-transformer, rtdl, re-stack-candidate]
---

## plan
built on:   root — a fresh FT-Transformer NN arm, but trained on the strong rich
            FE set `fs_realmlp_fe` (the breakthrough feature-set that lifted every
            base from ~0.949 to ~0.969), instead of the old `fs_research` set the
            original FT-T (node_0014) used.
change:     Re-train an FT-Transformer (rtdl / rtdl_num_embeddings library) on
            `fs_realmlp_fe`. Same OOF-on-folds protocol as the other rich-FE bases
            (n28/n33). Goal is a de-correlated strong NN base to feed the CORE stack.
            One atomic change vs node_0014: swap the feature-set fs_research →
            fs_realmlp_fe (and retune capacity to the richer input).
hypothesis: node_0014's FT-T capped at 0.957317 ONLY because it ran on the weaker
            fs_research; on fs_realmlp_fe the same family should reach the ~0.969
            band like RealMLP/TabM did, giving a new de-correlated NN arm for the stack.
target:     Balanced Accuracy (maximize). Standalone bar = old FT-T 0.957317 (must
            clear it decisively); stack bar = lifts the CORE15 stack above champion
            baseline 0.969808.

## notes
Standalone argmax cv=0.966854 ± 0.000126; DE-threshold cv=0.966948 ± 0.000157.
Massively beats old FT-T (node_0014) 0.957317 — confirming fs_research was the bottleneck.
Reaches the NN band (TabM node_0033=0.968053), stopping at ~0.967 level.

re-stack A/B: CORE15+n51 = 0.969680 vs champ 0.969808 — does NOT lift the stack (delta=-0.000128 < 2*sem ~0.0005).
FT-T on fs_realmlp_fe is a strong standalone base but adds no net value to the existing CORE15 stack.
All 5 folds ~1430s total (~24 min). Early stopping at 29-57 epochs (of 80 max). VRAM peak ~12.8GB on RTX 5090.
