---
id: node_0112
desc: Broad revival re-stack all never-pooled bases
op: combine
parents: [node_0091, node_0014, node_0071, node_0075, node_0086, node_0087, node_0090, node_0094, node_0098]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970355
sem: 0.000249
folds: [0.971208, 0.970067, 0.969934, 0.969938, 0.970626]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "NO STUDENT KEPT (wash). Broad forward-select of all 8 never-pooled single-model bases (n014 own-FT-T, n071 5-seed DCN, n075 5-seed TabM, n086 z-resid TabM, n087 z-resid LGBM, n090 OvR chain, n094 error-pocket GBDT, n098 RBF) onto the n091 FULL pool @C0.003. Baseline reproduced EXACTLY (0.970355). Per-candidate deltas all negative (e.g. +n014 −0.000051, +n094 −0.000031); none cleared the +1·sem keep-bar → final pool == n091, lift 0.000000. Combined with n105 (3 distill students, none kept), EVERY valid never-pooled base in the run history has now been tested against the pool — none helps. The 'uncombined history' question is definitively closed; n091's pool is the complete useful set."
leak: clean
lb: null
submitted: null
created: 2026-06-15T11:33Z
decided: 2026-06-15
---

## plan
built on:   champion n091 FULL pool @C=0.003 (n099 loader, verbatim). Human directive: "keep combining
            models from previous runs if not already."
change:     BROAD forward-select of EVERY valid never-pooled SINGLE-MODEL base (not derivative stacks)
            onto the n091 pool under C=0.003: n014 (our own FT-Transformer), n071 (5-seed DCN), n075
            (5-seed TabM), n086 (z-resid TabM), n087 (z-resid LGBM), n090 (OvR chain), n094 (error-pocket
            GBDT), n098 (RBF). n105 already tested the 3 distill students (none kept); this covers the rest.
hypothesis: one of these bases — excluded during n091's construction under the OLD regime — may net positive
            under C=0.003 shrinkage (which revived weak-but-independent bases). Most are known-correlated
            (n086/090 ~0.72) or weak (n098 0.95), so honest EV is low; this is the exhaustive sweep the
            director asked for, closing the "uncombined previous runs" question definitively.
target:     BA maximize · keep any base whose forward-select add lifts OOF CV > 1·sem; promote final pool
            only if it beats 0.970355 by > 2·sem.

HOW: reuse n099/n105 full-pool loader + nested_cv_arm_logreg VERBATIM. Reproduce n091 baseline (0.970355)
first. Greedy forward-select over the 8 candidates; stop when best add ≤ 1·sem. Saved OOF only, no retrain.

## notes
well=exploit/revival. Derivative stacks (n063/64/69/70/72/76/77/78/80/88/89/99/100/102/104/105) deliberately
EXCLUDED — adding a stack-of-the-pool back to the pool is circular (no independent signal; journal lesson).
