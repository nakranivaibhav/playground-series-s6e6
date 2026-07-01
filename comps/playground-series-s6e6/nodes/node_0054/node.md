---
id: node_0054
desc: 5-seed 5fold control stack (frozen-anchored)
op: improve
parents: [node_0041]
uses_data: []
family: ensemble
status: dead
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.969808
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null, leak_clean: null, cv_too_good: null, passed: null}
gate_note: null
leak: null
lb: null
submitted: null
created: 2026-06-09
decided: null
tags: [stack, multi-seed, control, diagnostic]
---

## plan
built on:   node_0041 CORE+CatBoost 15-base stack (bases byte-identical) + DE threshold meta. Same 15 base OOF/test log-probs.
change:     5-seed N_FOLDS=5 control: re-fit the balanced-multinomial LogReg meta over 5 seeds, but anchor seed-42's
            partition to the frozen folds.json (never re-make folds). Average stacked OOF/test over the 5 partitions;
            DE threshold on the averaged OOF. Serves as the N_FOLDS=5 diagnostic against node_0053's N_FOLDS=10 —
            isolates true variance-reduction from fold-count contamination.
hypothesis: if node_0053's lift is real variance-reduction it should appear here too at 5 folds; if it only appears at
            10 folds it is fold-count contamination, not signal.
target:     BA maximize · promote/submit only if CV > 0.969808 by >1·sem AND it holds on the untouched holdout;
            LB-gate before trusting (node_0047 mirage precedent).

## result — SKIPPED (diagnostic moot)
This node's only purpose was to diagnose whether node_0053's multi-seed lift was real vs meta-contamination. node_0053 showed NO lift (+0.000037, within noise) and its untouched-holdout BA matched single-seed (0.969970 vs 0.969963) — honesty already confirmed, no lift to diagnose. Not built; marked dead to avoid wasted compute.
