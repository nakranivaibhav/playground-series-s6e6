---
id: node_0046
desc: pseudo-label self-train (GBDT bases)
op: improve
parents: [node_0041]
uses_data: [fs_realmlp_fe]
family: gbdt
status: dead
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.969808
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: null, cv_too_good: null, passed: null}
gate_note: null
leak: null
lb: null
submitted: null
created: 2026-06-08T16:11Z
decided: null
tags: [pseudo-label, self-training, in-distribution]
---

## plan
built on:   champion stack node_0041 (its test predictions provide the pseudo-labels).
change:     In-distribution self-training. Use the champion's test predictions to pick
            HIGH-CONFIDENCE test rows (e.g. max class prob ≥ 0.99, class-balanced cap),
            treat them as extra labeled training rows, and retrain our GBDT bases
            (LightGBM node_0030 recipe + CatBoost node_0039 recipe on fs_realmlp_fe) with
            train_fold + pseudo-test appended. Val folds stay PURE real train rows.
hypothesis: The test set is the SAME synthetic distribution as train (no drift, unlike
            SDSS17), so confident pseudo-labels add in-distribution density that sharpens
            the dense GALAXY/STAR boundary (47% of our errors) → stronger, partly-fresh
            bases that lift the stack.
target:     re-stack with pseudo-labeled bases swapped in beats champion 0.969808 by >2·sem.

## notes
Leakage care: pseudo-labels live ONLY on test rows (no true labels there) — never on
train/val; OOF computed on real train folds only. For rigor prefer per-fold pseudo-labels
(label test with the fold's own model) to avoid champion-uses-full-train optimism; if using
fixed champion test preds, flag the mild optimism. Self-gate the leakage suite.

## result — WASH
LightGBM pseudo-label base built (71,370 balanced conf≥0.99 pseudo-rows appended to each
train fold, val pure), solo CV **0.967125** — squarely in the existing GBDT band, leak-clean.
Re-stack into CORE15: **0.969714 = −0.0001 vs champion** (slight regress — it's just another
correlated GBDT the stack already spans). On top of the n47 specialist stack it adds only
+0.000033 (~0.1σ, pure noise). CatBoost half was killed mid-fold-1 by the session interruption
and not resumed — the LightGBM verdict already showed in-distribution pseudo-labeling washes
for our saturated stack. Not promoted. (Distinct from external SDSS17, which HURT via drift —
here the pseudo-rows are in-distribution but simply redundant with existing bases.)
