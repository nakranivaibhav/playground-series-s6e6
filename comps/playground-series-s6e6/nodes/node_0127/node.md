---
id: node_0127
desc: MoE-gated 2-expert mixture meta
op: improve
parents: [node_0091]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970420
sem: 0.000316
folds: [0.971594, 0.970065, 0.969987, 0.969883, 0.970570]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-18
decided: 2026-06-18T01:40Z
tags: [stack, meta, mixture-of-experts, gated, outside, improve, structural-gate]
---

## plan
built on:   node_0091 (champion C0.003 balanced multinomial LogReg over the 63-base clipped-log-prob
            pool, nested in-fold C, frozen folds, argmax). Same pool, same target, same folds.
change:     Replace the single global LogReg meta with a **2-expert GATED MIXTURE**: a small learned
            softmax GATE g(x) over a few raw inputs (redshift, u−g, g−r) routes each row between TWO
            balanced-multinomial LogReg experts over the SAME base log-prob pool, fit jointly (EM, or a
            tiny torch module), nested fold-honest at C=0.003. Final prob = sum_k g_k(x)·softmax(expert_k).
            ONE atomic change: the meta functional form (global linear → 2-expert learned routing);
            pool/target/folds unchanged.
hypothesis: the 12 holdout-significant complementary fix-blocks are real but row-entangled under
            ADDITIVE (band×base) interaction (n122 washed, kept coefs but didn't generalize). A
            MULTIPLICATIVE learned gate can find a soft input-space partition where fix/break rows
            separate — IF such a partition exists. This is the one meta form never tried (n100/n122
            only added interaction columns to one global fit); the correct test of "is the entanglement
            separable by a LEARNED boundary?"
target:     Balanced Accuracy maximize. SANITY: with the gate frozen uniform, must reproduce n091
            (~0.97030); if 2-expert fold-0 CV < n091 − 1·sem, cheap-kill before full 5-fold. GATE
            (validation.md): tools/pred_diagnostic.py bootstrap P(>n091) ≥ 0.90 AND a HOLDOUT-confirmed
            net-positive fix-block (n122/n0047 guard — working-only improvement is NOT enough, the
            holdout must move). HONEST ODDS LOW (~10-15%): the capturability_check + NCL-cliff priors
            say no partition separates fix from break; but it's the definitive close of the
            learnably-separable question, and cheap (CPU minutes).

OUTSIDE well (functional-form gap). READ: champion/src/solution.py (the meta to fork); nodes/node_0122
results (the additive-interaction wash — why multiplicative is the distinct test); probes/
capturability_check.py + hidden_signal_sweep.csv (the entanglement evidence). CPU; minutes; uv run --no-sync.

## notes
well = outside. The one un-run meta functional form: learned routing, not added interaction columns.
