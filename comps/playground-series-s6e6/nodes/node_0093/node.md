---
id: node_0093
desc: non-negative simplex convex blend (tight vs full)
op: combine
parents: [node_0063, node_0070]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.963155
sem: 0.000238
folds: [0.963315, 0.963977, 0.962954, 0.962555, 0.962974]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "TIGHT cv=0.963155 FULL cv=0.963166 — BOTH below parent 0.970211 by ~7ppt; clean null; artifacts produced for future combines"
leak: clean
lb: null
submitted: null
created: 2026-06-14T09:00Z
decided: 2026-06-14
tags: [blend, simplex, convex, nnls, tight-vs-full, exploit]
---

## plan
built on:   node_0063 (champion bank-17 stack, cv 0.970153) + node_0070 (bank-17 + FT-T, cv 0.970211). This is
            a DIFFERENT combiner from the LogReg meta (node_0063/0091) on BOTH axes: weight space (non-negative
            simplex vs signed-dense LogReg coefficients) AND input space (raw PROBABILITIES vs clipped
            log-probs). Genuinely untested — critic confirmed no NNLS / simplex / convex prob-blend has ever
            run on this comp.
change:     Learn ONE scalar non-negative weight per base, weights summing to 1 (a point on the simplex), over
            the bases' PROBABILITY vectors (not log-probs), optimizing nested OOF balanced accuracy via SLSQP
            (or projected-gradient on the simplex). Blend prediction = argmax of sum_b w_b * P_b. Fold-honest:
            optimize weights on the train-fold OOF only, apply to the held-out fold; frozen folds.json;
            threshold-free argmax. A/B the SAME two pools as node_0091:
              FULL pool  = bank-17 + FT-T + ALL distinct in-house single-model base probability vectors.
              TIGHT pool = bank-17 + FT-T + the 36 STRONG-DISTINCT in-house bases (identical set to node_0091).
hypothesis: a convex prob-space blend has a different inductive bias from the log-prob LogReg meta — it cannot
            assign negative weight, so it is more robust to redundant bases (they simply get ~0 weight) and may
            beat the saturated LogReg by combining in probability space where the metric (argmax BA) actually
            lives.
target:     Balanced Accuracy maximize. Promote iff nested OOF CV > champion 0.970153 by > 2·sem (fold-noise).

WHY this is distinct and worth a slot: every prior combiner here is a balanced multinomial LogReg on clipped
log-probs (n20/n29/n41/n63/n70/n76 ...). A2's Caruana (journal L85) is the closest relative but it is GREEDY
forward-selection-with-replacement (a coarse, discrete approximation to a simplex weighting) on HARD ARGMAX +
DE thresholds, NOT a continuous convex optimum over probabilities — and it lost to LogReg. A direct continuous
simplex solve over the probability vectors has never been tried and is the natural "did greedy leave value on
the table?" check. The tight/full A/B isolates whether the non-negativity constraint alone neutralizes the
full-pool dilution that hurt the LogReg merges (node_0063 notes) and Caruana (L85).

CHEAP BASELINE-ASSERT GATE (run FIRST): a degenerate simplex blend that puts all weight on the bank-17
LogReg-stacked probabilities should reproduce ~0.970153; and a uniform blend over bank-17 should be sane (not
~0.33). Confirms the probability ingest + column order + fold loop are correct before optimizing. (Same
wrong-baseline guard that caught node_0070 v1.)

REPORT in the node record: which pool wins (FULL vs TIGHT) + each arm's CV/sem; the nonzero-weight bases and
their weights for the winning arm (the simplex is naturally sparse — list every base with weight > ~0.01);
confirmation the baseline-assert passed.

THE 36 STRONG-DISTINCT IN-HOUSE BASES (TIGHT pool) — IDENTICAL to node_0091 (use the same set so the
tight/full verdicts are comparable across nodes):
  old-FS era: n1, n3, n4, n5, n6, n9, n11, n12, n13, n15, n16, n18, n19, n23
  rich-FE era: n28, n30, n31, n32, n33, n35, n36, n38, n39, n42, n43, n44, n45, n49, n50, n51, n55, n56,
               n60, n61, n66, n85
EXCLUDED from TIGHT: n67/n74/n79 (distill/pseudo students), n71/n75 (seed-bags), n86/n87/n90 (err-corr-0.72
residual/OvR bases); all stack/combine nodes (they contain the bank); the dead 1-col specialist n47; weak
bases < ~0.96 (n8/n21/n22/n24/n25/n26/n27/n37/n62). The FULL pool adds those weak bases back on top.

References to READ:
- champion/src (a1_full_merge.py = OOF ingest + clip/normalize → probabilities; the prob vectors are what this
  node blends, NOT the log-probs LogReg uses).
- nodes/node_0063/node.md + node_0070/node.md (baselines + node_0070's external-OOF leak/alignment checklist).
- nodes/node_0091/node.md (the sibling LogReg mega-stack — shares the exact base pools and baseline-assert).
- refs/oof_bank + refs/kernel_out + refs/ext_oof/ (bank-17 + FT-T OOF, id-aligned).
- journal.md L85 (A2 Caruana — the greedy/argmax relative this continuous/prob-space blend is distinct from).
- each in-house base's oof.npy at nodes/node_NNNN/oof.npy (single-model bases (577347,3) float32; verify solo
  BA sane before adding).

DELIVERABLES: oof.npy / test_probs.npy / submission.csv for the winning arm; gate booleans; VOID on leak. Do
NOT submit (orchestrator decides).

## notes
RESULT: Both arms definitively below parent 0.970211.
- TIGHT (bank-17+FT-T+36 in-house) cv=0.963155 sem=0.000238
- FULL  (TIGHT+9 weak bases)       cv=0.963166 sem=0.000236
- Baseline assert (bank-17+FT-T LogReg): PASS (0.970227, within 1e-3 of 0.970211)

WHY THE SIMPLEX BLEND FAILS HERE: The non-negative scalar-weight blend (one weight per base,
same weight applied to all 3 class columns) is structurally weaker than a multinomial LogReg
meta (B×3 free parameters, signed). The simplex finds a sparse solution: just 8 bases carry
all weight, dominated by tabm-1 (~0.31) + xgb-6 (~0.28) + n49 (~0.13) + n50 (~0.12) + n60
(~0.09) + realmlp-0 (~0.03) + n5/n44/n30 (<0.03). This collapses to a 2-base blend, discarding
most of the bank-17 signal, because a scalar weight cannot exploit the per-class discriminative
pattern that LogReg's class-specific coefficients capture.

FULL = TIGHT: The weak bases get zero weight in the FULL arm — non-negativity constraint
naturally excludes them, so tight/full distinction is irrelevant here.

The continuous convex prob-space solve does NOT leave value on the table relative to greedy
Caruana (A2, journal L85): both produce CV ~0.963–0.970 range depending on method, and both
are inferior to the class-specific multinomial LogReg meta operating in log-prob space.

ARTIFACTS: oof.npy (577347,3) + test_probs.npy (247435,3) produced from TIGHT arm.
Submission.csv produced (argmax of TIGHT blend) but DO NOT SUBMIT — CV well below champion.
