---
id: node_0091
desc: L2-LogReg mega-stack (tight vs full)
op: combine
parents: [node_0063, node_0070]
uses_data: []
family: ensemble
status: champion
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970355
sem: 0.000249
folds: [0.971208, 0.970067, 0.969934, 0.969938, 0.970626]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.97121
submitted: 2026-06-14
created: 2026-06-14T09:00Z
decided: 2026-06-14
tags: [stack, mega-pool, l2-shrinkage, tight-vs-full, exploit]
---

## plan
built on:   node_0063 (champion: bank-17 balanced multinomial LogReg on clipped log-probs, cv 0.970153)
            + node_0070 (bank-17 + FT-Transformer ext base, cv 0.970211 — the best honest stack). The meta
            FAMILY (balanced multinomial LogReg on clipped log-probs) stays; what changes is (a) the
            REGULARIZATION (nested in-fold C grid, not the fixed C=0.1 the champion uses) and (b) the BASE
            POOL it sees — fit on TWO pools and report both.
change:     Fit an L2-regularized, class-balanced multinomial LogReg meta with a NESTED in-fold C grid (select
            C per outer fold on inner CV — never on the outer val fold) over two base pools, A/B them:
              FULL pool  = bank-17 + FT-T + ALL distinct in-house single-model base OOF (~213 cols).
              TIGHT pool = bank-17 + FT-T + only the 36 STRONG-DISTINCT in-house bases listed below.
            Headline CV = the better of the two arms. Everything else (clipped log-probs ±30, balanced class
            weight, fold-honest nested fit, frozen folds.json, threshold-free argmax scoring) byte-identical
            to the champion recipe.
hypothesis: with strong L2 shrinkage the meta can absorb the FULL in-house pool WITHOUT the dilution A2 saw
            (shrinkage zeroes the redundant bases), so the union of bank-17 + FT-T + in-house could clear the
            frontier; the TIGHT arm protects attribution if shrinkage is not enough.
target:     Balanced Accuracy maximize. Promote iff headline CV > champion 0.970153 by > 2·sem (fold-noise).

This is the human's "let shrinkage sort it" bet, run as a controlled A/B against the attribution-safe tight
pool. It is genuinely worth running because the two prior full-pool merges DILUTED at a FIXED meta strength —
the open question is whether REGULARIZATION strength (never varied on this stack — MEMORY L45/L49 "stop tuning
the meta" was about config/seed/loop and the COLUMN set was closed by B1/n80, but the C grid on a 200-col pool
was never swept) changes that verdict. Record explicitly which pool wins, the C-vs-CV curve, and the 5–10
largest |coef| bases per class.

WHY both arms (rationale to record): A2 (journal L85, 2026-06-10) ran bagged Caruana with-replacement over a
58-model pool (our 41 + bank-17) maximizing hard-argmax BA + per-fold DE thresholds → CV 0.969950, BELOW
bank-17 LogReg 0.970153. node_0063's own notes show every full-pool LogReg merge dilutes: 15+17 = 0.970025,
bank17+4uniq = 0.970128, bank17+tabm = 0.970030 — all < bank-17 0.970153. So the FULL arm is a real test of the
shrinkage hypothesis (does C-tuning rescue what fixed-C merges lost?), and the TIGHT arm is the
attribution-protected control (strong bases only, no known-redundant adds). If FULL beats TIGHT and both beat
champ, shrinkage won; if TIGHT >= FULL, the redundant bases still hurt even under shrinkage.

CHEAP BASELINE-ASSERT GATE (run FIRST, before any C sweep — costs seconds): reproduce the bank-17 + FT-T
reference stack and confirm CV ≈ 0.970211 (node_0070). If it does not reproduce within noise, the OOF ingest /
column-order / clip-norm is wrong — STOP and fix before sweeping C. (This is the same misbuild guard that
caught node_0070 v1 fitting on the wrong baseline — journal 2026-06-12T14:33Z.)

THE 36 STRONG-DISTINCT IN-HOUSE BASES (TIGHT pool) — solo OOF BA >= ~0.96 AND not in the known-redundant
exclusion set. USE EXACTLY THIS SET (the same set is fixed across node_0091/0093/0094 so verdicts are
comparable):
  old-FS era: n1, n3, n4, n5, n6, n9, n11, n12, n13, n15, n16, n18, n19, n23
  rich-FE era: n28, n30, n31, n32, n33, n35, n36, n38, n39, n42, n43, n44, n45, n49, n50, n51, n55, n56,
               n60, n61, n66, n85
EXCLUDED from TIGHT (the known-redundant set, by the decision rule): n67/n74/n79 (distill/pseudo-label
students — too correlated with the bank they learned from), n71/n75 (seed-bags — confirmed wash), n86/n87/n90
(err-corr-0.72 residual/OvR bases — not decorrelated). Also excluded as bases: every stack/combine node
(n7/n10/n17/n20/n29/n40/n41/n52/n53/n63/n64/n69/n70/n72/n76/n77/n78/n80/n84/n88/n89 — they already CONTAIN the
bank, would double-count), the dead 1-col specialist n47 (CV mirage, permanently excluded), and weak bases
< ~0.96 (n8 0.955, n21 0.950, n22 0.943, n24 0.949, n25 0.949, n26 0.959, n27 0.941, n37 0.939, n62 0.958).
The FULL pool ADDS those weak bases (and any seed-bag/variant OOF) back on top of the tight set.

References to READ:
- champion/src (a1_full_merge.py = OOF ingest + clip+normalize pattern; a1_submit.py = the balanced-LogReg
  meta fit) — the canonical recipe to copy and extend with the C grid.
- nodes/node_0063/node.md + node_0070/node.md (the bank-17 and bank-17+FT-T baselines; node_0070's CRITICAL
  leak/alignment checklist for external OOF applies to the bank columns here too).
- refs/oof_bank + refs/kernel_out + refs/ext_oof/ (the public bank-17 OOF + FT-T OOF, all id-aligned to our
  folds, n_train 577347 / n_test 247435).
- journal.md L85 (A2 Caruana null), 2026-06-10 A1 notes (full-pool dilution numbers), 2026-06-12T14:33Z
  (the node_0070 v1 wrong-baseline misbuild — why the baseline-assert gate exists).
- each in-house base's oof.npy at nodes/node_NNNN/oof.npy (single-model bases are (577347,3) float32; verify
  each loads and its solo OOF BA is sane, not ~0.33, before adding it).

DELIVERABLES in the node record + train.log: which pool wins (FULL vs TIGHT) + each arm's CV/sem; the
C-vs-CV curve for the winning pool; the 5–10 largest-|coef| bases per class; confirmation the baseline-assert
(bank-17+FT-T ≈ 0.970211) passed. Produce oof.npy / test_probs.npy / submission.csv for the winning arm.
Write the gate booleans; VOID on any leak. Do NOT submit (orchestrator decides).

## notes
RESULT (2026-06-14): FULL pool wins over TIGHT (0.970355 vs 0.970301). Shrinkage hypothesis
partially confirmed: the FULL pool (all weak bases included) beats TIGHT with strong L2 (C=0.003).
However, both are below the promote bar (0.970597 = champ 0.970153 + 2*0.000222). Both beat
node_0070 (0.970211). The finding: L2 shrinkage at C=0.003 does help absorb the weak bases, but
not enough to clear the frontier. The nested C-selection consistently prefers C=0.003 (very strong
regularization) across all folds.

C-vs-CV curve (FULL arm, inner 4-fold BA means by fold):
- fold 0: C=0.003 wins (0.96994 vs 0.96993 at C=0.01)
- fold 1: C=0.003 wins (0.97041 vs 0.97038 at C=0.01)
- fold 2: C=0.01 wins (0.97047 vs 0.97047 at C=0.003 — very close)
- fold 3: C=0.003 wins clearly
- fold 4: C=0.003 wins
Final refit C=0.003.

Top-weight bases (FULL arm, C=0.003, sum|coef| across classes):
  1. node_0039 (0.9772)  2. cat-3 (0.8762)  3. node_0043 (0.5823)
  4. tabm-1 (0.5743)    5. realmlp-2 (0.5636)  6. node_0003 (0.3648)
  7. xgb-6 (0.3482)     8. node_0033 (0.3373)   9. node_0015 (0.3347)
  10. node_0056 (0.3146)

Baseline assert PASSED: bank-17+FT-T C=1.0 → cv=0.970227 (expected≈0.970211, delta=0.000016).

Artifacts: oof.npy (577347,3) float32, test_probs.npy (247435,3) float32, submission.csv.
Tool: LogisticRegressionCV with 4-fold inner CV for C selection (not 24 sequential fits — 60x faster).
Runtime: ~40 min total (baseline assert 5-fold + TIGHT 5-fold LRCV + FULL 5-fold LRCV + refits),
  slower than estimated due to parallel jobs competing for CPU.
