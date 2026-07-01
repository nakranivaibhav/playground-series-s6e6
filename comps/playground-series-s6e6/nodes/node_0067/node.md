---
id: node_0067
desc: transductive soft-label distillation of bank stack
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969414
sem: 0.000110
folds: [0.969751, 0.969367, 0.969143, 0.969565, 0.969246]
baseline_cv: 0.968053
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "cv=0.969414 between parent n33 and champion n63; solo +0.00136 vs parent; bank-17+n67 hurts stack (-0.000129); bank-17+n33+n67 = 0.970251 (+0.000098 vs champ) but <<2sem (0.000444) — wash; no promote"
leak: clean
lb: 0.96998
submitted: 2026-06-12T13:37Z
created: 2026-06-12
decided: 2026-06-12
tags: [wildcard, distillation, transductive, gpu]
---

## plan
built on:   root (new draft). Student arch = TabM-richFE from node_0033.
change:     Distill the bank-17 stack into a single student: per fold, train the student with KL loss on SOFT
            teacher probabilities over train-fold rows (teacher = fold-honest stacked OOF) PLUS all test rows
            (teacher = per-fold stacked test_probs), then predict val-fold and test.
hypothesis: soft-label transductive distillation compresses the bank ensemble into one strong de-correlated
            student that, unlike weak diversity arms, is strong enough to lift the saturated stack.
target:     Balanced Accuracy maximize; interesting if solo CV > 0.968 (TabM n33 0.968053); promotable if bank
            restack CV > 0.970153.

GPU — serialize behind other GPU nodes. This is the WILDCARD (two coupled levers in one hypothesis): soft
targets carry the ensemble's dark knowledge AND the student sees test-distribution INPUTS.

Distinct from dead node_0046 (HARD pseudo-labels into a GBDT retrain → wash). Labels are NEVER touched — teacher
probs come from the stack's test PROBABILITIES, no ground truth involved; leak-safe because the teacher is
fold-honest OOF on the train side.

CRITICAL honesty rule (reviewer-confirmed): the teacher for fold f — on BOTH train-fold rows AND test rows —
must be the fold-f meta (per-fold stacked OOF + that fold's test probs), NEVER the refit-on-all teacher.

References to READ:
- Student arch: `comps/playground-series-s6e6/nodes/node_0033/src` (TabM on fs_realmlp_fe, GPU-fast).
- Teacher artifacts: `comps/playground-series-s6e6/champion/oof.npy` + `test_probs.npy` (rows aligned to frozen
  folds — node_0063 notes). NOTE: champion oof/test_probs are the refit teacher; you must reconstruct the
  PER-FOLD meta for the honesty rule — read champion/src a1_submit.py to mirror the fold-f stacking.

Loss: KL(student ‖ teacher) with temperature ~2, optional small CE-to-true-label mix (~0.3) on train rows.
Evaluate the student's OOF balanced accuracy against the TRUE labels. Then restack as an 18th bank base.
CV-mirage guard (node_0047 lesson): the student is a COMPLETE classifier scored on honest OOF, not an
error-pocket specialist — eligible. Kill: fold-0 solo < 0.967.

## notes
cv=0.969414±0.000110 (5 folds: 0.9698,0.9694,0.9691,0.9696,0.9692). +0.00136 vs parent n33 (0.968053).
Per-class recalls (OOF full): GALAXY=0.9596, QSO=0.9769, STAR=0.9717. Interesting pattern: distillation
shifts errors from QSO/STAR toward GALAXY (teacher is ~0.97 BA on those classes; the soft labels compress
uncertainty to the ensemble's belief, which is QSO/STAR-biased).
Bank restack probe: bank-17+n67=0.970024 (-0.000129 vs champ 0.970153); n67 alone hurts; 
bank-17+n33+n67=0.970251 (+0.000098, <<2·sem 0.000444 — wash). No promotion.
Verdict: valid data-centric win vs parent, but below champion and does not lift the saturated stack.
The wildcard hypothesis (distillation compresses dark knowledge into a student strong enough to lift a 
saturated ensemble) is TRUE on solo CV (+0.00136 is real and consistent across all 5 folds) but FALSE 
on stack contribution — the student's errors too highly correlated with the bank it was distilled from.
