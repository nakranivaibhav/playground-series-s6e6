---
id: node_0141
desc: learning-to-rank TabM dual-head base
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: dead
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.968676]
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "CHEAP-KILL at fold-0: err_corr vs n070 = 0.8492 >= 0.65 threshold. Fold-0 BA=0.968676 (passes BA check). Ranking head does NOT decorrelate from n070 error structure — same confusable pairs, same trunk features. Node dead per plan."
leak: clean
lb: null
submitted: null
created: 2026-06-21T07:36Z
decided: 2026-06-21T08:10Z
tags: [nn, tabm, learning-to-rank, pairwise-margin, contrastive, wildcard, gpu]
---

## plan
built on:   root (a wildcard draft — one COUPLED hypothesis: a TabM trained with a
            learning-to-RANK objective alongside the usual CE head). Copy
            nodes/node_0033/src as the scaffold (the TabM FE + fold-honest OOF/test
            loop over frozen folds.json + fs_realmlp_fe); change the OBJECTIVE, not
            the FE.
change:     ONE wildcard hypothesis (coupled heads = one objective, licensed in the
            wildcard well; ablate next round if it wins): add a pairwise-margin /
            contrastive ranking head to the TabM trunk on top of the standard
            cross-entropy head. The CE head learns the class posterior as usual; the
            ranking head learns, via a pairwise-margin or contrastive loss, to
            ORDER rows by class-membership margin (push apart the entangled
            GAL↔STAR / GAL↔QSO pairs in score space). The two losses are summed
            (small weight on the ranking term). The output written to oof/test is
            the CE-head 3-class posterior; the ranking term is an auxiliary
            objective that shapes the trunk. LIBRARIES-FIRST for the ranking loss:
            torch.nn.functional.margin_ranking_loss or pytorch-metric-learning
            (uv add) — hand-roll only if neither fits, and say so with the reason.
hypothesis: cross-entropy optimises calibrated per-class probability but is
            indifferent to the ORDERING margin between the entangled classes; a
            ranking objective explicitly maximises the separation margin on the
            confusable pairs. A trunk shaped by both may produce a score geometry
            that both (a) is decorrelated from the CE/GBDT bank (a new objective is
            a new error structure) and (b) sharpens exactly the macro-recall
            boundary balanced-accuracy rewards — the loss-objective axis attacked
            with ranking rather than the reweighting that n111/n094/n059/n123 closed.
target:     Balanced Accuracy maximize. CONCRETE CHEAP-KILL (run on fold-0 BEFORE
            any full 5-fold): continue ONLY if fold-0 err-corr vs node_0070 < 0.65
            AND solo fold-0 BA ≥ 0.960; else STOP. If it passes, run the full
            5-fold → oof.npy, then the stack-add to n091 via the structural gate
            (bootstrap P ≥ 0.90 + holdout fix-block + the mirage guardrail).

## build protocol (cost-staged cheap-kill)
1. SMOKE the dual-head TabM: small subsample, few epochs — verify the ranking loss
   integrates, pipeline runs, project timing + VRAM.
2. FOLD-0 (background + marker /tmp/s6e6_node_0141.done): compute BOTH solo fold-0
   BA AND err-corr vs node_0070 (load node_0070/oof.npy on the fold-0 val rows).
   APPLY THE CHEAP-KILL: err-corr < 0.65 AND BA ≥ 0.960 to continue, else STOP.
   (This wildcard carries a kill criterion BEFORE any long run, per the policy.)
3. FULL 5-fold only if the cheap-kill passes → oof.npy + test_probs.npy +
   submission.csv over the frozen folds.

## leakage discipline (same standard as parent-scaffold n33)
- The ranking-pair sampling draws pairs from the TRAIN FOLD ONLY (never pairs a
  val/test row) — read the loss/batch loop to confirm. The OOF written is the CE
  head's leave-fold-out prediction, covering every train row once.
- Stateless fs_realmlp_fe once; factorize/KBins/TargetEncoder fit train-fold-only;
  folds from frozen folds.json.

## references to READ
- nodes/node_0033/src/solution.py + features.txt — the TabM FE + fold-honest
  OOF/test scaffold to copy and extend with the second loss head.
- torch margin_ranking_loss docs / pytorch-metric-learning — libraries-first for
  the ranking objective (CLAUDE.md hard-rule 8).
- nodes/node_0111|0094|0059|0123/node.md + journal entries (loss-REWEIGHTING axis
  — STAR-weighted, error-pocket, cleanlab, focal — all closed/wash) — this node
  attacks the objective with RANKING, a different axis than reweighting; the
  cheap-kill is calibrated to stop fast if it lands in the same wall.
- nodes/node_0070/oof.npy + tools/pred_diagnostic.py — the err-corr reference for
  the cheap-kill and the structural stack gate.

## notes
- Wildcard discipline: CE head + ranking head = ONE hypothesis. If it WINS (clears
  the structural gate), ablate next round — run the same TabM trunk with the CE head
  ONLY to attribute the lift to the ranking objective vs extra training signal.

## fold-0 results (cheap-kill triggered)
- fold-0 BA: 0.968676 (above 0.960 kill threshold — passes)
- err-corr vs node_0070 (fold-0 val rows): 0.8492 (above 0.65 kill threshold — KILLS)
- timing: fold-0 = 141s, projected 5-fold = ~12min
- VRAM: 5.94 GB peak

## kill verdict
The pairwise-margin ranking head trained jointly with CE does NOT decorrelate errors
from the n070 baseline (TabM CE-only on the same feature set). err_corr=0.8492 is very
high — the ranking objective reshapes the loss landscape but the trunk still latches
onto the same informative features in the same way, producing the same confusion pattern
on the entangled GAL/QSO/STAR pairs. The hypothesis (ranking = decorrelated error
structure) is falsified by the cheap-kill criterion. Node is dead; no 5-fold run.

Lesson: adding a ranking auxiliary loss to TabM on these stellar features does not
produce the decorrelated representation needed for stack-add diversity. The error
structure is dominated by the feature set (fs_realmlp_fe), not the loss objective,
at least at this alpha=0.1 ranking weight. A different framing (e.g. class-conditioned
contrastive pretraining, or a different feature set) might be needed.
