---
id: node_0139
desc: ambiguity-aux target gates the stack
op: draft
parents: [root]
uses_data: [fs_realmlp_fe, fs_ambig]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969328
sem: 0.000289
folds: [0.970454, 0.968816, 0.969119, 0.969056, 0.969195]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "Holdout vs n028 delta is +0.000015 (near-zero); working lift vs n028 +0.000263 (P=0.998). The aux target adds a modest lift but barely moves holdout. err-corr vs n070=0.8785 (slightly higher than n028's 0.8730). Stack-add potential is marginal — the model is highly correlated with n028 (prob-corr ~0.999). Not a mirage (holdout does not reverse) but gain is small."
leak: clean
lb: null
submitted: null
created: 2026-06-21T07:36Z
decided: null
tags: [nn, multi-task, auxiliary-target, ambiguity-gate, wildcard, gpu]
---

## plan
built on:   root (a wildcard draft — one COUPLED hypothesis, licensed only in the
            wildcard well: an auxiliary "ambiguity" target AND the gate that uses
            it, trained together). Copy nodes/node_0028/src as the scaffold (the
            RealMLP/NN FE + fold-honest OOF/test loop over frozen folds.json that
            already builds fs_realmlp_fe and writes oof.npy/test_probs.npy);
            extend the model head, do not change the FE pipeline plumbing.
change:     TWO coupled pieces forming ONE hypothesis (this is the wildcard
            license — if it wins, ablate next round to attribute):
            (1) NEW feature-set fs_ambig — a per-row AMBIGUITY label derived
                ONLY from the existing champion-bank OOF disagreement (e.g. row is
                "ambiguous" if the pooled base argmax-vote entropy is high / the
                bases split across the entangled GAL↔STAR or GAL↔QSO pair). This is
                a LABEL built from fold-honest OOF, used as an AUXILIARY target —
                NOT the competition class. fs_ambig leak-safety = fit_in_fold (the
                ambiguity label is derived from train-fold-honest OOF + train
                labels; it must be built per fold from that fold's OOF, never from
                a refit-on-full-train or test prob).
            (2) a MULTI-TASK NN: shared trunk on fs_realmlp_fe, two heads —
                the 3-class head (the real objective) AND the ambiguity-aux head;
                the aux head's predicted ambiguity GATES the 3-class logits (e.g.
                the trunk learns an ambiguity-aware representation, and at inference
                the class head's confidence is modulated where the model expects
                confusion). The output written to oof/test is the 3-class
                prediction; the aux head is an auxiliary signal that shapes the
                shared representation.
hypothesis: every region/uncertainty GATE tried so far was applied POST-HOC to
            frozen base outputs (n122 interacted meta, n127 MoE gate, the
            capturability hard-gate) and all washed/miraged because the entangled
            fixes/breaks are interleaved at row level in the FROZEN predictions.
            A model that LEARNS an ambiguity-aware representation END-TO-END (the
            aux target shapes the trunk during training, not after) might separate
            the entangled rows the post-hoc gates could not — the entanglement may
            be breakable in representation space even if it is not breakable in
            output space.
target:     Balanced Accuracy maximize. CHEAP-KILL on fold-0: if the 3-class-head
            fold-0 BA < 0.965 (below the n028/n033 NN tier), kill the wildcard —
            the aux objective is hurting the primary task. If it clears tier, run
            the full 5-fold → oof.npy, then BOTH (a) solo CV vs the n028 baseline
            and (b) err-corr vs n070 + stack-add to n091 via the structural gate
            (bootstrap P ≥ 0.90 + holdout fix-block + the n0047/n127 mirage
            guardrail — a working-only lift that reverses on holdout is a kill).

## build protocol (cost-staged cheap-kill)
1. Build fs_ambig FIRST (cheap, CPU): the per-fold ambiguity label from the bank
   OOF — verify it is fold-honest (each fold's label from that fold's OOF only).
2. SMOKE the multi-task NN: small subsample, few epochs — pipeline + timing + VRAM.
3. FOLD-0 only (background + marker /tmp/s6e6_node_0139.done) → 3-class-head tier
   read. Cheap-kill at BA < 0.965.
4. FULL 5-fold only if fold-0 clears tier → oof.npy + test_probs.npy + submission.

## leakage discipline (CRITICAL — aux target is OOF-derived)
- fs_ambig is fit_in_fold: the ambiguity label is computed from the TRAIN-FOLD
  rows' fold-honest OOF + their true labels, applied within that fold only; val/
  test rows are NEVER assigned an ambiguity label from their own (unavailable)
  labels or from a full-train refit. Read the fold loop to confirm.
- The aux head trains on the train fold's ambiguity labels only; the OOF written is
  the 3-class head's leave-fold-out prediction, covering every train row once.
- Stateless fs_realmlp_fe once; factorize/KBins/TargetEncoder fit train-fold-only;
  folds from frozen folds.json.

## references to READ
- nodes/node_0028/src/solution.py + features.txt — the NN FE + fold-honest OOF/test
  scaffold to copy and extend with the second head.
- journal 2026-06-17T12:26Z (capturability: entangled signal real but
  row-interleaved) + nodes/node_0122|0127/node.md (post-hoc region/MoE gates
  WASH/mirage) — the motivation for an END-TO-END learned gate and the mirage
  guardrail this node must clear.
- the champion-bank OOF (nodes/node_0091|0070/oof.npy) — the source the ambiguity
  label is derived from.
- tools/pred_diagnostic.py — the structural + holdout gate.
- champion/src/solution.py — the n091 stack-add target.

## notes
- Wildcard discipline: this bundles the aux target + the gate as ONE hypothesis. If
  it WINS (clears the structural gate), the next round MUST ablate — run the same
  trunk WITHOUT the aux head to attribute the lift to the gate vs the extra
  representation capacity.

## RESULT (2026-06-21)
cv=0.969328 ± 0.000289 (5 folds: [0.970454, 0.968816, 0.969119, 0.969056, 0.969195])
runtime: ~12 min total (fold-0 probe 109s → 5-fold ~715s)

vs n028 (parent NN scaffold):
  - working CV: +0.000263 (bootstrap P=0.998 — REAL gain)
  - holdout (fold-4): +0.000015 (near-flat — the gain barely persists on holdout)
  - McNemar p=0.022, net fixes +114 across all classes/bands

vs n091 (champion mega-stack):
  - working CV: -0.001027 (expected — n091 is a 60+ base stack; a single NN can't beat it)
  - holdout: -0.001431

err-corr vs n070: 0.8785 (slightly higher than n028's 0.8730)
prob-corr vs n028: ~0.999 (the aux head minimally changed the prediction geometry)

fs_ambig statistics:
  - ~3.0-3.1% of train-fold rows are labelled "ambiguous" (high n070 OOF entropy + wrong pred)
  - aux_loss converges from 0.15 → 0.01-0.10 over 6 epochs, suggesting the head learns a real signal
  - AUX_LOSS_ALPHA=0.15 was used (primary CE dominates)

Per-class recalls (full OOF):
  GALAXY: 0.9592 / QSO: 0.9767 / STAR: 0.9721

Verdict:
  The wildcard hypothesis partially worked: the aux head adds a measurable, statistically
  real improvement over the bare n028 NN (+0.000263, P=0.998). However, the gain barely
  persists on the inviolable holdout (+0.000015) and the model remains highly correlated
  with n028 (prob-corr ~0.999). The aux representation did not fundamentally separate the
  entangled rows — the end-to-end learned gate hypothesis is weakly supported.

  STATUS: valid — the node is honest, leak-clean, and better than n028. It can serve as
  a base for the stack in place of or alongside n028. However, the tiny holdout delta
  means it should NOT be used as a standalone champion candidate.

  Next: ablate (run same trunk WITHOUT aux head) to check if the +0.000263 comes from
  the representation or just from the slight hyperparameter variance.
