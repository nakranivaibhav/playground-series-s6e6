---
id: node_0136
desc: disagreement-feature augmented meta
op: combine
parents: [node_0091, node_0070, node_0039, node_0033]
uses_data: []
family: ensemble
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970348
sem: 0.000260
folds: [0.971269, 0.970096, 0.969895, 0.969914, 0.970567]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-21T07:36Z
decided: null
tags: [stack, meta, disagreement-features, base-uncertainty, data]
---

## plan
built on:   the champion n091 mega-stack recipe (balanced multinomial L2-LogReg on
            clipped base log-probs, nested in-fold C grid, frozen folds.json,
            argmax). The base POOL and the meta FAMILY stay byte-identical to n091
            (champion/src/solution.py). Parents listed are the OOF sources the
            disagreement features are DERIVED from: n091 (the stack itself /
            reference), n070 (the strongest external stack), n039 (load-bearing
            CatBoost), n033 (TabM) — all have saved oof.npy.
change:     ONE atomic change — AUGMENT the meta's feature matrix with a small,
            fixed block of per-row BASE-DISAGREEMENT features computed from the
            existing base OOF (no new model, no retraining a base):
              - argmax-vote entropy across the pooled bases (how split is the vote);
              - per-class probability dispersion (std across bases of each class
                prob);
              - max−2nd-max margin of the mean base prob (decision confidence);
              - top-2 class identity disagreement flag (do bases disagree on WHICH
                two classes are in contention).
            These ~6-10 scalar columns are appended to the n091 base-logprob matrix
            and the SAME L2-LogReg meta is re-fit. Hypothesis-bearing claim: a
            GLOBAL per-base coefficient (n091) is region-blind, but a disagreement
            feature lets the meta DOWN-WEIGHT the pooled vote exactly in the
            high-entropy rows (the entangled GAL↔STAR / GAL↔QSO zone) where the
            12-node holdout-significant complementary signal lives.
hypothesis: the row-level complementary signal is real but entangled (journal
            2026-06-17T12:26Z capturability + n122/n127 closed region-gating). It
            was uncapturable by region (redshift band) or a learned multiplicative
            gate. Disagreement features are a DIFFERENT conditioning axis —
            data-derived uncertainty, not geometry — so the meta may finally route
            on "where the bases fight" instead of "where in z-space" and recover a
            sliver of the entangled signal.
target:     Balanced Accuracy maximize. Promote iff the augmented-meta CV beats
            champion 0.970355 by the structural gate (tools/pred_diagnostic.py
            bootstrap P(>champ) ≥ 0.90 AND a holdout-fold fix-block that HOLDS —
            the n0047/n122/n127 mirage guardrail: a working-only lift that reverses
            on the untouched holdout is a kill, not a keep).

## build protocol
- SANITY ASSERT FIRST (seconds): with the disagreement block ZEROED, the augmented
  meta must reproduce n091 ≈ 0.970355. If it does not, the OOF ingest / column
  order / clip-norm is wrong — STOP and fix before reading any lift (the same
  misbuild guard that caught n070 v1 and was reused for n122).
- The disagreement features are computed PER ROW from the base OOF columns only —
  they are fold-honest by construction (each base's OOF is already
  leave-fold-out), so the block carries no extra leak as long as it is built from
  the OOF matrix, never from a refit-on-full-train prob. Verify this in the fold
  loop read.
- This is a probe-cheap combine (no GPU, no base retrain) — minutes on CPU.

## references to READ
- champion/src/solution.py + a1_full_merge.py / a1_submit.py — the n091 OOF-ingest
  (clip ±30, normalize) + nested-C balanced-LogReg meta to extend with the block.
- nodes/node_0122/node.md + journal 2026-06-17T17:04Z (soft region-interacted meta
  WASH) and nodes/node_0127/node.md + journal 2026-06-17T21:23Z window (MoE-gated
  meta mirage) — region/partition conditioning is CLOSED; this node tries the
  UNCERTAINTY axis those did not, and inherits their exact mirage guardrail.
- journal 2026-06-17T12:26Z (capturability_check: the entangled complementary
  signal is real, fixes/breaks interleaved row-by-row) — the motivation.
- tools/pred_diagnostic.py — the structural + holdout gate.
- the parent oof.npy files (nodes/node_0091|0070|0039|0033/oof.npy) and the n091
  FULL-pool base list (the disagreement stats are computed over that same pool).

## notes
ORCHESTRATOR-GATED (developer agent overflowed context mid-run at 376 tool calls; the
backgrounded compute completed and the orchestrator took over the marker, read the final
summary, ran the bootstrap arbiter, and wrote this record). Build is correct: SANITY ASSERT
PASS (disagreement block zeroed reproduces n091 within tol: cv 0.970289 vs 0.970355).
Two arms swept: TIGHT+DISAG cv 0.970284, FULL+DISAG cv 0.970348 (winner). Disagreement
block (vote-entropy, per-class prob dispersion, STAR-vs-GALAXY vote count, stack max-margin)
adds only +0.000059 over the zeroed baseline.
VERDICT: VALID, NO-PROMOTE. Bootstrap vs n091 (B=3000): P(cand>champ)=0.415, delta -0.000006,
95% CI [-0.000074,+0.000061] = INDISTINGUISHABLE. Flip analysis: net -80 (121 fixes / 201
breaks), McNemar p=9.69e-06 but in the WRONG direction (net negative); per-class net GALAXY
-97 / QSO -9 / STAR +26 — no holdout-significant fix-block a stack can exploit. Per-row
base-disagreement features do NOT unlock the entangled GAL/STAR signal — same wall as the
region/MoE conditioning attempts (n122/n127). Champion n091 stands.
