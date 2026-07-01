---
id: node_0098
desc: RBF base full-OOF + champion stack-add test
op: improve
parents: [node_0097]
uses_data: [fs_rbf_nystroem]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.946874
sem: 0.000249
folds: [0.946986, 0.947179, 0.947416, 0.945960, 0.946827]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "DECISIVE STACK-ADD RESULT — RBF base HURTS the champion stack. Solo BA=0.946874 (5-fold mean, sem=0.000249), err-corr vs node_0070=0.5625 (confirmed decorrelation). Baseline assert PASS: FULL baseline cv=0.970355 (exact match to champion). TIGHT+98: cv=0.970275 (delta=-0.000026 vs TIGHT baseline); FULL+98: cv=0.970306 (delta=-0.000049 vs FULL baseline). Per-fold FULL+98 deltas: [+0.000029, -0.000062, +0.000024, -0.000209, -0.000023]. Best arm FULL+98 cv=0.970306 is BELOW champion 0.970355 by 0.000049; far below promote bar 0.970920 (champ+2*sem). RBF decorrelation lever CLOSED: n62 weak-but-decorrelated lesson confirmed holds even at err-corr 0.54 — decorrelation insufficient to overcome 2.4pp BA gap vs champion bases. Runtime=42.3min."
leak: clean
lb: null
submitted: null
created: 2026-06-14T11:43Z
decided: 2026-06-14
---

## plan
built on:   node_0097 (Nystroem RBF on rich FE, 2000 components, gamma 0.0556) — solo BA 0.9466,
            err-corr 0.541 vs node_0070. KILLED twice (n096/n097) on the solo-BA floor BEFORE ever
            running the actual question.
change:     DROP the solo-BA gate and run the DECISIVE TEST. Run node_0097's exact config to a FULL
            5-fold OOF (we already know solo BA ≈ 0.947 and err-corr ≈ 0.54 — no gate needed), then
            ADD that OOF as one more base column to the CHAMPION node_0091 full-pool LogReg meta and
            refit fold-honest (nested C). The ONE question: does a strongly-decorrelated (0.54) but
            weak (0.947) RBF base ADD to the saturated champion stack, or wash/hurt?
hypothesis: the RBF kernel base is the only thing since FT-Transformer to decorrelate from the bank
            (err-corr 0.54 vs the 0.70-0.79 floor of every other reframing). Stacking theory says a
            sufficiently decorrelated base can lift even when individually weak. The n62 prior (weak
            +decorrelated 0.655/0.958 HURT the stack) is at a HIGHER correlation; at 0.54 the
            complementary signal may finally outweigh the weakness. This is the test that prior
            couldn't pre-decide.
target:     Balanced Accuracy maximize. Promote the resulting STACK (register as a combine node off
            node_0091 + node_0098 if it wins) ONLY if its fold-honest OOF BA beats champion node_0091
            (0.970355) by > 2·sem. If it washes or hurts, the RBF decorrelation lever is CLOSED —
            record that cleanly (a decorrelated base too weak to help is the n62 lesson confirmed at
            lower corr).

HOW (hand the developer the concrete experiment):
- Reuse node_0097's pipeline VERBATIM (nodes/node_0097/src): rich FE fs_realmlp_fe → StandardScaler
  (fit_in_fold) → sklearn Nystroem(rbf, n_components=2000, gamma=0.0556, landmarks fit on a train-fold
  subsample) → balanced multinomial LogReg. NO gamma re-sweep (use 0.0556), NO solo-BA kill — run all
  5 folds to a complete OOF (577347,3) + test_probs (247435,3).
- THE DECISIVE STACK-ADD: read the champion recipe nodes/node_0091/src/solution.py (balanced
  multinomial LogReg on clipped log-probs over the full base pool, nested in-fold C). Add THIS node's
  OOF/test as ONE more base column to that exact pool; refit the meta fold-honest with the same nested-C
  protocol. REPORT: champion stacked CV 0.970355 (reproduce as baseline — assert ≈0.970355 first),
  the new stacked CV with node_0098 added, the delta, and per-fold deltas. Also report this base's
  full-5-fold solo BA + its full-OOF err-corr vs node_0070 (confirm it stays ~0.54).
- References to READ: nodes/node_0097/src + node.md (the config to rerun); nodes/node_0091/src/
  solution.py (the champion stack to add into — reproduce its 0.970355 baseline first); nodes/
  node_0070/oof.npy (err-corr baseline); MEMORY [ensemble] n62 (the weak+decorrelated-HURT prior this
  tests at lower corr).
- Self-gate (kaggle-leakage): scaler/landmarks fit train-fold-only; folds frozen; OOF full/no-NaN;
  dist sane; submission schema; the stack meta refit fold-honest (nested C, never on the scored fold).
  Write gate booleans + leak. stage: built. Do NOT submit (orchestrator decides).
- If the 5-fold + stack refit is long, background with the marker pattern
  (DONE=/tmp/playground-series-s6e6_node_0098.done, redirect to train.log).

## notes
This is the cheapest decisive test of the RBF lever. If the stack-add wins, the NEXT node scales kernel
capacity (SGD/incremental LogReg to dodge the 4000-component OOM, push solo BA up, likely larger stack
gain). If it washes, RBF is closed and we've learned the n62 weak-but-decorrelated rule holds even at
err-corr 0.54.
