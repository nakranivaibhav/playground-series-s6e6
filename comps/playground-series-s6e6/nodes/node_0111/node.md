---
id: node_0111
desc: STAR-recall-weighted TabM base
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: buggy
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: null, passed: false}
gate_note: "ERRCORR gate tripped: best multiplier (2.0) fold-0 err_corr=0.8287 >= 0.70. STAR_MULT=1.5 gave err_corr=0.8026, BA=0.967953; STAR_MULT=2.0 gave err_corr=0.8287, BA=0.968033. BA gate passed (both >= 0.965), but decorr gate failed for both. Same arch+features as n033 — asymmetric loss weights alone cannot shift error geometry enough. Full 5-fold not run."
leak: clean
lb: null
submitted: null
created: 2026-06-15T11:09Z
decided: 2026-06-15T17:37Z
---

## plan
built on:   root — DATA-centric well. Copy node_0033's TabM-richFE recipe VERBATIM (`nodes/node_0033/src`).
change:     Train TabM-richFE with ASYMMETRIC per-class sample weights: STAR (minority, 14%, the BA
            bottleneck) up-weighted to 1.5–2× the balanced value; QSO/GALAXY left at balanced. Fold-0 sweep
            STAR multiplier ∈{1.5,2.0}, freeze best.
hypothesis: asymmetrically up-weighting the STAR minority lifts STAR recall (the macro-BA bottleneck) AND
            shifts the base's per-class error profile enough to decorrelate from n033 (<0.70), where
            symmetric balanced weights (n016) and post-hoc stack thresholds (n089, best b=1.0) could not.
target:     BA maximize · GATE fold-0 solo BA ≥ 0.965 AND err-corr vs node_0070/n033 < 0.70 → else cheap-kill
            (decorrelation, not BA alone, is the point — same arch/features as n033 means it likely stays
            correlated; if so it adds nothing and dies even at BA≥0.965). If clears → full OOF + restack >2·sem.

DELTA vs closed STAR knobs (why this is not a repeat):
- n089 STAR-BOOST was POST-HOC probability re-scale on the STACK output (best b=1.0 = no boost); it cannot
  change any BASE's learned error geometry. This changes the base's TRAINING DISTRIBUTION → a different
  per-class decision surface the meta can exploit.
- n016 used SYMMETRIC balanced weights (all classes to uniform prior). This is ASYMMETRIC — STAR pushed PAST
  balanced while QSO/GALAXY stay at balanced, deliberately trading a little GALAXY recall for STAR recall.

HOW (TIGHT — single base, NO full-pool loader; do NOT read node_0091's solution.py):
- cp nodes/node_0033/src/solution.py → nodes/node_0111/src/solution.py. Keep fs_realmlp_fe, tabm loop, PLR,
  frozen folds, OOF/test/submission writing. ONLY change: the per-class sample weights feeding the loss —
  compute the balanced weights as n033/n016 do, then multiply the STAR weight by the swept multiplier.
- GATE ORDER: fold-0 only first (both multipliers); solo BA + err-corr vs nodes/node_0070/oof.npy. If best
  multiplier's BA<0.965 OR err-corr≥0.70 → STOP, record. Else all 5 folds at best multiplier.
- Outputs nodes/node_0111/{oof.npy, test_probs.npy, submission.csv, train.log}. Self-gate (kaggle-leakage):
  weights are train-row only (val/test scored unweighted); folds frozen; OOF full/no-NaN/each-row-once;
  dist sane; schema. Write gates + cv/sem/folds + leak + err-corr (gate_note); stage: built. Do NOT submit.
  `uv run`, tabm library. GPU — marker DONE=/tmp/playground-series-s6e6_node_0111.done.

## notes
well=data. Complete 3-class classifier (NOT a narrow error-pocket specialist → n047 mirage rule does not
apply; restack is honest). The err-corr gate is the real test: same arch+features as n033, only loss weights
differ, so decorrelation is the open question.
