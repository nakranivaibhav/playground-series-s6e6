---
id: node_0123
desc: GALAXY-recall focal-loss GBDT base
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: gbdt
status: dead
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null, leak_clean: true, cv_too_good: false, passed: false}
gate_note: "cheap-kill tripped: fold-0 BA=0.960323 (best gamma=2.0) < 0.962 threshold; focal loss underperforms standard log-loss by ~6 fold-points"
leak: clean
lb: null
submitted: null
created: 2026-06-17
decided: 2026-06-17T23:57Z
---

## plan
built on:   root — a NEW LightGBM base. Template to copy: nodes/node_0030/src/solution.py
            (LightGBM on fs_realmlp_fe, cv 0.966952). Keep its FE input + fold loop; change ONLY
            the training objective.
change:     Swap the standard multiclass log-loss for a MULTICLASS FOCAL LOSS (custom objective,
            tunable gamma) that concentrates gradient on HARD / misclassified examples — i.e. the
            low-z GALAXY↔STAR boundary that is the BA bottleneck. This is the ONE atomic change:
            objective only; FE, folds, tree params stay as n030. NOT plain class-reweighting
            (that is known-wash — n111 STAR-weighted TabM corr 0.83, DE-threshold washed); the
            new element is focal hard-example focus, which can sharpen the boundary rather than
            just shift it. Tune gamma on fold-0 (e.g. {1,2,3}).
hypothesis: focusing the objective on hard boundary examples yields a base with a genuinely
            SHARPER low-z GALAXY/STAR decision surface — either lifting GALAXY recall net-positive,
            or producing a structurally DIFFERENT, CAPTURABLE boundary (unlike n118, whose GALAXY
            gains were entangled with STAR losses). The new diagnostic decides which.
target:     Balanced Accuracy maximize. Cheap-kill fold-0 BA < 0.962. Judged by the structural
            gate (validation.md): valuable if (a) stack-add to n091 clears the bootstrap-P≥0.90
            bar, OR (b) tools/pred_diagnostic.py shows a NEW capturable holdout fix-block in the
            low-z GALAXY/STAR zone that is NOT entangled (fixes ≫ breaks there, holds on holdout).
            Honest prior: likely a boundary-frontier wash like n118 — this is the cheap, definitive
            last targeted attack on the bottleneck.

DATA well, bottleneck-targeted. READ: nodes/node_0030/node.md + src (the LightGBM-on-richFE base
to copy); data.md fs_realmlp_fe recipe; validation.md (the structural gate). Focal multiclass in
LightGBM = a custom fobj (softmax → focal gradient/hessian) — libraries-first: use the standard
focal formulation, do not hand-roll the tree. fs_realmlp_fe is the established fold-honest set.
CPU; minutes. Emit oof/test/submission; run pred_diagnostic.py vs n091 after scoring.

## notes
CHEAP-KILL AT GAMMA TUNING STAGE.

Gamma sweep on fold-0:
  gamma=1.0: fold-0 BA=0.960265 (best_iter=650, 41.4s)
  gamma=2.0: fold-0 BA=0.960323 (best_iter=760, 49.7s)  <-- best
  gamma=3.0: fold-0 BA=0.959555 (best_iter=571, 38.5s)

Best gamma=2.0, fold-0 BA=0.960323 < 0.962 threshold → CHEAP-KILL. Status: dead.

VERDICT: Focal loss hurts this model. Standard log-loss on the same FE achieves ~0.9666/fold
(node_0030). Focal at gamma=2.0 reaches only 0.9603 — about 6 fold-points below. The hypothesis
that focal would sharpen the GALAXY/STAR boundary is DISPROVED: the hard-example weighting
appears to destabilize the multiclass gradient for a task where boundaries are not purely
example-hardness-driven but likely feature-geometry-driven (redshift as the key discriminant).

Implementation note: found and fixed a critical infinite loop bug (while-loop seeking
leakage_scan.py walked to filesystem root then looped forever). Also found that LightGBM 4.6.0
requires custom objective to be passed as `params["objective"] = callable` (not `fobj=` kwarg
on lgb.train which was removed). Both fixed in this node's src.

No artifacts emitted (no oof.npy, no submission.csv) — run terminated at cheap-kill.
Runtime: ~2.2 min (3 gamma candidates × ~44-50s each on fold-0).
