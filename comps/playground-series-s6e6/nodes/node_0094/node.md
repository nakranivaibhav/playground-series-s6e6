---
id: node_0094
desc: error-pocket-targeted decorrelated base
op: draft
parents: [root]
uses_data: [fs_realmlp_fe, fs_errpocket_w]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966021
sem: 0.000240
folds: [0.966477, 0.966688, 0.965741, 0.965425, 0.965774]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "NULL RESULT by decisive gate: fold-0 err-corr vs node_0070 = 0.7945 (threshold < 0.65). Pocket weighting did NOT decorrelate — this base must NOT be fed to any stack. cv=0.966021 is below parent node_0030 (0.966952); err-corr 0.79 is WORSE than prior decorrelation attempts (0.70-0.72). Hypothesis falsified: a complete classifier with pocket-upweighting makes mistakes in the same places as the bank."
leak: clean
lb: null
submitted: null
created: 2026-06-14T09:00Z
decided: 2026-06-14
tags: [gbdt, lightgbm, fs_realmlp_fe, fs_errpocket_w, error-pocket, instance-weight, decorrelation, data, null-result]
---

## plan
built on:   root (a NEW draft — a complete 3-class LightGBM base on fs_realmlp_fe, reusing node_0030's recipe
            verbatim). This is a DIVERSITY FEEDER for the stack, not a champion-beater: it need not beat the
            champion solo; it must be a complete, honest 3-class classifier whose ERRORS are decorrelated from
            the bank (which would let it lift the stack where every recent base washed at err-corr ~0.70+).
change:     Train node_0030's LightGBM (fs_realmlp_fe, native cats, balanced handling, early-stop on the
            fold's val) UNCHANGED EXCEPT add per-instance training weights from a new fit_in_fold weight set
            fs_errpocket_w that UP-WEIGHTS rows in the bank's hardest error pockets. The weight is the only
            change vs node_0030; the model stays a complete 3-class classifier (no specialist label-fit, no
            narrow-zone retargeting).
hypothesis: the bank's residual error is CONCENTRATED (the low-z GALAXY/STAR confusion zone — cleanlab n59
            confirmed 2.3× concentration there). A complete classifier trained to pay extra attention to those
            cells should make DIFFERENT mistakes than the bank in that zone → err-corr < 0.65 (the bar every
            recent decorrelation attempt failed at ~0.70-0.72: n86/n87/n90), earning a real stack lift.
target:     Balanced Accuracy maximize. DECISIVE fold-0 gate: err-corr vs the bank < 0.65 AND solo fold-0
            BA >= 0.965; if either fails, KILL before the full 5-fold (this is a diversity feeder — if it isn't
            decorrelated it cannot help, and we will not spend GPU/CPU finishing it). On a pass: full 5-fold,
            then a re-stack A/B onto the best honest stack (node_0076/node_0070) to measure its marginal.

WHY this is worth a slot (and how it differs from the dead error-pocket lever): node_0047 was a NARROW
GALAXY-vs-STAR SPECIALIST hand-fit at low-z, added as a stack column — a label-fit error-pocket model that
mirrored the meta's labels → CV mirage 0.970881, LB crash 0.96242 (MEMORY L40; permanently excluded). THIS
node is the opposite: a COMPLETE 3-class classifier, only its sample WEIGHTS are tilted toward the error
pocket. The revival caution (proposer policy) bars reviving a narrow specialist but explicitly trusts a
complete classifier's CV. The leak/mirage risk is therefore lower, but NOT zero (the weights are
label-derived) — hence the explicit fit_in_fold discipline and the human LB-gate below.

=== REQUIRED LEAK-SAFETY DISCIPLINE (the weight scheme is fit_in_fold) ===
fs_errpocket_w is leak-safety class **fit_in_fold** and is registered as such in data.md. The instance weight
is LABEL-DERIVED (uses the row's true class) AND cross-row (a density over redshift-bin × magnitude-bin ×
true-class cells), so it MUST be built TRAIN-FOLD-ONLY:
  - The error-pocket MAP (which redshift×magnitude×class cells the bank gets wrong, and how often) is read
    from node_0070's SAVED fold-honest OOF (nodes/node_0070/oof.npy — each train row's prediction was made by
    a model that never saw that row). Reading that fold-honest OOF as the error source is SAFE and is the
    sanctioned (b)-style revival input.
  - The BIN EDGES (redshift quantile bins, magnitude bins) and the PER-CELL error densities used to set a
    train row's weight are computed on the TRAIN FOLD ONLY — never on the val fold, never on test, never on
    full train. Apply the train-fold-derived weights to the train rows only; val/test rows are scored with no
    weight (weights affect TRAINING only, not inference). Bin edges and per-cell densities are re-derived
    inside each outer fold's train side, exactly like fs_tgt_enc / fs_zresid.
  - Self-gate (before training, costs seconds): the true-class column and id/row-order are NOT in the feature
    list; the weight is used ONLY as the LightGBM sample_weight, never as a feature; bin edges/densities are
    fit inside the fold loop (read your own loop to confirm); OOF covers every train row once.
  - NOTE: if this base ever feeds a stack that gets SUBMITTED to the LB, that stack carries a MANDATORY HUMAN
    LB-GATE (the label-aware-weight mirage path, cf. node_0047 / round_plan label-touching rule). Flag it in
    the node's verdict.

STEP-0 (FREE, before any training — abort-if-diffuse): profile the bank's error cells from node_0070's OOF —
cross-tab OOF-errors over redshift-bin × magnitude-bin × true-class. If the errors are DIFFUSE/uniform (not
concentrated in identifiable cells), the up-weighting has no pocket to target → ABORT before building the
weights or training (same abort logic as C1/n59 STEP-0). Record the error-concentration finding either way.

STEP-1: build fs_errpocket_w (fit_in_fold, per above) → train node_0030's LightGBM with sample_weight = the
pocket weight → fold-0 gate (err-corr < 0.65 AND solo BA >= 0.965) → on pass, full 5-fold + re-stack A/B.

References to READ:
- nodes/node_0030/node.md + nodes/node_0030/src (the LightGBM recipe + fold-honest OOF/test scaffold to copy
  verbatim — fs_realmlp_fe FE, native cats, balanced handling, early-stop). node_0028/src builds fs_realmlp_fe
  end-to-end if you need the FE pipeline.
- data.md fs_realmlp_fe row (the FE recipe; stateless) and the NEW fs_errpocket_w row (the weight recipe;
  fit_in_fold) — both already registered.
- nodes/node_0070/oof.npy (the fold-honest bank OOF = the error-source map; n_train 577347×3).
- nodes/node_0059/node.md + journal 2026-06-10T11:48Z C1-cleanlab (the confirmed 2.3× error concentration in
  the low-z GALAXY/STAR confusion zone — where the pocket is) and round_plan C1 STEP-0 (the abort-if-diffuse
  pattern).
- nodes/node_0047/node.md + MEMORY.md L40 (the specialist CV-mirage this node is deliberately NOT — read it to
  stay on the complete-classifier side of the line).
- journal entries for n86/n87/n90 (the err-corr-0.72 decorrelation failures this node must beat at < 0.65).

DELIVERABLES: STEP-0 error-concentration finding; fold-0 err-corr + solo BA (and KILL if it misses); on pass,
full-5-fold cv/sem/folds + the re-stack marginal onto node_0076/node_0070; oof.npy / test_probs.npy /
submission.csv; gate booleans (with the explicit fit_in_fold self-check result); VOID on leak. Do NOT submit
(orchestrator decides; any submission of a stack containing this base needs the human LB-gate).

## notes
