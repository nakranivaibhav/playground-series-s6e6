---
id: node_0070
desc: bank-17 + new external decorrelated bases (fwd-select)
op: improve
parents: [node_0063]
uses_data: []
family: ensemble
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970211
sem: 0.000251
folds: [0.971103, 0.970119, 0.969771, 0.969718, 0.970343]
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.97087
submitted: 2026-06-12T17:55Z
created: 2026-06-12
decided: null
tags: [public-bank, stack, external-oof, look-outside, finals-slot-1-eligible]
---

## plan
built on:   node_0063 champion (17-base balanced multinomial LogReg on clipped log-probs + DE per-class
            threshold). Meta recipe byte-identical; the base SET grows by adding NEW external OOF.
change:     Add genuinely NEW, decorrelated external base models (sourced 2026-06-12 into refs/ext_oof/, NOT in
            Deotte's bank) to the stack via GREEDY FORWARD SELECTION — start from bank-17, at each step add the
            single candidate base that most improves the fold-honest stacked CV; stop when no remaining
            candidate improves CV by more than a small epsilon (e.g. +0.00003). Forward-select (not blind
            bundle) so each base's marginal contribution is attributable.
hypothesis: the saturated bank only grows from bases that are strong AND decorrelated; every IN-HOUSE add has
            washed (n52/n64/n67), but these are orthogonal external families (a GNN, an FT-Transformer, sklearn
            ExtraTrees/HGB) the stack has never seen — their decorrelation should lift CV where in-house adds
            could not.
target:     Balanced Accuracy maximize; beats champion if CV > 0.970153 by > 2·sem (fold-noise). A genuine
            external lift here is the first promote-eligible result since n63.

CPU-only — OOF ingest + LogReg refit per forward-selection step, minutes. Use `uv run --no-sync`.

CANDIDATE BASES (all verified id-aligned to our folds: n_train 577347, n_test 247435; ranked by decorrelation EV):
- refs/ext_oof/ravi_gnn_mlv1/   GNNV1  (oof_GNNV1_1.npy / pred_GNNV1_1.npy) — Graph NN, solo BA 0.9345 — TOP decorrelation
- refs/ext_oof/pilkwang_5090/   ft_transformer_lite (oof_*/sub_* csv, cols proba_GALAXY/QSO/STAR) — FT-Transformer, 0.9294
- refs/ext_oof/pilkwang_5090/   extratrees_soft — ExtraTrees, 0.9473 — only randomized-bagging tree
- refs/ext_oof/pilkwang_5090/   hgb_balanced — sklearn HistGradientBoosting, 0.9564 — highest new solo, new GBDT impl
- refs/ext_oof/pilkwang_5090/   tabm_lite (0.9312), logit_elastic (0.8982) — lower EV, include as candidates
  (pilkwang also ships ridge_l2/logit_l2/hgb_regularized/seed2026 variants — include if trivially loadable)

References to READ: champion/src (a1_full_merge.py ingest+merge pattern, a1_submit.py DE-threshold fit);
the sourcing report is in the journal/this round; refs/ext_oof/ for the files.

CRITICAL leak/alignment checks (these are EXTERNAL OOF — same class as n63, verify carefully):
- Every candidate OOF is EXACTLY 577347 rows, id-ordered to train.csv (the rejected lzsecurity bank had 731273
  rows w/ external SDSS concatenated — confirm NONE of yours have that). Test is EXACTLY 247435 rows, ids
  577347..824781.
- Convert to probabilities (clip+normalize) consistent with the champion's `norm`; csv files use cols
  proba_GALAXY/QSO/STAR (map to our GALAXY=0,QSO=1,STAR=2 order); npy are (n,3) already — VERIFY the column
  order by computing each base's solo OOF balanced accuracy (must match the report ~value, not ~0.33).
- Folds loaded from frozen folds.json; DE threshold fit fold-honest (nested), never on full train.

A/B to report: champion 0.970153 baseline; the forward-selection PATH (which base added at each step + its CV);
final selected set + CV + sem; and the per-base marginal deltas. If final CV > champion by > 2·sem → PROMOTE
candidate (this is the look-outside payoff). Produce oof.npy, test_probs.npy, submission.csv, train.log; write
gates + cv/sem/folds into frontmatter. Do NOT submit (orchestrator decides).

## notes
