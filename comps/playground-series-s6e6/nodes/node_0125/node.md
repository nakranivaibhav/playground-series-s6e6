---
id: node_0125
desc: spatial kNN class-fraction base (falsification)
op: draft
parents: [root]
uses_data: [fs_realmlp_fe, fs_spatialknn]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.967323
sem: 0.000161
folds: [0.967760, 0.967521, 0.967428, 0.966934, 0.966971]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "FALSIFICATION VERDICT: gain over n030 (+0.000371) holds on holdout (working=0.967411 vs holdout=0.966971, delta=+0.000440 vs parent). NOT a mirage. But n0125 vs champion n091: -0.003032 (real loss per bootstrap). GALAXY recall +0.010 but STAR recall -0.014. stack-add to n091: -0.000829. err-corr vs n070: 0.808. Spatial kNN signal is real but this LightGBM is weaker than champion."
leak: clean
lb: null
submitted: null
created: 2026-06-18
decided: 2026-06-18T00:58Z
tags: [gbdt, lightgbm, spatial-knn, class-fraction, target-encode, falsification, fit_in_fold, outside, draft]
---

## plan
built on:   root — a NEW LightGBM base. Template: nodes/node_0030/src/solution.py (LightGBM on
            fs_realmlp_fe). ONE atomic change: ADD fs_spatialknn, a fit_in_fold spatial target-encode.
change:     Add fs_spatialknn (leak-safety: **fit_in_fold** — CRITICAL): for each row, the FRACTION of
            its K nearest sky-neighbours (KD-tree on (alpha, delta)) in each class, for K in {10,50,200}
            → 3 classes × 3 K = 9 features. The KD-tree is built on the TRAIN-FOLD rows ONLY; val/test
            rows query that train-fold tree; train rows SELF-EXCLUDE themselves from their own neighbour
            set. Never build the tree on full-train or test. (This is omadon's fold-safe construction;
            refs/scan_2026-06-17b/omadon_*.)
hypothesis: FALSIFICATION. omadon (refs, 2026-06-17) claims this fold-safe spatial class-fraction adds
            +0.003 BA ("biggest lever in the comp"). Our bank predicts a coordinate-value-reuse MIRAGE:
            n083 (class ~independent of coords, ~50% match < base rate), n060 (generator REUSES alpha/
            delta values 32-38% → kNN reads coord-reuse as in-fold structure, overfits), n013 (leak-safe
            positional regressed −6σ/5 folds). Test which is true with the holdout the comp board lacks.
target:     Balanced Accuracy maximize. THE TELL (validation.md gate, working-vs-holdout split is the
            falsification): if solo BA jumps on WORKING folds (0-3) but the HOLDOUT (fold 4) net-fix vs
            n030 is flat/negative, OR err-corr stays high with the gain confined to working — it is the
            in-fold coordinate-reuse mirage → mark dead, do NOT submit. ONLY if the gain HOLDS on the
            untouched holdout (working≈holdout lift) does it earn one LB-probe to confirm vs private.
            Cheap-kill fold-0 < 0.965.

OUTSIDE well, falsification. READ: refs/scan_2026-06-17b/omadon_s6e6-spatial-knn-class-fraction-features/
(the recipe); nodes/node_0013/node.md (the −6σ leak-safe positional precedent); nodes/node_0060/node.md
(the coord-reuse provenance finding); nodes/node_0030/src (base template); discussions.md 2026-06-18
entry. fs_spatialknn = fit_in_fold — the dev MUST verify the KD-tree is train-fold-only by reading its
own fold loop. CPU; minutes (KD-tree query on ~462k×3 K is fast). Report the WORKING-vs-HOLDOUT split
explicitly — that is the verdict.

## notes
well = outside. A falsification node — expected to die like n013, but concrete enough to test with our holdout.
