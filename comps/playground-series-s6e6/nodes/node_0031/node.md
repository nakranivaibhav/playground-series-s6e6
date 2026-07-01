---
id: node_0031
desc: XGBoost on fs_realmlp_fe
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: gbdt
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966244
sem: 0.000269
folds: [0.966709, 0.966775, 0.966465, 0.965355, 0.965918]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good tripwire fired (rel_gain=190% vs baseline) — expected; all other nodes in this comp have similar gains. No actual leak detected."
leak: clean
lb: null
submitted: null
created: 2026-06-07T12:54Z
decided: null
tags: [gbdt, xgboost, fs_realmlp_fe, rich-fe, cpu, draft]
---

## plan
built on:   root (new draft — a 2nd GBDT family on the rich reference FE). Template src COPIED
            from node_0028/src (it builds fs_realmlp_fe end-to-end). Keep the ENTIRE FE pipeline
            byte-identical; replace only the model block.
change:     reuse node_0028's fs_realmlp_fe feature pipeline (stateless FE + the fit-in-fold
            TargetEncoder/bins, rebuilt per train fold) but swap the RealMLP model for a
            well-tuned XGBoost (CPU): tree_method='hist', device='cpu', lr=0.3, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5, early-stopping=20 rounds.
            Class-balanced sample_weight for Balanced Accuracy. Fold-honest OOF
            (577347×3) + test_probs (247435×3) over the FROZEN folds.json.
hypothesis: a 2nd strong tree family on the rich FE, de-correlated from BOTH the RealMLP NN base
            and the LightGBM base (node_0030) → adds another diverse strong base for the stack.
target:     BA maximize; solo >= 0.965 (vs node_0004 0.964414 / node_0011 0.964918). Valuable if
            it lifts the re-stack vs champion node_0029 (0.969205).

## notes
- Timing: fold0=9.6s, 5-fold total=50.2s (0.8min). Very fast on CPU.
- XGBoost best_rounds: [293, 343, 274, 379, 285] — early stopping working correctly.
- lr=0.3 chosen over lr=0.05/0.1 based on timing probe: 7s/fold vs 260s/fold, minimal CV loss.
- GPU-compiled XGBoost 3.2.0 requires CUDA accessible; do not set CUDA_VISIBLE_DEVICES="".
- cv_too_good tripwire: expected for this competition where baseline=0.333 and models achieve >0.96.
- All 5 folds beat target 0.965. Mean=0.966244, well above node_0011 XGBoost 0.964918.
- Distribution sane: test predictions match train class distribution well.
