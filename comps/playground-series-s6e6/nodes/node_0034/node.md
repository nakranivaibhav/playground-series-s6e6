---
id: node_0034
desc: CatBoost on fs_realmlp_fe
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: gbdt
status: buggy
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.333333
gates: {schema_ok: false, oof_full: false, no_nan: false, dist_sane: false, leak_clean: false, cv_too_good: false, passed: false}
gate_note: "OOM kill during fold 4 — no artifacts produced. Fix: reduce depth or border_count to cut RAM usage."
leak: null
lb: null
submitted: null
created: 2026-06-07T13:50Z
decided: null
tags: [gbdt, catboost, fs_realmlp_fe, draft, cpu, oom]
---

## notes
OOM kill during fold 4 (PID 477789 killed at 2026-06-07T15:09 UTC).
Completed folds: fold0=0.969410 (best_iter=1152), fold1=0.968244 (best_iter=883),
fold2=0.967964 (best_iter=957), fold3=0.967892 (best_iter=763).
Partial 4-fold mean=0.968378 (std/sqrt(4)=0.000344).
Killed process RSS=29.5GB, total-vm=66GB — depth=8 + border_count=254 too expensive.
Fix: reduce border_count from 254 to 128 (halves per-split memory), or depth 8→6.
Do NOT re-launch without reducing memory footprint.

## plan
built on:   root (new draft — completes the GBDT trio on the rich FE alongside
            LightGBM node_0030 and XGBoost node_0031). Template src to COPY from
            node_0028/src (keeps the fs_realmlp_fe FE pipeline + fold-honest OOF/test_probs
            scaffold over the frozen folds.json); swap the model to CatBoost. node_0023 is
            the tuned-CatBoost recipe reference (the undertraining fix).
change:     replace the model with a well-tuned CatBoost on the SAME fs_realmlp_fe
            feature-set. Use native categorical handling (the integer-floor categorical
            views + cross-combos), balanced class weights, a proper iteration budget with
            early-stopping on the fold validation. Fold-honest OOF over the FROZEN
            folds.json → oof.npy (577347×3) + test_probs.npy (247435×3). CPU (prefer CPU to
            avoid GPU contention with node_0033/node_0035).
hypothesis: a de-correlated tree base (CatBoost's ordered boosting + native cats differs from
            LightGBM/XGBoost) on the rich FE completes the GBDT trio and gives the re-stack a
            3rd tree signal it currently lacks on fs_realmlp_fe.
target:     BA maximize; solo ≥ 0.964 (vs node_0023 0.962737 on fs_colors). Valuable if it
            lifts the re-stack vs champion node_0029 (0.969205) — re-run restack_probe.py.
