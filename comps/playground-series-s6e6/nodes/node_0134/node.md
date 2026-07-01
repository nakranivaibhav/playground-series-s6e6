---
id: node_0134
desc: Mitra finetune on rich FE
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: dead
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.954199]  # fold-0 only (20k context); below tier, not run to full 5-fold
baseline_cv: 0.333333
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: null, cv_too_good: null, passed: null}
gate_note: "FM-finetune family CLOSED. 50-step BA 0.9542 / 500-step BA 0.9519 (5x more FT = no change → context-bound, NOT under-trained). DECORRELATED (err-corr 0.61 vs champion) but too weak → fold-0 stack-add WASHES: -0.00033 over champion n091, +0.0001 over bare bank (noise). Same outcome as RBF/flux. With TabPFN-FT (full data) also 0.9589<tier, the FM family is thoroughly characterized as below-tier + stack-neutral on this 577k table. dead."
leak: null
lb: null
submitted: null
created: 2026-06-20T13:21Z
decided: null
tags: [nn, foundation-model, mitra, autogluon, finetune, fs_realmlp_fe, draft, gpu]
---

## plan
built on:   root (2nd finetuned-FM draft, sibling of node_0133). Reuses node_0133's
            FE machinery verbatim (stateless_fe + fit_fold_categoricals +
            add_target_encoding over frozen folds.json) → SAME fs_realmlp_fe inputs.
change:     swap TabPFN-v2 for Mitra (Amazon's tabular FM, via AutoGluon MitraModel,
            fine_tune=True). Driven per-fold by TabularPredictor(eval_metric=
            'balanced_accuracy', hyperparameters={'MITRA': {...}}) for fold-honest OOF.
hypothesis: TabPFN-v2 finetune (n133) reached only 0.9589 — below tier — because its
            SCM-only synthetic prior is far from this data's GBDT-shaped boundaries.
            Mitra pretrains on a MIXED prior (SCM + TREE-ENSEMBLE tasks), structurally
            closer to our load-bearing CatBoost/GBDT family, so its finetune has the
            best shot of any FM at actually REACHING tier (≥0.965) = a genuine new base.
target:     BA maximize; solo ≥ 0.965 tier. Cheap-kill fold-0 at BA<0.960 (same bar as
            n133). If it clears tier → pred_diagnostic err-corr vs n091 bank + stack test.
            If it ALSO caps below tier → strong 2nd data point that the FM family is
            closed here (SCM and tree-mix priors both under-reach on this 577k table).

## build protocol (staged cheap-kill)
1. SMOKE (MITRA_SMOKE=1): 20k-row subsample, fold-0, time-capped — verify AG+Mitra runs
   end-to-end + project timing. (Mitra weights download from HF on first use.)
2. FOLD-0 (MITRA_FOLD0=1): real fold-0 → tier read. Background + marker file.
3. FULL 5-fold: only if fold-0 clears tier → oof.npy + test_probs.npy + submission.csv.

## leakage discipline (same standard as parent n133/n33)
- Stateless FE once; factorize/KBins/TargetEncoder fit on train-fold only (fit_in_fold).
- AG does its own internal val split FROM the train fold for early-stop — never the OOF
  val_idx; folds from frozen folds.json. predict_proba columns reordered to CLASSES.

## notes
- 2026-06-20 fold-0: BA 0.954199, fit 345s on 22k-ctx subsample, val-predict 109s (115470 rows),
  proj 5-fold ~38min. AG-internal val (its own holdout of the subsample) 0.949. lr/epochs = Mitra defaults.
- 50k context infeasible (AG mem est 244GB); 20k is near the practical ceiling for this GPU/box.
- COMPARISON: frozen TabPFN 0.943-0.949 · TabPFN-FT (full data) 0.9589 · Mitra-FT (20k ctx) 0.9542.
  Finetuning lifts frozen but neither FM closes the ~1pp gap to the GBDT/TabM tier (0.965-0.968).
  Two different priors (SCM-only, tree-mix) both under-reach → consistent with the information ceiling.
- 2026-06-20 PROPER-FT TEST (fine_tune_steps 50->500, warmup 1000->100): BA 0.9519 ≈ 0.9542 (NO lift),
  err-corr 0.61 ≈ 0.62 (unchanged), stack-add over champion -0.00033 (still washes). 5x more finetuning
  changed NOTHING → Mitra is context-bound at ~0.952 (the 20k-row data deficit), not under-trained.
  err-corr probe: probes/mitra_errcorr.py · fold-0 stack proxy: probes/mitra_fold0_stack.py.
- DECISION: FM-finetune family closed. Decorrelated-but-weak, washes the stack like every prior
  weak-decorrelated base (RBF 0.53/flux 0.485). TabPFN-FT full-data 0.9589 already bounds the FM
  ceiling below the 0.965 tier, so pushing Mitra context higher (30k+) is futile. Finals untouched.
