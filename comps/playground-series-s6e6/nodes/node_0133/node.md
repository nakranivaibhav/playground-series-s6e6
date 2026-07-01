---
id: node_0133
desc: TabPFN-v2 finetune on rich FE
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: running
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.958857]  # fold-0 only (tier read); full 5-fold not yet run
baseline_cv: 0.333333
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: null, cv_too_good: null, passed: null}
gate_note: "fold-0 BA 0.958857 < 0.960 tier bar — below GBDT/TabM (0.965-0.968), at frozen-TabICL level. BUT best TabPFN here (+0.010 vs frozen v2.5 0.949): finetuning mechanism VALIDATED. Standalone-tier: killed. Open: decorrelation value (needs full-5fold OOF, 3.7hr) vs pivot to Mitra arm. DECISION PENDING."
leak: null
lb: null
submitted: null
created: 2026-06-20T07:41Z
decided: null
tags: [nn, foundation-model, tabpfn, finetune, fs_realmlp_fe, draft, gpu]
---

## plan
built on:   root (new draft — a genuinely untried FAMILY: a gradient-finetuned
            tabular foundation model). Reuses node_0033/src FE machinery verbatim
            (stateless_fe + fit_fold_categoricals + add_target_encoding over the
            frozen folds.json) so the model sees the SAME best feature-set
            (fs_realmlp_fe) every strong NN base used.
change:     replace the TabM model with `tabpfn.finetuning.FinetunedTabPFNClassifier`
            (TabPFN-v2, gradient finetune of the pretrained transformer). This is the
            FIRST time any foundation model is FINETUNED here — n22/n25/n27 (TabPFN)
            and n26 (TabICL) were all FROZEN in-context (capped ~10k context on 577k
            rows → weak, 0.942–0.959). Large-data subsampling handled natively:
            n_finetune_ctx_plus_query_samples=10k chunks, n_inference_subsample_samples=50k
            support, n_estimators_final_inference=8 subsample-ensemble.
hypothesis: finetuning adapts the pretrained prior to OUR distribution and lifts the
            frozen-TabPFN 0.949 ceiling toward tier (literature: FT gains scale with
            dataset size, largest on big tables like our 577k). Even if it caps below
            the GBDT/TabM tier, a foundation-transformer is a NEW REPRESENTATION — the
            journal's only historical source of decorrelation (flux 0.485, RBF 0.53,
            FT-T 0.53) — so a weak-but-decorrelated base still has stack value.
target:     BA maximize; solo ≥ 0.965 tier (vs frozen TabPFN n25 0.949). Cheap-kill on
            fold-0: if BA < 0.960, kill the finetune-family draft (like n103/n57/n81).
            If it clears tier, run pred_diagnostic.py err-corr vs the n091 bank.

## build protocol (staged cheap-kill — TabPFN FT is expensive)
1. SMOKE (TABPFN_SMOKE=1): 30k-row subsample, 3 epochs, tiny subsample — verify the
   pipeline runs end-to-end + project per-fold timing. Seconds-to-minutes.
2. FOLD-0 (TABPFN_FOLD0=1): real fold-0 only → tier read + true timing projection.
   Background + marker file (train is multi-minute). Cheap-kill at BA<0.960.
3. FULL 5-fold: only if fold-0 clears tier → oof.npy (577347×3) + test_probs.npy
   (247435×3) + submission.csv over the frozen folds.

## leakage discipline (same standard as parent n33)
- Stateless FE computed once (no target, no cross-row stats).
- factorize maps, delta KBins, TargetEncoder: fit on train-fold rows only (fit_in_fold).
- FinetunedTabPFNClassifier does its OWN internal 10% val split FROM the train fold for
  early-stop — never touches our OOF val_idx; folds loaded from frozen folds.json.
- No standardization (TabPFN does its own preprocessing).

## notes
- 2026-06-20 fold-0 tier read: BA=0.958857, fit 2548s (~42min), val-predict 144s, peak VRAM 12.3GB.
  Projected full 5-fold ≈224min (3.7hr). lr=3e-5, 30ep (no early-stop trigger), ctxq=10k, infer-sub=50k.
- RESULT vs frozen TabPFN: n22 v2 0.943, n25 v2.5 0.949 → FINETUNE 0.9589 (+0.010). Mechanism works,
  caps below tier. Sits in the gap: stronger than the weak-decorrelated bases that failed to stack
  (RBF n096 0.947/corr0.53 stack-add -0.00005; flux n103 0.940/corr0.485) but weaker than tier (0.965+).
- Decorrelation unknown (no OOF yet). The ONE scenario it has stack value: corr<0.65 AND strong enough
  at 0.9589 to flip stack-add positive where 0.94-tier decorrelated bases couldn't. Wall prior: likely wash.
