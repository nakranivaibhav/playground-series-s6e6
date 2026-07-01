---
id: node_0055
desc: DCN/CrossNet NN base richFE
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966037
sem: 0.000301
folds: [0.967070, 0.966128, 0.965274, 0.965651, 0.966061]
baseline_cv: 0.969808
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.97083  # CORE15+n55 re-stack probe (standalone DCN unsubmitted)
submitted: null
created: 2026-06-09
decided: null
tags: [nn, dcn, crossnet, revival, re-stack-candidate]
---

## plan
built on:   copy node_0033/src scaffold byte-identical (fs_realmlp_fe FE + frozen-folds.json OOF/test emit loop); swap ONLY the model.
change:     Replace TabM with the DCN from `refs/nn-v2-for-s6e6.py`: DenseInput (per-cat
            embeddings) → CrossNet (explicit feature-cross layers) PARALLEL with a deep
            MLP block, concat → head. Per-class-balanced CE + label smoothing, EMA,
            AdamW + cosine LR. qbin embeddings + low-freq artifact-count features built
            FIT-IN-FOLD (train-fold reference only). DROP the SDSS17 external append
            entirely (append_original=False) — NO external data. Emit oof.npy (577347×3)
            + test_probs.npy (247435×3) on folds.json.
hypothesis: An explicit-feature-cross architecture (DCN) is de-correlated from RealMLP/TabM
            and may add a signal the meta-stack can exploit beyond the saturated NN slot.
target:     Balanced Accuracy (maximize). Standalone CV reported (standalone bar = TabM
            node_0033 0.968053). Re-stack A/B CORE15+n55 vs champion 0.969808 — lifts ONLY
            if > 0.969808 by >2·sem (~0.0003); LB-gate before promotion (node_0047 mirage
            precedent).

## notes
re-stack A/B: CORE15+n55 = 0.969794 vs champ 0.969808 (delta=-0.000014 < threshold ~0.0003; DCN does NOT lift the saturated stack).

standalone DE-threshold BAC: 0.966037 ± 0.000301 vs TabM node_0033: 0.968053 (DCN is -0.002016 below TabM).
Total elapsed: 18.5min (5 folds, early stopping at epoch 21). Peak VRAM: 0.24 GB.

re-stack candidate: baseline_cv 0.969808 is the champion stack to beat via re-stack, not the
standalone bar. NEVER promote on CV alone — spend an LB gate first (node_0047 CV mirage: CV
+0.001 / LB −0.008). fit_in_fold features (qbin embeddings, low-freq artifact counts) must be
built inside each train fold only.
