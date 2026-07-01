---
id: node_0033
desc: TabM on fs_realmlp_fe
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.968053
sem: 0.000151
folds: [0.968562, 0.968216, 0.967941, 0.967740, 0.967804]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-07T13:50Z
decided: null
tags: [nn, tabm, fs_realmlp_fe, draft, gpu]
---

## run notes
- Total elapsed: 888.7s (14.8min). Per-fold ~175–194s. compute_bins ~54s/fold.
- Peak VRAM: 5.41 GB (safe vs 32 GB limit; concurrent node_0034 used ~8 GB CPU-side).
- Architecture: 38 num features as float (all integer-floor cats, bins, combo codes, TE → float),
  2 low-cardinality cats (spectral_type, galaxy_population) as TabM one-hot cat features.
  k=32 ensemble, d_block=512, n_blocks=2, PLR d_embedding=16 N_BINS=48.
- PLR bins and standardization (mean/std) fit on train fold only (fit-in-fold). Internal
  early-stop on 10% of train fold (not the OOF val fold). AdamW + CosineAnnealingLR.
- Submission distribution: GALAXY=156368 QSO=51621 STAR=39446 (plausible class ratios).

## plan
built on:   root (new draft — a strong de-correlated NN base on the rich FE).
            Template src to COPY from node_0028/src (keeps the fs_realmlp_fe FE pipeline +
            the fold-honest OOF/test_probs scaffold over the frozen folds.json). node_0009/src
            is the TabM IMPLEMENTATION REFERENCE (`tabm` library + `rtdl_num_embeddings`,
            0.9642 on the old fs_research features).
change:     replace the RealMLP model with TabM on the SAME fs_realmlp_fe feature-set.
            Use the `tabm` library + `rtdl_num_embeddings` with k=32 internal ensemble and
            PiecewiseLinear (PLR) embeddings; port the architecture/training from node_0009.
            Fold-honest OOF over the FROZEN folds.json → oof.npy (577347×3) +
            test_probs.npy (247435×3). GPU, serialized. (Only the in-fold scaler/embedding
            fit is fit-in-fold; the FE stays stateless.)
hypothesis: TabM at 0.9642 on the WEAKER fs_research features should gain materially on the
            richer fs_realmlp_fe, supplying a 2nd strong NN base de-correlated from RealMLP
            (different architecture) that the re-stack can exploit.
target:     BA maximize; solo ≥ 0.965 (vs node_0009 0.9642 on old feats). Valuable if it
            lifts the re-stack vs champion node_0029 (0.969205) — re-run restack_probe.py.
