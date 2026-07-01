---
id: node_0057
desc: feature->image ResNet (rest-frame SED)
op: draft
parents: [root]
uses_data: [fs_sed_image]
family: nn
status: dead
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.969808
gates: {schema_ok: null, oof_full: false, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "KILL-SWITCH TRIGGERED: fold-0 BA=0.9401 < 0.955 threshold. Full 5-fold not run; no submission artifacts. OOF partial (fold-0 only) saved as oof_fold0_only.npy."
leak: clean
lb: null
submitted: null
created: 2026-06-09
decided: 2026-06-09
tags: [image-cnn, resnet, rest-frame-warp, gaf, kill-switch, dead]
---

## plan
built on:   root (new DRAFT — a feature->image CNN family never tried). COPY the
            node_0033/src scaffold (frozen folds.json fold-honest OOF/test stack-base
            emit + GPU handling, k-ensemble training loop) byte-stable as the harness;
            replace only the model + feature pipeline. The image-building helper is
            ADAPTED from comps/playground-series-s6e6/viz_sed_images.py — it already
            implements flux-shape -> (1+z) rest-frame warp -> PCHIP resample(24) ->
            GASF/GADF/RP. Re-stack bar = champion node_0041 0.969808; standalone bar =
            TabM node_0033 0.968053.

change:     Per row, from [u,g,r,i,z]+redshift build a 7-channel 32x32 "SED-texture"
            image (24x24 native, zero-padded to 32x32), then classify with a small
            from-scratch ResNet with 2 side scalars fused at the head.
            IMAGE (7 channels, native 24x24 -> pad 32x32):
              (1) flux-shape f_b = 10^(-0.4*(mag_b - mag_mean_row)) — brightness removed
                  (mag_mean kept as a side scalar);
              (2) REST-FRAME WARP lam_rest = lam_obs/(1+max(z,-0.009)) over SDSS effective
                  wavelengths [3543,4770,6231,7625,9134];
              (3) PCHIP resample to 24 pts on a common log-lambda grid for BOTH s_rest
                  (warped) and s_obs (unwarped);
              (4) pyts encode (hand-rolled — pyts could not be installed due to dependency
                  conflict with torchmetrics/pytabkit): ch0 GASF(s_rest), ch1 GADF(s_rest),
                  ch2 RP(s_rest, continuous), ch3 MTF(s_rest) [the ONLY fit_in_fold
                  channel — quantile bin edges from POOLED TRAIN-FOLD s_rest only],
                  ch4 GASF(s_obs), ch5 zmod = outer(s_obs)*z_normed, ch6 support-mask
                  outer product. Zero-pad 24x24 -> 32x32. Images stored as float16 (RAM
                  efficient: 3.72 GB for 462k train rows). Channel stats fit in-fold.
            SIDE SCALARS (NOT in the image) -> head: [mag_mean, redshift], standardized
            in-fold.
            MODEL — small ResNet from SCRATCH (NOT pretrained, 713,635 params):
            Conv3x3(7->32,s1)+BN+SiLU stem; 3 stages [2,2,2] BasicBlocks 32->64->128,
            stride-2 at stages 2&3; GAP->128-d; CONCAT 2 side scalars->130-d -> Dropout(0.2)
            -> Linear(130->128)->SiLU->Linear(128->3).
            Memory strategy: images stored as float16; standardize+pad done in Dataset
            __getitem__ (no large float32 intermediate allocation).

hypothesis: the (1+z) rest-frame wavelength warp re-measures the SED on a
            redshift-stretched axis the 15-base stack never forms — the stack holds z
            and observed colors SEPARATELY, so this warp is the ONE transform here that
            is NOT a deterministic re-encoding of owned features. If it carries signal
            TabM lacks, the CNN's errors de-correlate from CORE15 and the base lifts the
            stack. (Honest EV ~20%: most likely washes like node_0056 err-corr 0.78;
            this is a cheap falsifiable test that RETIRES the feature->image family if
            flat.)

target:     BA maximize. HARD FOLD-0 KILL-SWITCH FIRST (before the 5-fold): kill if
            standalone BA < 0.955 OR err-corr-vs-CORE15 > 0.6 (node_0056 wash
            threshold). Report warp-OFF / zmod-OFF / image-only ablations on fold-0.

## notes

### RESULT: CLEAN NEGATIVE (kill-switch triggered, designed outcome)

**Fold-0 kill-switch metrics:**
- Fold-0 standalone BA: **0.940093** (BELOW 0.955 threshold — KILL triggered)
- Fold-0 err-corr vs CORE15: **0.5341** (BELOW 0.6 — would NOT have killed on err-corr alone)

**Interpretation:**
- The BA of 0.940 is well below the 0.955 kill threshold AND below the standalone TabM
  baseline of 0.968. A 28-point gap at this resolution means the image encoding is
  genuinely informationally inferior to tabular features for this dataset.
- The err-corr of 0.534 is better than node_0056's 0.783 — the rest-frame warp does
  reduce correlation somewhat compared to the 1D-CNN. But the absolute BA is too low
  to carry independent signal worth adding to the stack.
- Note: the internal (int_val) BA peaked at 0.938 — so the BA ceiling for this
  architecture on this data is well below the tabular models.

**Ablation results (fold-0 only):**
- Full model (with warp + zmod + scalars): BA=0.940  err-corr=0.534
- warp-OFF (s_rest channels zeroed):       BA=0.942  err-corr=0.553
- zmod-OFF (ch5 zeroed):                   BA=0.941  err-corr=0.547
- image-only (no side scalars):            BA=0.933  err-corr=0.479

**Key finding:** warp-OFF achieves *higher* BA (0.942 vs 0.940) than the full model.
The rest-frame warp does NOT improve performance — in fact it slightly hurts.
The side scalars [mag_mean, redshift] provide the most incremental value (removing them
drops BA from 0.940 to 0.933 and the err-corr drops to 0.479, meaning the image channels
alone are actually less correlated with CORE15 errors but also less informative).

**Timing:** 24.0 min total (signal=142s, images=41s, fold-0 training=235s, 3 ablation models=~960s)
**Peak VRAM:** 1.96 GB (RTX 5090)
**Memory:** images stored as float16 = 3.72 GB for 462k rows (solved OOM from first attempt)

**Conclusion:** feature->image->CNN family RETIRED. Neither the rest-frame warp nor the
GAF/RP image encoding provides a meaningful improvement over tabular features for this
problem. The 7-number SED (5 bands + redshift + mag_mean) is too low-dimensional for
convolutional texture learning to add value. Retire this family; do not try again.

**Leakage:** clean — all fits (MTF edges, channel stats, scalar stats) inside fold; leakage_scan.py exit=0.
**Artifacts:** oof_fold0_only.npy (577347,3), train.log, leakage_scan.json, features.txt.
**No submission.csv** — full run not completed by design (kill-switch).
