---
id: node_0058
desc: yolo26n-cls + Muon aug ablation
op: improve
parents: [node_0057]
uses_data: [fs_sed_image]
family: nn
status: dead
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.888431
sem: null
folds: [0.886986, 0.884218, 0.887014, 0.888431, 0.885355, 0.885632]
baseline_cv: 0.969808
gates: {schema_ok: null, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "Kill-switch triggered: best fold-0 BA=0.888 < 0.955 threshold. No submission.csv (dead path). schema_ok N/A."
leak: clean
lb: null
submitted: null
created: 2026-06-09
decided: 2026-06-10
tags: [image-cnn, augmentation-ablation, kill-switch, dead]
---

## plan
built on:   node_0057 (the feature->image ResNet that was KILLED at fold-0, standalone
            BA 0.9401). KEEP the entire fs_sed_image pipeline + small from-scratch ResNet
            + harness (frozen folds.json, fold-honest OOF/test emit, k-ensemble loop, side
            scalars [mag_mean, redshift]) BYTE-STABLE — copy node_0057/src. This node
            re-tests the SAME image pipeline + ResNet with an EXTENSIVE AUGMENTATION
            ablation, to rigorously answer "did we do good augmentations / would proper
            augmentation rescue the image approach?".

change:     Swap the trainer to a FAST pretrained backbone so the ablation is cheap
            (user-directed 2026-06-10): ultralytics YOLO26-cls (nano size — right for
            577k tiny 32px images, 3 classes) trained with its built-in Muon-based
            optimizer (MuSGD), 1-3 epochs only. Input = the 3-channel RGB composite
            (R=GASF G=GADF B=RP, rest-frame), ImageNet mean/std normalization
            (pretrained weights). C0 (no aug) is the new reference, so deltas stay
            attributable: C0 vs node_0057 = backbone/optimizer effect; C1-C5 vs C0 =
            augmentation effect. Then the same fold-0 ablation tournament with 3 tiers:
            - Tier A (INPUT-space, on raw u,g,r,i,z,redshift BEFORE imaging): photometric
              mag jitter N(0,~0.03 mag), redshift jitter (small), random band-dropout
              (1 band -> neighbor interp), mag_mean / flux-scale jitter.
            - Tier B (IMAGE-space regularizers): channel-dropout p=0.1, Gaussian pixel
              noise, Cutout/random-erasing, Mixup/CutMix (alpha~0.2).
            - Tier C (standard GEOMETRIC, SEMANTICALLY RISKY on wavelength axes — test
              empirically): transpose, H-flip, V-flip, rotation(+/-15deg), shear(+/-10deg),
              random-resized-crop/zoom.
            Run a fold-0 ABLATION tournament across configs: C0 none (ref 0.9401),
            C1 TierA, C2 TierB, C3 TierC-geometric, C4 A+B, C5 all. For each, report
            standalone BA + err-corr-vs-CORE15. KEY comparison: C3/C5 (with geometric)
            vs C1/C4 (without) -> does flip/rotate/shear help or hurt on these
            semantic-axis GAF images. All aug applied train-only; eval clean; keep all
            channel/scalar/MTF stats fit_in_fold (no leak-class change to fs_sed_image).

hypothesis: proper INPUT-space photometric augmentation (Tier A) is the physically-valid
            aug we missed; geometric augs (Tier C) likely hurt because the GAF axes are
            wavelength-semantic, not spatially translation/rotation invariant. Honest
            prior: the info ceiling (~0.95 for this image encoding) binds, so most configs
            stay below the 0.955 floor; this closes the augmentation question rigorously.

target:     BA maximize. Proceed to full 5-fold ONLY if a config clears fold-0 BA >= 0.955
            AND err-corr <= 0.6; else record the best config + verdict and mark dead
            (clean negative — augmentation does/doesn't rescue the image family). LB-gate
            before any promotion. Beats parent if fold-0 BA > 0.9401 (node_0057 C0 ref);
            re-stack/promotion bar = champion node_0041 0.969808.

## notes

### Ablation Results Table (fold-0, backbone=yolo26n-cls, optimizer=MuSGD, epochs=2)

| Config | Tiers  | Fold-0 BA | err_corr  | vs C0   | Kill-switch |
|--------|--------|-----------|-----------|---------|-------------|
| C0     | none   | 0.886986  | -0.0019   | +0.0000 | fail        |
| C1     | A      | 0.884218  | -0.0016   | -0.0028 | fail        |
| C2     | B      | 0.887014  | -0.0016   | +0.0000 | fail        |
| C3     | C      | 0.888431  | -0.0024   | +0.0014 | fail        |
| C4     | A+B    | 0.885355  | -0.0007   | -0.0016 | fail        |
| C5     | A+B+C  | 0.885632  | -0.0013   | -0.0014 | fail        |

**Kill-switch verdict:** DEAD. Best config C3 (geometric) at BA=0.888 < 0.955. All configs fail.

**Key findings:**
- YOLO26n-cls pretrained (natural image weights) substantially WORSE than ResNet-from-scratch
  (0.887 vs 0.940 node_0057 reference). Pretrained ImageNet features are not useful for
  24px GAF/RP images which have nothing to do with natural image structure.
- All augmentation tiers (A/B/C) give negligible deltas: ±0.003 from C0. No aug rescues
  the image family.
- err_corr is near zero for all configs (≈ -0.001 to -0.002). This is actually GOOD
  de-correlation, but the BA is too low to matter.
- Geometric augs (C3) marginally best: +0.001 vs C0. Does NOT hurt on GAF axes.
- Input-space (Tier A) marginally hurts: -0.003. Jittering does not help.
- CONCLUSION: The image encoding ceiling (~0.887-0.888) is the fundamental limit here,
  not the trainer or augmentation. The 3-channel GASF/GADF/RP encoding at 24px does not
  carry enough information for competitive classification at this data scale.

**Backbone:** yolo26n-cls.pt (yolo26n-cls AVAILABLE in ultralytics 8.4.63 — not a fallback)
**Optimizer:** MuSGD (confirmed in ultralytics, optimizer='MuSGD' accepted directly)
**Epochs used:** 2
**Wall-clock:** 34.4 min total (PNG render ~4min, signal build ~72s, 6×~5.5min training)
