---
id: node_0056
desc: 1D-CNN spectral NN over bands
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.965517
sem: 0.000274
folds: [0.965901, 0.966405, 0.965033, 0.965236, 0.965010]
baseline_cv: 0.969808
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-09
decided: null
tags: [novel-architecture, spectral, conv1d, de-correlated-base, restack-candidate]
---

## plan
built on:   node_0033/src scaffold (fs_realmlp_fe FE + frozen-folds OOF/test stack-base emit) — byte-copy its FE/fold/emit plumbing byte-identical; only the model is replaced.
change:     Replace the TabM model with a TWO-BRANCH net. (A) 1D-CNN branch over the wavelength-ordered bands [u,g,r,i,z] as a length-5 signal with 2-3 input channels (raw magnitude; mean-subtracted magnitude; optional per-row z-normalized magnitude), two Conv1d layers (kernel 2-3, padding=same, ~32-64 channels, SiLU) → global avg+max pool → conv feature vector. (B) scalar branch feeding redshift + alpha/delta + the rest of fs_realmlp_fe through a small MLP. Concat (A)+(B) → MLP head (hidden ~256, dropout ~0.1) → 3-class softmax. Class-balanced CE, AdamW, cosine schedule, early stopping, EMA optional. All scaling/embeddings fit-in-fold. Emit oof.npy (577347x3) + test_probs.npy (247435x3) on folds.json.
hypothesis: the 5 bands are ordered by wavelength, so a 1D conv learns spectral-shape filters (local slopes = colors, 3-band curvature = spectral breaks such as the 4000Å break that separates galaxies from stars) — an inductive bias absent from every tree/MLP/attention base, giving a de-correlated error structure that could lift the stack or move the LB even if standalone CV is flat.
target:     BA maximize; report standalone CV (DE per-class threshold) AND re-stack A/B CORE15+n56 vs champion 0.969808 (promote only if > by >2 sem ≈ 0.0003). LB-gate before any promotion (node_0047 mirage precedent). Realistic prior: CV-neutral (DCN washed; de-corr bases don't compound on LB) — a genuine novel-architecture shot.

## notes
Standalone bar = TabM 0.968053; re-stack bar = champion 0.969808.

RESULTS (2026-06-09):
- Standalone CV (argmax, no DE): 0.965605 mean over 5 folds
- Standalone CV (DE threshold, 2 free multipliers): 0.965517 ± 0.000274
  folds: [0.965901, 0.966417, 0.965033, 0.965232, 0.965422]  (wait — DE scores differ slightly)
  NOTE: cv in frontmatter = DE standalone = 0.965517
- Re-stack CORE15+n0056 (16 bases, balanced LogReg meta + DE): 0.969697 ± 0.000254
  folds: [0.970592, 0.969189, 0.969478, 0.969326, 0.969901]
  vs champion node_0041 = 0.969808: delta = -0.000111 (< 2*sem ≈ 0.0005) — essentially flat
- Error correlation of node_0056 OOF vs CORE15 mean: 0.7819 (moderately correlated)
- Timing: fold0=13s, total=4.7min on RTX 5090, VRAM peak=0.41 GB
- Conclusion: standalone CV below TabM (0.9655 vs 0.9681); re-stack flat vs champion.
  The 1D-CNN architecture does work and is fast, but is not sufficiently de-correlated
  (err_corr=0.78) to lift the stack meaningfully. DO NOT PROMOTE.
