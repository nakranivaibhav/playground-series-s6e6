---
id: node_0118
desc: generator decimal-fingerprint GBDT base
op: draft
parents: [root]
uses_data: [fs_genfp, fs_realmlp_fe]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966266
sem: 0.000166
folds: [0.966681, 0.966594, 0.965779, 0.966142, 0.966133]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-16T13:01Z
decided: 2026-06-16T13:54Z
tags: [gbdt, lightgbm, generator-forensics, decimal-fingerprint, data, draft, fs_genfp]
---

## plan
built on:   root — a NEW LightGBM base. Template src to copy: nodes/node_0030/src/solution.py (LightGBM on rich fs_realmlp_fe, cv 0.966952). The change is the feature INPUT: fs_realmlp_fe AUGMENTED with the new stateless fingerprint set fs_genfp.
change:     Train a LightGBM base on fs_realmlp_fe AUGMENTED with a NEW stateless feature-set fs_genfp (leak-safety: stateless): per-feature synthetic-generator quantization fingerprints — for each base numeric (u,g,r,i,z,redshift,alpha,delta), the number of significant decimals, the fractional/mantissa residual after rounding to k places, last-digit value, and trailing-zero count. NO target, no cross-row stat → stateless. fs_genfp is row-wise deterministic.
hypothesis: the synthetic generator leaves class-conditional decimal/quantization fingerprints distinct from the photometric SED signal, giving a GBDT a partly-orthogonal axis the all-photometric bank never sees.
target:     Balanced Accuracy maximize; cheap-kill if fold-0 solo BA < 0.962; valuable only if solo ≥0.965 AND stack-add to n091 > 0.970355 by > 2·sem.

DATA-CENTRIC, generator-forensics well. The data is FULLY synthetic: journal 2026-06-16T12:20Z + research.md confirm 0 row-identity match to the real SDSS17 at every precision and SHIFTED marginals (u [-0.14,28.25] vs real [9.82,32.78], redshift mean 0.723 vs 0.577) → a tabular generator (GAN/diffusion, the Playground norm) RESHAPED the distributions. Such generators leave CLASS-CONDITIONAL float-quantization artifacts (different sampling/rounding per latent class). n060 only tested coordinate-VALUE membership in the real catalog (washed) and the public DCN kernel added crude art_*_floor tokens — but NOBODY has fed the per-feature decimal/mantissa fingerprint as a primary feature to a dedicated base. This is a NON-photometric representation the decorrelation wall (proven for flux/RBF/residual/colors, all photometric) never touched. READ: nodes/node_0030/node.md + its src (the LightGBM-on-richFE base to copy); nodes/node_0060/node.md (the provenance-flag precedent — what washed and why, so you build the digit-level version it stopped short of); data.md fs_realmlp_fe recipe (refs/realmlp-v5-for-s6e6.py) for the base FE. GATE: this is high-risk — first verify fold-0 solo BA ≥ tier (cheap-kill <0.962) AND check the fingerprint features carry train↔test-CONSISTENT signal (compute a single-feature↔class AUC on a sample; if a fingerprint perfectly predicts class it is a generator leak that won't hold on private — flag, don't celebrate). If solo BA passes, full 5-fold + err-corr vs n070 + stack-add to n091. fs_genfp = stateless; emit oof/test/submission.

## notes
well = data.

RESULT (NULL — does not promote, but a clean honest base). Solo 5-fold BA 0.966266 — passes the cheap-kill (≥0.965), so the decimal/mantissa fingerprints DO carry tier-level signal. Leak check clean: worst single-fingerprint↔class AUC 0.516532 (_fp_redshift_trail0) — no fingerprint perfectly predicts class, so no generator-leak smell; train↔test consistent. BUT err-corr vs n070 = 0.8112 (correlated, not orthogonal) and stack-add to n091 = +0.000015 (a wash, well under 2·sem). Verdict: the generator-fingerprint axis is real but NOT decorrelated from the photometric bank — it re-encodes the same class signal through quantization rather than adding an independent axis. Confirms the decorrelation↔strength wall again: a 5th+ angle still lands ≥0.72 corr at tier strength.
