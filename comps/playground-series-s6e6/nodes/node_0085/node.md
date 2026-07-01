---
id: node_0085
desc: Physics redshift-locus monotone GBDT base
op: draft
parents: [root]
uses_data: [fs_realmlp_fe, fs_physics_locus]
family: gbdt
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966742
sem: 0.000275
folds: [0.967227, 0.967353, 0.966833, 0.965841, 0.966456]
baseline_cv: 0.970227
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "monotone constraints removed (60x slowdown in LightGBM multiclass); physics locus features stateless and clean"
leak: clean
lb: null
submitted: null
created: 2026-06-13T11:56Z
decided: 2026-06-13T14:34Z
---

## plan
built on:   root (wildcard framing draft); copies node_0030 (LightGBM richFE) for the pipeline and ADDS fs_physics_locus alongside fs_realmlp_fe.
change:     New fs_physics_locus (stateless): physically-motivated features — distance from the STAR locus (|redshift|≈0, expected color track), the QSO UV-excess color-box residual, and the GALAXY red-sequence/blue-cloud color-magnitude track residual — fed to a LightGBM with MONOTONE constraints on redshift (QSO recall ↑ with z, STAR recall ↑ as z→0). A physics-prior framing no current base encodes.
hypothesis: Physics-locus residuals + monotone-z constraints encode the true class manifold the synthetic label corrupts, giving a base whose errors decorrelate from the free-boundary bank models.
target:     balanced accuracy maximize; solo ~0.966+; counts if fwd-selected onto bank17+FT-T (cv > 0.970227 by >2·sem).

Wildcard well — a new representation+objective framing, not another generic base. The
three classes separate by KNOWN astrophysics: STAR sits at redshift≈0 on a stellar color
track; QSO shows UV excess (u-g low) at high z; GALAXY follows the red-sequence/blue-cloud
color-magnitude relation. fs_physics_locus encodes residuals/distances from these physical
loci (all row-wise deterministic from u,g,r,i,z,redshift → stateless, no fit/target).

Pair with monotone redshift constraints in LightGBM so the boundary respects the physical
z-ordering (STAR low / GALAXY mid / QSO high) — this regularizes the noisy synthetic label
toward physics and should decorrelate from every existing base (which learn the boundary
freely).

Copy node_0030 (LightGBM richFE) and ADD fs_physics_locus alongside fs_realmlp_fe; recipe
ports into src/clean.py as a new stateless function. Read data.md L68-72 (fs_research
already has a QSO color-box + redshift regime flags — REUSE those exact formulas as the
starting point rather than re-deriving), node_0006 research-features rationale, and
node_0030/src for the LightGBM pipeline. Solo aim ~0.966+ so the OOF is a fwd-select
candidate onto bank17+FT-T.

well: wildcard
