---
id: node_0006
desc: LightGBM + research features
op: improve
parents: [node_0001]
family: gbdt
uses_data: [fs_colors, fs_research]
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.965004
sem: 0.000083
folds: [0.965074, 0.964764, 0.964852, 0.965179, 0.965152]
baseline_cv: 0.964569
shuffled_cv: 0.33290
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, shuffle_collapsed: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-05T12:46Z
decided: 2026-06-05T12:58Z
tags: [lightgbm, feature-engineering, research-features]
---

## plan
built on:   node_0001 — its exact (CV-proven near-optimal) hyperparameters.
change:     add research-derived stateless features (all unit-tested in src/clean.py):
            extended/curvature colors (u_r,u_i,g_i,r_z,c_ug_gr,c_gr_ri), redshift transforms
            (log1p_redshift,is_star_z,is_highz), QSO color-box flags (qso_box,uv_excess),
            galactic coords (gal_l,gal_b). 28 features total. Saves test_probs.npy.
hypothesis: QSO color-box + redshift transforms attack the hard QSO↔GALAXY boundary.
target:     beat node_0001 0.964569 beyond fold-noise.

## notes
WINNER: cv 0.965004 = +0.00044 vs node_0001 (~3.9 champ-σ), improved on ALL 5/5 folds.
Features were the real lever (vs node_0005's failed tuning). Current champion.
oof.npy + test_probs.npy present → primary arm of the planned combine node (node_0007 blend).
Not yet submitted (best clean single model; blend at ~0.9656 is the stronger submit candidate).
