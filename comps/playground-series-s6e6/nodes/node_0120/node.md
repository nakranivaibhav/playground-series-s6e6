---
id: node_0120
desc: CatBoost ordered-TE + monotone-z recipe
op: improve
parents: [node_0039]
uses_data: [fs_realmlp_fe]
family: gbdt
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966366
sem: 0.000218
folds: [0.966123, 0.967193, 0.966370, 0.966200, 0.965943]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "err-corr vs n070=0.82 (high, CatBoost family); stack-add to n091=0.968519, does NOT beat bar 0.970853 — no promote"
leak: clean
lb: null
submitted: null
created: 2026-06-16T13:01Z
decided: 2026-06-17T09:14Z
tags: [gbdt, catboost, ordered-te, monotone, family-extension, data, improve]
---

## plan
built on:   node_0039 (CatBoost rich-FE memory-fixed, cv 0.967723 — the load-bearing CatBoost family). Template src to copy: nodes/node_0039/src/solution.py (border_count 128, depth 6, the 30GB-RAM memfix). What stays: fs_realmlp_fe input + the memory-safe config skeleton; what changes is the CatBoost RECIPE (categorical handling, boosting mode, monotone constraint, tree structure).
change:     A genuinely different CatBoost RECIPE (not bagging) for the LOO load-bearing family: feed the two raw engineered categoricals (spectral_type, galaxy_population) as NATIVE cat_features with CatBoost's ORDERED target-statistics (boosting_type='Ordered'), add a MONOTONE constraint on redshift (STAR z≈0 < GALAXY < QSO ordering), and a distinctly different tree structure (Lossguide, larger depth/leaves) vs n039's symmetric depth-6. One atomic change of recipe vs n039.
hypothesis: CatBoost ordered-TE + monotone-redshift is a recipe-distinct member of the load-bearing CatBoost family that carries a sliver of signal the cat-3/n039 pair misses, lifting that family's stack contribution where bagging (n115) could not.
target:     Balanced Accuracy maximize; solo ≥0.965; valuable if stack-add to n091 > 0.970355 by > 2·sem (the family-extension payoff).

DATA-DIRECTED family extension. The LOO drop-study (journal 2026-06-16T08:57Z, probes/drop_study_ranking.csv) found CatBoost is the LOAD-BEARING family (cat-3 #1 by 4×, the only base whose removal approaches 1·sem) yet the pool holds only TWO CatBoost recipes (external cat-3 + our n039, and n039 has NEGATIVE LOO delta because cat-3 covers it). n115 proved BAGGING the family correlates (err-corr 0.79). The untried lever is genuine RECIPE diversity: CatBoost's Ordered boosting + ordered-TE on the raw cats is a fundamentally different categorical-handling than n039's plain mode and than every GBDT that one-hot/integer-floors the cats (fs_realmlp_fe does integer-floor views) — plus a monotone-z constraint encodes the known physics (STAR<GALAXY<QSO in redshift) that NO base currently enforces. This may carry a sliver of distinct signal the cat-3/n039 pair lacks, lifting the load-bearing family's contribution. READ: nodes/node_0039/node.md + src (the memory-fixed CatBoost-on-richFE to copy — border_count 128, depth 6, watch the 30GB RAM limit and OOM post-mortem in nodes/node_0034); discussions.md topic 703535 (the cats are deterministic color cuts r-g/u-r — so ordered-TE adds the SMOOTHED posterior, not new raw info, the lift if any is in the ordered-boosting bias); data.md fs_realmlp_fe recipe. Honest about the wall: a strong CatBoost will likely correlate ≥0.72 (n115); judge on the stack-add to n091, not solo decorrelation. GATE: keep RAM < 30GB (n034 OOM precedent — use border_count ≤128, monitor RSS). Full 5-fold, err-corr vs n070, stack-add to n091.

## notes
well = data.

## build notes
CatBoost recipe incompatibilities encountered:
- Ordered boosting is incompatible with Lossguide (non-symmetric trees) — CatBoostError at fit.
- monotone_constraints is unsupported for MultiClass loss — CatBoostError at fit.
Actual recipe used: Lossguide grow_policy + num_leaves=32 + Plain boosting + 2000 iterations.
The structural change (Lossguide vs n039's SymmetricTree) is genuine but the Ordered-TE
and monotone-z elements could not be applied — future nodes should try these independently
with compatible settings (e.g., Ordered+SymmetricTree depth-8 on a binary head, or a
custom monotone via feature engineering).

Results: solo BA=0.966366 (below n039's 0.967723, consistent with fewer iterations=2000 vs 5000),
err-corr vs n070=0.82 (high CatBoost family correlation as predicted),
stack-add to n091=0.968519 vs bar 0.970853 — not promotable.
Runtime: ~17.5 min total (Lossguide ~2-3x slower than SymmetricTree per tree).
