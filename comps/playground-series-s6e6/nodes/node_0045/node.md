---
id: node_0045
desc: RealMLP-ref + orig-priors (upgrade n28)
op: improve
parents: [node_0028]
uses_data: [fs_realmlp_fe, fs_origprior]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969050
sem: 0.000322
folds: [0.970310, 0.968696, 0.968767, 0.968536, 0.968942]
baseline_cv: 0.969065
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-08T15:33Z
decided: null
tags: [realmlp, original-priors, upgrade-in-place]
---

## plan
built on:   node_0028 (our breakthrough RealMLP-reference recipe, cv 0.969065, the
            BIGGEST contributor to the champion stack) — keep its FE (fs_realmlp_fe),
            architecture (n_ens=8, [512]³, PBLD embeddings, ScalingLayer, EMA, flat_cos),
            preprocessing, epochs, everything BYTE-IDENTICAL.
change:     ADD the original-SDSS17 prior features (fs_origprior) to node_0028's input
            matrix — the ONE atomic change. P(class|color-bin-key) computed on
            data/sdss17/star_classification.csv ONLY (orig rows NOT appended; orig labels
            disjoint from our train/val → leak-clean), reusing the builder in
            nodes/node_0044/src/fs_zoo.py:add_original_prior_features. Concatenate these
            prior columns to the existing fs_realmlp_fe matrix before the RealMLP sees it;
            scale them with the same robust preprocessing. Rebuild OOF on OUR folds.json.
hypothesis: Unlike ADDING a redundant base (which washes), UPGRADING our strongest
            included arm IN-PLACE with real-data class-structure signal makes the SAME
            arm stronger without adding stack correlation — so swapping n45 for n28 in
            CORE15 should LIFT the stack toward the 0.97126 public cluster.
target:     Balanced Accuracy (maximize) · solo CV beats node_0028 (0.969065); decisive
            test = re-stack CORE15 with n28→n45 swap beats champion node_0041 (0.969808)
            by >2·sem. Stack value proven in a follow-up combine.

## notes
Reference orig-prior recipe: refs/xgb-v5-for-s6e6.py + nodes/node_0044/src/fs_zoo.py
(add_original_prior_features). Orig data: data/sdss17/star_classification.csv (drop
-9999 placeholder rows). Leakage: priors on orig only; NEVER fit on train/val/test
labels; the RealMLP's own preprocessing (median_center/robust_scale) is fit in-fold.
