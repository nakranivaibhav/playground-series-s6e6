---
id: node_0030
desc: LightGBM on fs_realmlp_fe
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: gbdt
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966952
sem: 0.000272
folds: [0.967515, 0.967653, 0.966815, 0.966282, 0.966497]
baseline_cv: 0.333333
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: true, passed: true}
gate_note: "cv_too_good warn is expected: 0.967 vs 0.333 random baseline; node_0006 (0.965) and node_0028 (0.970) both valid at this range. All fits are inside fold loop (no_global_fit_in_source: ok)."
leak: clean
lb: null
submitted: null
created: 2026-06-07T12:54Z
decided: null
tags: [gbdt, lightgbm, fs_realmlp_fe, rich-fe, draft]
---

## plan
built on:   root (new draft — a GBDT base on the rich reference FE). Template src to COPY from
            node_0028/src (it builds fs_realmlp_fe end-to-end: the heavy stateless FE +
            fit-in-fold TargetEncoder/integer-floor categorical bins + the fold-honest OOF /
            test_probs scaffold over frozen folds.json). Keep the ENTIRE FE pipeline
            byte-identical; replace only the model block.
change:     reuse node_0028's fs_realmlp_fe feature pipeline (stateless FE + the fit-in-fold
            TargetEncoder/bins, rebuilt per train fold) but swap the RealMLP model for a
            well-tuned LightGBM: native categorical handling on the integer-floor/category-cross
            views, balanced class handling (class_weight or sample weights for the BA metric),
            generous num_iterations with early-stopping on the fold's val split. Fold-honest OOF
            (577347×3) + test_probs (247435×3) over the FROZEN folds.json.
hypothesis: our LightGBM (node_0006, 0.965 on fs_research) was feature-limited, not model-limited;
            the richer fs_realmlp_fe should lift it AND it stays de-correlated from the RealMLP NN
            family → adds a strong tree base the stack can exploit.
target:     BA maximize; solo ≥ 0.966 (vs node_0006 0.965004). Valuable if it lifts the re-stack
            vs champion node_0029 (0.969205) — re-run restack_probe.py to confirm.
