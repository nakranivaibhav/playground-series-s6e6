---
id: node_0020
desc: balanced logreg stack + DE threshold
op: combine
parents: [node_0006, node_0004, node_0001, node_0009, node_0011, node_0003, node_0019, node_0016, node_0014]
uses_data: []
family: ensemble
status: valid
stage: submitted
metric: Balanced Accuracy Score
direction: maximize
cv: 0.966627
sem: 0.000221
folds: [0.967359, 0.966051, 0.966481, 0.966409, 0.966835]
baseline_cv: 0.965889
shuffled_cv: null
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.96722
submitted: 2026-06-06
created: 2026-06-06T14:06Z
decided: 2026-06-06T14:16Z
tags: [stacker, logreg-meta, de-threshold]
---

## plan
built on:   the OOF probability matrices of 9 valid base nodes — node_0006, node_0004,
            node_0001, node_0009, node_0011, node_0003, node_0019, node_0016, node_0014
            (each base's per-class OOF stays byte-identical; this node only learns a
            meta-model on top, it re-trains none of them).
change:     STACKER, not a probability-average blend. Build the meta feature matrix as
            log(clip(p, 1e-7, 1)) of each base's OOF probabilities, concatenated → 9 bases
            × 3 classes = 27 columns. Meta = sklearn LogisticRegression(class_weight='balanced',
            C=1.0, max_iter=2000) (multinomial). Fold-honest: for each held-out fold, fit the
            meta on the OTHER 4 folds' OOF, predict the held fold → honest stacked OOF. Then
            per-class threshold calibration via scipy differential_evolution over a 3-vector
            w=(w_GAL, w_QSO, 1.0), bounds (0.1, 5), maximizing balanced accuracy, fit on the
            other folds' stacked OOF and applied to the held fold (argmax(prob*w)). For TEST:
            fit meta on the full stacked OOF, fit w on the full stacked OOF, apply to the
            stacked base test_probs. Reference impl already validated in `stack_probe.py`.
hypothesis: a balanced multinomial LogReg stack reweights the per-class recalls that balanced
            accuracy rewards (a plain meta inherits majority bias); the DE threshold adds a
            final continuous calibration. Discussions confirm this is the top-of-LB recipe.
target:     Balanced Accuracy Score (maximize) · beats champion node_0010 (0.965889) beyond
            2·sem — probe shows honest CV 0.966627 ± 0.000221 = +0.000738 (~4 sem).

## notes
Reference implementation validated in `comps/playground-series-s6e6/stack_probe.py`
(honest CV = 0.966627 ± 0.000221, +0.000738 vs champion node_0010, ~4 sem). Consumes base
OOF only, so `uses_data: []` — the data lineage is the `combine` edges in graph.md.
