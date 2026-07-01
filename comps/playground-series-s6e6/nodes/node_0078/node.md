---
id: node_0078
desc: honest post-bag STAR-recall prior calibration
op: improve
parents: [node_0070]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970054
sem: 0.000238
folds: [0.970946, 0.969912, 0.969595, 0.969732, 0.970087]
baseline_cv: 0.970211
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "CV 0.970054 < parent 0.970211 (lift=-0.000157, 1*sem=0.000238); DE multipliers HURT on bagged OOF in all 5 folds (delta range -0.000052 to -0.000214); same finding as n69 — argmax beats DE after bagging at any level"
leak: clean
lb: null
submitted: null
created: 2026-06-13T09:13Z
decided: 2026-06-13T09:55Z
---

## plan
built on:   node_0070 (bank17+FT-T stacked OOF) — kept; this adds a post-bag calibration step.
change:     On the bagged bank17+FT-T stacked OOF, fit per-class prior MULTIPLIERS (the Deotte 'BOOST' knob, esp. STAR/QSO minority recall) via differential_evolution maximizing balanced accuracy, but fit them NESTED on the BAGGED (not single-seed) OOF so the calibration survives bagging — the lever that died as a single-seed DE-threshold in n69.
hypothesis: the per-class prior multiplier carries real minority-recall signal that was masked by single-seed noise; refitting it on the variance-reduced bagged OOF recovers the metric edge in a form that generalizes to private.
target:     Balanced Accuracy maximize; beats node_0070 if bagged+refit-multiplier CV > 0.970211 and the multipliers' OOF lift exceeds 1·sem.

Balanced accuracy = macro per-class recall; minority STAR(14%)/QSO(20%) recall is the metric's binding constraint (research.md balanced-accuracy levers; discussions topic 704512 ceiling recipe uses exactly DE over 3 per-class multipliers bounds 0.1-5). Our DE-threshold edge gave champion its LB lead over Deotte's argmax (0.97073 vs ~0.97028) BUT n69 found it does NOT survive seed-bagging — it was fitting single-seed fold noise. The untried honest version: refit the per-class multipliers on the BAGGED OOF (averaged over 5 seeds) so they capture stable prior structure, not seed noise.

READ: nodes/node_0070/src (stacked OOF + DE-threshold fit), nodes/node_0069/node.md (post-bag DE-threshold death lesson + bagging mechanics), discussions.md topic 704512 (DE multiplier recipe). Distinct from n69: n69 bagged THEN argmax (dropped threshold); this bags THEN refits the multiplier on the bagged OOF. Fit nested fold-honest, never on full train. CPU, minutes.

well: data.
