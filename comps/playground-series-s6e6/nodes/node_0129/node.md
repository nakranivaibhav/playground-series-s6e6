---
id: node_0129
desc: ensemble-of-ensembles LogReg meta over 6 stacks
op: combine
parents: [node_0091, node_0070, node_0116, node_0063, node_0041, node_0040]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970410
sem: 0.000248
folds: [0.971224, 0.970174, 0.969971, 0.969949, 0.970733]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "FINALS CONTENDER (not a promotion). HIGHEST CV in the bank = 0.970410 (> champion n091 0.970355); LB 0.97118 ≈ champion 0.97121 (within noise, -0.00003). Doesn't clear the 2·sem/bootstrap promote bar (P=0.849) so champion is unchanged — BUT kept as a top end-of-comp finals candidate on the trust-CV-over-LB principle. Evaluate in probes/finals_robustness.py alongside n091/n116."
leak: clean
lb: 0.97118
submitted: 2026-06-18T06:46Z
created: 2026-06-18
decided: 2026-06-18T06:50Z
tags: [stack, ensemble-of-ensembles, combine, meta-over-stacks, finals-contender]
---

## plan
built on:   the 6 strongest stack nodes (n091 champion + n070/n116/n063/n041/n040), each itself a
            balanced-LogReg meta over the 63-base bank.
change:     ensemble-of-ensembles — a fresh balanced multinomial LogReg @C=0.003 over the SIX stacks'
            clipped log-probs (the champion's combiner form), nested per-fold on frozen folds for the
            honest OOF, refit on all train for the test prediction.
hypothesis: (user-directed) test whether stacking our own stacks adds anything on the LB despite the
            CV wash — the meta-over-stacks lifts working-fold CV +0.000056 (probes/ensemble_of_ensembles.py).
target:     Balanced Accuracy maximize. NOTE: does NOT clear the structural promote gate (bootstrap
            P=0.849 < 0.90); this is an LB probe, not a promotion. Prior: WASH — the stacks share the
            same bank + folds, so a meta over them re-derives n091's information (LOO saturation).

## notes
Human-directed LB probe (user: "try submitting this"). The probe (probes/ensemble_of_ensembles.py)
showed simple-average -0.000010 / LogReg-meta +0.000056 vs champion, P=0.849. Submitting the LogReg-meta
variant. Reproduce: `uv run --no-sync python nodes/node_0129/src/solution.py`.
