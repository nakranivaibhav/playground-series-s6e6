---
id: node_0088
desc: revival re-stack of n55 discard
op: combine
parents: [node_0076, node_0055]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970227
sem: 0.000244
folds: [0.971110, 0.970035, 0.969895, 0.969730, 0.970364]
baseline_cv: 0.970227
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "n55 (DCN) delta=-0.000000, not selected; node outputs baseline stack (n76 reproduced exactly). cv=0.970227 does not beat champion 0.970153 by >2*sem (threshold=0.970641). No promote."
leak: clean
lb: null
submitted: null
created: 2026-06-13T15:51Z
decided: 2026-06-13T17:10Z
---

## plan
built on:   node_0076 (FT-T base + bagged-argmax stack, cv 0.970227) + node_0055 (DCN, saved OOF). Reuses `nodes/node_0076/src`.
change:     Forward-select the SAVED OOF of n55 (DCN, best non-n70 re-stack PROBE LB 0.97083) as a candidate addition onto the n76 stack. Promote only if it lifts honest CV >2·sem. No retraining. (n67/n74 deliberately EXCLUDED — node_0084 already re-tested both onto n76 at +0.000014 no-promote on 2026-06-13.)
hypothesis: n55 (DCN) is a de-correlated honest classifier whose neutrality was judged against a pre-FT-T stack; on the current n76 stack it may now clear fold-noise.
target:     Balanced Accuracy maximize; beats n76 0.970227 if n55 lifts CV by >2·sem.

Revival re-stack habit (cheap, no retrain), trimmed per reviewer to the ONLY never-folded-in candidate. The residual structure shifted when n70/n76 added FT-T and dropped DE-threshold, and n55's re-stack PROBE hit LB 0.97083 (best non-n70 LB) but was never folded into the FT-T-era stack.

READ: `nodes/node_0076/src` (a1 merge + 5-seed bagged-argmax LogReg meta), `nodes/node_0084/node.md` (prior re-stack mechanics + the n67/n74 neutral verdict we are NOT repeating), `nodes/node_0055/oof.npy` (id-aligned 577347/247435).

Trust the CV — n55 is a COMPLETE classifier (honest), NOT a label-fit specialist (n47-style specialists permanently excluded). Forward-select with eps +0.00003; report n55 marginal delta + final CV/sem.

CPU, minutes; `uv run --no-sync`. Frozen folds.json.

well: exploit
