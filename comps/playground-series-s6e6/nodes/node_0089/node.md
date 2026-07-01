---
id: node_0089
desc: STAR-recall BOOST knob sweep on meta
op: improve
parents: [node_0076]
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
gate_note: "best_b=1.0 (no improvement; sweep b in {1.25,1.5,2.0} all lower). cv=parent=0.970227; does not beat by >2*sem=0.000488. No LB check needed — no improvement."
leak: clean
lb: null
submitted: null
created: 2026-06-13T15:51Z
decided: 2026-06-13T17:10Z
---

## plan
built on:   node_0076 (FT-T base + bagged-argmax stack, cv 0.970227). Reuses `nodes/node_0076/src`; no base change.
change:     On the n76 bagged-argmax LogReg meta, sweep the per-class sample-weight BOOST knob (Deotte recipe: multiply STAR-class weight in the balanced LogReg by b ∈ {1.0, 1.25, 1.5, 2.0}) fit fold-honest on OOF, pick the b that maximizes OOF balanced accuracy. ONE atomic meta knob, no base change.
hypothesis: Up-weighting STAR inside the balanced LogReg fit (not post-hoc) lifts macro-recall where the post-hoc prior calibration washed.
target:     Balanced Accuracy maximize; beats n76 0.970227 if best-b CV > by >2·sem.

Balanced accuracy is recall-macro and STAR (14%) is the recall-limiting class (research.md anchor: hard STAR↔GALAXY at low z). The Deotte stacker exposes a BOOST knob (research.md lines 56-58, `refs/cdeotte_lr_stacker`) we have NOT swept honestly — n78's post-bag prior calibration washed, but that was a post-hoc multiplier, NOT the in-LogReg class weight which calibrates jointly with the fit.

READ: `nodes/node_0076/src` (5-seed bagged-argmax balanced LogReg), `refs/cdeotte_lr_stacker` for the BOOST implementation, `nodes/node_0078/node.md` (what the washed post-hoc calibration did differently).

Fit b on OOF inside the honest CV, report CV/sem per b; this is the only remaining honest meta lever besides the C sweep.

CPU, minutes; `uv run --no-sync`. Frozen folds.json.

well: exploit

## notes
BOOST sweep result (per-b cv/sem, all 5-seed bagged-argmax):
  b=1.00  cv=0.970227  sem=0.000244  folds=[0.971110,0.970035,0.969895,0.969730,0.970364]
  b=1.25  cv=0.970009  sem=0.000252  folds=[0.970903,0.969547,0.969668,0.969697,0.970231]
  b=1.50  cv=0.969761  sem=0.000289  folds=[0.970814,0.969282,0.969262,0.969528,0.969916]
  b=2.00  cv=0.968935  sem=0.000287  folds=[0.969978,0.968508,0.968394,0.969095,0.968702]

Monotonically decreasing with STAR-weight boost. The balanced LogReg already handles STAR
optimally without extra upweighting. Outputs written for b=1.0 (= parent n76). Runtime: ~34m CPU.
