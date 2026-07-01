---
id: node_0110
desc: TabM multi-task with redshift aux head
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: buggy
stage: built
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "STRENGTH KILL: fold-0 BA=0.8856 < 0.962 threshold. Lambda sweep: lam=0.1 -> BA=0.8856 err-corr=0.5361; lam=0.5 -> BA=0.8853 err-corr=0.5347. Best lam=0.1. Aux reconstruction cannot recover redshift signal from photometry — BA cratered from ~0.970 to 0.886, consistent with n095 drop (0.72 without aux). Decorrelation is strong (err-corr=0.54 < 0.70) but the strength kill fires first. No OOF/submission artifacts."
leak: clean
lb: null
submitted: null
created: 2026-06-15T11:09Z
decided: 2026-06-15
---

## plan
built on:   root — WILDCARD (coupled change licensed in this well, rule 4). Copy node_0033's TabM-richFE
            recipe VERBATIM (`nodes/node_0033/src/solution.py`).
change:     Add an auxiliary regression head predicting REDSHIFT (held OUT of the input features), trained
            jointly: total loss = balanced-CE(class) + lambda·Huber(redshift). The model must RECONSTRUCT
            redshift from photometry rather than consume it. (Aux target + its loss = one coupled hypothesis.)
hypothesis: redshift is the dominant class separator and every base CONSUMES it as input → all bases lean
            on it identically, a structural source of the ≥0.70 correlation. Forcing the net to reconstruct
            redshift indirectly through photometric structure yields a representation that still separates
            classes (strong) but along different error directions (decorrelated). Attacks the shared-signal
            correlation directly — which framing/specialist nodes (n106 corr0.796) did not.
target:     BA maximize · cheap-kill fold-0 solo BA < 0.962 OR err-corr vs node_0070 ≥ 0.70 → STOP;
            sweep lambda∈{0.1,0.5} at fold-0 only, freeze best. If clears → full OOF + restack onto n091 >2·sem.

HOW (TIGHT — single base, NO full-pool loader; do NOT read node_0091's solution.py):
- cp nodes/node_0033/src/solution.py → nodes/node_0110/src/solution.py. Keep its fs_realmlp_fe build,
  tabm library training loop, balanced CE, PLR embeddings, frozen-fold loop, OOF/test/submission writing.
- CHANGES: (1) DROP raw redshift (and its direct derivatives log1p(redshift), g/redshift, i/redshift) from
  the INPUT features. (2) Add a second linear head on the TabM trunk outputting a scalar redshift estimate;
  total loss = balanced-CE(class) + lambda·Huber(redshift_true, redshift_pred), redshift standardized for
  the Huber term (target stats from TRAIN-FOLD only — fit_in_fold). (3) At inference use ONLY the class head.
  NOTE n095 cratered to BA 0.72 when redshift was simply DROPPED — the aux reconstruction must actually
  learn the manifold (keep lambda modest); that is why the fold-0 strength kill is tight.
- GATE ORDER: fold-0 only first (both lambdas); solo BA + err-corr vs nodes/node_0070/oof.npy. If best
  lambda's BA<0.962 OR err-corr≥0.70 → STOP, record. Else run all 5 folds at the best lambda.
- Outputs nodes/node_0110/{oof.npy, test_probs.npy, submission.csv, train.log}. Self-gate (kaggle-leakage):
  redshift-standardization stats fit train-fold-only; folds frozen; OOF full/no-NaN/each-row-once; dist
  sane; schema. Write gates + cv/sem/folds + leak + err-corr (gate_note); stage: built. Do NOT submit.
  `uv run`, tabm library (rule 8). GPU — marker DONE=/tmp/playground-series-s6e6_node_0110.done.

## notes
well=wildcard. If it wins, ablate (drop the aux head, keep redshift as input) next round to attribute the
lift to the aux task vs the input change. The redshift target for the Huber loss is the TRUE redshift —
this is a legitimate auxiliary supervised target (not leakage): redshift is a known train feature, and the
class label is never in the inputs.
