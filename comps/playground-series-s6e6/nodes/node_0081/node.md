---
id: node_0081
desc: SAINT row-attention base on richFE
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: dead
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.964672]
baseline_cv: 0.970227
gates: {schema_ok: false, oof_full: false, no_nan: false, dist_sane: false, leak_clean: true, cv_too_good: false, passed: false}
gate_note: "Cheap kill: fold-0 BA=0.964672 < 0.9675 kill threshold. No artifacts produced."
leak: clean
lb: null
submitted: null
created: 2026-06-13T11:56Z
decided: 2026-06-13T14:34Z
---

## plan
built on:   root (new family draft); copies node_0033 (TabM on fs_realmlp_fe) for data loading / folds loop / OOF convention — nothing kept byte-identical, this is a new model family.
change:     Train a SAINT-style transformer (feature self-attention + INTERSAMPLE/row attention over the batch) on fs_realmlp_fe, save 5-fold OOF + test_probs. The row-attention inductive bias is the one transformer variant the bank lacks (FT-T is feature-attention only).
hypothesis: Intersample (row) attention captures a decision pattern absent from the feature-attention FT-T and the bank GBDTs, so its OOF adds marginal fwd-select value on bank17+FT-T like FT-T did.
target:     balanced accuracy maximize; solo ~0.967+; counts if its OOF is fwd-selected onto bank17+FT-T (cv > 0.970227 by >2·sem).

This is the repeatable lever: FT-Transformer (node_0070, the lone external base that
helped) proved a genuinely-new attention inductive bias can add marginal signal to the
saturated bank. SAINT adds INTERSAMPLE attention (attention across rows in a batch), a
structurally different bias from FT-T's column-attention and from TabM/MLP.

Copy node_0033 (TabM on fs_realmlp_fe) for the data loading, fs_realmlp_fe recipe
(data.md L72; FE in refs/realmlp-v5-for-s6e6.py), the 5-fold folds.json loop, balanced
handling, and OOF/test_probs saving convention (n_train×3 / n_test×3 aligned to folds).
Read node_0070/src/solution.py for the FT-T rtdl recipe (embeddings, schedule,
early-stop, n_ens) to mirror as a competent baseline. Use rtdl/rtdl_revisiting
(installed) or a thin SAINT loop over an rtdl FT-T block plus an intersample-attention
block; standard recipe (num embeddings, AdamW, cosine schedule, early stop) — this is the
competent baseline, not the experiment.

CHEAP KILL (GPU run): after fold-0 finishes, if standalone fold-0 BA < 0.9675 (below
TabM/FT-T tier) ⇒ STOP the run, do not train remaining folds, mark dead. Solo target is
to reach ~0.967-0.968 (FT-T/TabM tier) so its OOF is a fwd-select candidate.

Read journal 2026-06-12T15:01Z (n70) for the fwd-select-onto-true-bank-17 protocol and
the HARD baseline-reproduce assert lesson (n70 v1 misbuild).

well: outside
