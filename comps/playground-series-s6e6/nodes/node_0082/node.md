---
id: node_0082
desc: NODE differentiable-tree base on richFE
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
folds: [0.957533]
baseline_cv: 0.970227
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null, leak_clean: null, cv_too_good: null, passed: null}
gate_note: fold-0 BA 0.957533 < 0.9665 kill threshold; NODE below base tier; self-killed dead
leak: clean
lb: null
submitted: null
created: 2026-06-13T11:56Z
decided: 2026-06-13T14:34Z
---

## plan
built on:   root (new family draft); copies node_0033 (TabM on fs_realmlp_fe) for fs_realmlp_fe loader, folds.json loop, and OOF/test_probs convention — new model family, nothing kept byte-identical.
change:     Train NODE (Neural Oblivious Decision Ensembles — differentiable oblivious decision trees) on fs_realmlp_fe, save 5-fold OOF + test_probs as a fwd-select candidate. A tree-shaped NN bridges GBDT and MLP decision boundaries — a hybrid bias the bank has neither in pure form.
hypothesis: A differentiable-oblivious-tree decision geometry decorrelates from both the bank's hard-split GBDTs and its smooth NNs, so its OOF adds fwd-select value where redundant GBDT/MLP adds did not.
target:     balanced accuracy maximize; solo ~0.966+; counts if fwd-selected onto bank17+FT-T (cv > 0.970227 by >2·sem).

NODE learns ensembles of soft oblivious trees by gradient descent — its decision geometry
is between the bank's GBDTs (hard axis-aligned splits) and its MLP/TabM/FT-T (smooth).
That intermediate bias is genuinely untried and is the kind of decorrelation the saturated
bank still rewards (per the FT-T n70 result that decorrelated bases DO help even when
sub-noise).

Copy node_0033 for the fs_realmlp_fe loader, folds.json loop, and OOF/test_probs
convention. Prefer a library NODE impl (pytorch-tabular's NODE, or the original node
package via uv add) per hard-rule 8 — try the library first, hand-roll only if it fails
the torch build, and say so. Standard recipe: entmax/sparsemax bins, ~2 layers × ~1024
trees depth-6, AdamW, early stop on fold val BA.

CHEAP KILL (GPU run): after fold-0 finishes, if standalone fold-0 BA < 0.9665 ⇒ STOP, do
not train remaining folds, mark dead.

Read node_0070/src/solution.py (FT-T) for the embeddings/schedule template and the n70
journal line (2026-06-12T15:01Z) for the fwd-select-on-true-bank-17 protocol +
baseline-reproduce assert.

well: outside
