---
id: node_0096
desc: Nystroem RBF random-feature linear base
op: draft
parents: [root]
uses_data: [fs_rbf_nystroem]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.970153
gates: {schema_ok: true, oof_full: false, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "FOLD-0 KILL: solo_BA=0.9418 < 0.96 threshold (gate killed; err_corr=0.530 which DID pass the decorrelation criterion but BA criterion failed). Full OOF not produced — oof.npy contains fold-0 rows only (rest zeros). cv=null (no 5-fold run). best_gamma=0.07692308 (1/n_features). Nystroem RBF n_components=1000, LogReg balanced multiclass. Runtime=~1.4min total. This kernel-approximation class is too weak solo at this n_component budget — the BA floor is ~0.94 on 13-dim input, well below the 0.96 kill threshold. Decorrelation is plausible (err_corr=0.53) but solo BA is the disqualifying factor."
leak: clean
lb: null
submitted: null
created: 2026-06-14T10:41Z
decided: 2026-06-14
---

## plan
built on:   root (a genuinely new model CLASS — kernel-approximation random features into a linear/shallow head — never tried in this comp)
change:     Map the ~13-dim standardized core photometric vector through a sklearn **Nystroem RBF kernel-approximation** transform (`fs_rbf_nystroem`, fit_in_fold) into a high-dim random-feature space (n_components ~1000-2000), then fit a balanced multinomial LogReg (or a shallow MLP head) on those features. This is an RBF-kernel-SVM-equivalent decision geometry made tractable at this row count — a model class/geometry the search has NOT visited (every prior base is a tree, a tabular MLP/transformer, or a linear model on raw features).
hypothesis: An RBF random-feature geometry carves the class boundaries differently from trees and tabular NNs (radial similarity to landmarks vs axis-aligned splits / learned dense features), so its OOF errors may be de-correlated enough from the bank-17 + FT-T stack to add signal — the LAST untested decorrelation axis named in the graph.md header is "different INPUT REPRESENTATION / model geometry," and a kernel map is exactly that.
target:     Balanced Accuracy Score, maximize · beats parent if CV > 0.970153 (champion baseline). REALISTICALLY a decorrelation probe judged by the fold-0 err-corr gate, not expected to top the champion solo.

### LOW-EV — honest flag
HONESTLY low expected-value as a CV-beater: a linear/shallow head on RBF features will almost certainly score BELOW the tuned tree/NN bases solo. Its ONLY path to value is genuine decorrelation — a fundamentally different decision geometry producing errors the stack hasn't seen. That is plausible (kernel methods are untried here) but unproven, and every recent decorrelation attempt (residuals, OvR, error-pocket, instance-weighting) has hit the ~0.70 err-corr wall. If it doesn't clear the gate it dies cheaply at fold-0. This is a wildcard, played for its novelty of model class, not its CV.

### The cheap fold-0 err-corr KILL gate (run FIRST, before any full 5-fold run)
COPY the gate harness from `nodes/node_0094/src/solution.py` (it already implements exactly this pattern against node_0070). Procedure:
1. Build `fs_rbf_nystroem` on fold-0's train split only (landmarks + StandardScaler fit on train-fold, transform applied to fold-0 val).
2. Fit the balanced head (LogReg first; shallow MLP only if LogReg looks promising) on fold-0 train, predict fold-0 val.
3. Compute solo Balanced Accuracy on fold-0 val AND the error-correlation vs node_0070's fold-0 OOF slice (`nodes/node_0070/oof.npy`, fold-0 val indices).
4. **KILL the node (status dead, one journal line) if err-corr ≥ 0.65 OR solo BA < 0.96.** Only if BOTH pass, proceed to the full 5-fold OOF build + restack-candidacy.

### HOW to build fs_rbf_nystroem (fit_in_fold)
- Input vector (~13 dims): the standardized core photometric columns u, g, r, i, z, redshift PLUS the 7 fs_realmlp_fe color pairs (u-g, g-r, r-i, i-z, u-r, g-i, r-z). Standardize with a `StandardScaler` fit on the train fold only.
- Apply `sklearn.kernel_approximation.Nystroem(kernel="rbf", n_components=~1000-2000, gamma=<fold-0 micro-sweep>)`. The Nystroem **landmarks** (the subset of train rows it samples) and the scaler are both fit on the TRAIN FOLD ONLY; the fitted transform is then applied to val+test. This is the fit_in_fold reference.
- gamma: do a SMALL fold-0 micro-sweep (a few values around 1/n_features and 1/(n_features·var)) at gate time; pick the best by fold-0 val BA, then freeze it for the full run. Keep the sweep cheap.
- Head: balanced multinomial LogReg (`class_weight="balanced"`) as the primary; a shallow 1-hidden-layer MLP is an allowed alternative ONLY if LogReg clears the gate but looks like it's leaving signal on the table. Pick ONE and state which in the RESULT.
- n_components is a compute knob — start ~1000; the mandatory single-unit timing probe (CLAUDE.md) governs whether ~2000 is affordable across 5 folds. Do NOT let the random-feature matrix blow up memory; transform in batches if needed.
- Implement the recipe inside node_0096's own `src/`.

### References to READ
- `nodes/node_0094/src/solution.py` — copy the fold-0 err-corr gate harness (already wired to node_0070).
- data.md line 88 (fs_realmlp_fe) — the source of the 7 color-pair definitions for the core vector.
- sklearn docs: `kernel_approximation.Nystroem` (RBF map) + `linear_model.LogisticRegression(class_weight="balanced", multi_class="multinomial")`.
- `nodes/node_0070/oof.npy` — the fold-honest OOF the err-corr gate measures against.
- graph.md header — "last untested decorrelation axis = different INPUT REPRESENTATION" (the rationale) and the closed-list of washed decorrelation attempts (why this is LOW-EV).
