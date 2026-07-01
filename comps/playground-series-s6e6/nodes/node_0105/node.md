---
id: node_0105
desc: Revival re-stack distill bases under shrinkage
op: combine
parents: [node_0091, node_0067, node_0079]
uses_data: []
family: ensemble
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970355
sem: 0.000249
folds: [0.971208, 0.970067, 0.969934, 0.969938, 0.970626]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "NO STUDENT KEPT (wash). Forward-selected the never-pooled distill/pseudo students (n067 solo 0.969414, n079 0.967406, n074 0.968528) onto the n091 FULL pool under C=0.003 shrinkage. Baseline reproduced EXACTLY (0.970355). NONE cleared the +1·sem keep-bar (best add ≤ 1·sem) → final pool == n091, lift 0.000000. The correlation-exclusion of distill/pseudo students HOLDS even under shrinkage (their signal is a re-derivation of the bank they learned from; shrinkage that revives weak-but-INDEPENDENT bases does not revive correlated ones). n091's exclusion was correct. Revival axis closed."
leak: clean
lb: null
submitted: null
created: 2026-06-15T07:06Z
decided: 2026-06-15
---

## plan
built on:   champion n091 pool under its L2-LogReg @C=0.003. Copy node_0099/src/solution.py as the working
            full-pool loader (DO NOT re-derive; DO NOT read n091 solution.py — overflowed prior devs).
change:     Forward-select-ADD the saved OOF of distill/pseudo-label STUDENTS explicitly EXCLUDED from
            n091's pool: n067 (transductive distill, solo 0.969414), n079 (honest disjoint-teacher pseudo
            TabM, 0.967406), optionally n074 (CLOUT pseudo-test, 0.968528). No retraining.
hypothesis: n091's C=0.003 shrinkage REVERSED the weak-decorrelated-base dilution verdict; students
            excluded for correlation under the OLD fixed-C regime may now net positive under shrinkage.
target:     BA maximize · OOF CV > 0.970355 by >1·sem to keep a student; >2·sem to promote.

HOW: load n091 pool (via n099 loader) + nodes/node_0067/oof.npy + nodes/node_0079/oof.npy (+ n074 oof if
present). Reproduce n091 baseline 0.970355 first. Forward-select each student's log-probs into the C=0.003
LogReg arm; keep a student only if its add lifts OOF CV beyond fold-noise. These are COMPLETE 3-class
classifiers (honest) — trust their CV; the n047 mirage rule does NOT apply. Kill: no student exceeds 1·sem
→ correlation-exclusion holds even under shrinkage, close it. Cheap (saved OOF only).
Outputs: oof.npy, test_probs.npy, submission.csv, train.log w/ per-student forward-select deltas.

## notes
well=exploit/revival. n055 DROPPED — verified already in n091's pool (would duplicate n088). Genuinely-
never-pooled strong discards are the distill/pseudo students only.
