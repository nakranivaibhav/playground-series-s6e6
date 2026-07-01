---
id: node_0047
desc: GALAXY-vs-STAR specialist (low-z)
op: improve
parents: [node_0041]
uses_data: [fs_realmlp_fe]
family: gbdt
status: dead
stage: submitted
metric: Balanced Accuracy Score
direction: maximize
cv: 0.970881
sem: 0.000312
folds: [0.971947, 0.970630, 0.970579, 0.970111, 0.971139]
baseline_cv: 0.969808
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: 0.96242
submitted: 2026-06-09
created: 2026-06-08T16:11Z
decided: 2026-06-09
tags: [specialist, galaxy-star, low-redshift, boundary]
---

## plan
built on:   champion stack node_0041 — this node produces a NEW base column for the stack.
change:     A surgical binary specialist for the dominant error. Train a LightGBM (rich FE
            fs_realmlp_fe) on a BINARY GALAXY-vs-STAR task, restricted to the low-redshift
            confusion zone (rs < ~0.15, where GALAXY→STAR is 47% of all stack errors), fit
            IN-FOLD. Emit its OOF + test probability of STAR-vs-GALAXY as new base columns
            (set to the class prior outside the low-z zone / for QSO rows so it's a no-op
            elsewhere). Add as a base in the re-stack.
hypothesis: GALAXY recall (0.961) is the balanced-accuracy bottleneck and GALAXY↔STAR at
            low redshift is the mechanism; a specialist focused on exactly that boundary,
            with no QSO/high-z noise, gives the meta a sharper signal there than any
            general base → lifts the stack via the lagging GALAXY recall.
target:     re-stack with the specialist base added beats champion 0.969808 by >2·sem.

## notes
Leakage: binary target + any encoder fit IN-FOLD only; the low-z mask is on the leak-safe
redshift feature (no labels). OOF on OUR folds. Self-gate the leakage suite.

## result — CV MIRAGE, REVERTED (submitted, LB crashed)
Re-stack CORE15 + specialist: nested-CV **0.970881 ± 0.000312 = +0.001073 (~3.4 sem)** over
champion 0.969808, static leakage scan CLEAN. Looked like a breakthrough → promoted + SUBMITTED
(full_auto, 1 slot). **Public LB = 0.96242 — a −0.0080 COLLAPSE vs node_0041's 0.97043.** That
gap is ~10× the public-slice noise (~0.00087) → NOT noise; the CV gain is an artifact. REVERTED:
node_0041 reinstated as champion; node_0047 marked dead.

WHY (hypothesis): the specialist is a SECOND model fit on the SAME train labels, then its OOF
column is fed to a meta also fit on those labels. Even with per-fold OOF, the specialist's
in-zone OOF predictions are systematically more confident/optimistic on train than its full-train
model is on test (the binary low-z task overfits its narrow zone), so the meta-LogReg overweights
the specialist column → train OOF looks great, test generalizes far worse. Standard single-level
stacking already double-uses labels safely; a NARROW hand-targeted specialist sub-model amplifies
that into a true CV-OOF overfit the nested CV can't see (the optimism is in BOTH the OOF and the
nested holdout — they share the specialist's train-fit bias). LESSON: a stack base that is itself
a label-fit sub-model targeted at a known error pocket is a CV-mirage hazard; gate it on the LB,
not CV alone. Do NOT re-add without an LB confirmation on a tiny probe submission.
