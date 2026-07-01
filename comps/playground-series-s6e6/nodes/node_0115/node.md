---
id: node_0115
desc: True-bag CatBoost (LOO-crowned top family)
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: gbdt
status: buggy
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: []
baseline_cv: 0.970316
gates: {schema_ok: null, oof_full: false, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "CHEAP-KILL at fold-0. True-bag CatBoost (5 bags, bootstrap + rsm0.7 + Bernoulli subsample0.7, leak-correct INNER early-stop split; val_idx never in any fit). Fold-0 bagged BA 0.967195 (strong; per-bag 0.9656-0.9666) BUT err-corr vs bank n070 = 0.805 overall / 0.793 per-class = ABOVE the 0.72 wall — MORE correlated than a typical strong base, not less. Bagging cuts VARIANCE not error-STRUCTURE, so it strips the idiosyncratic noise that was the only sliver of decorrelation and agrees MORE with the consensus. Done right (right family CatBoost, right method true-bag not seed) and it STILL cannot decorrelate -> no full 5-fold, no restack; 'bag/add more bases with the top models' definitively CLOSED. (En route fixed an infinite-loop bug copied from n039: a parent-walk for tools/leakage_scan.py, a tool that was deleted — spun at 100% one-core with a 0-byte log; had also silently killed the dev-agent run.)"
leak: clean
lb: null
submitted: null
created: 2026-06-16T08:59Z
decided: 2026-06-16
tags: [gbdt, catboost, true-bagging, decorrelation, data-directed, drop-study, cheap-kill]
---

## plan
built on:   root — a NEW base, seeded by the drop_study (probes/drop_study_ranking.csv, journal
            2026-06-16T08:57Z). That LOO study crowned the CatBoost family the single load-bearing
            family (cat-3 +0.000158, 4x any other base) while TabM sat near the bottom (tabm-0 rank 62,
            harmful). The prior seed-bag node_0075 bagged the WRONG family (TabM). Template src copied:
            nodes/node_0039/src/solution.py (CatBoost depth=6/border=128 on fs_realmlp_fe).
change:     Build a 5-bag TRUE-BAGGED CatBoost (not a seed-bag) on fs_realmlp_fe. Per bag b:
            (1) bootstrap-resample the train-fold rows WITH REPLACEMENT (bag seed); (2) CatBoost with
            rsm=0.7 (random column subspace) + bootstrap_type='Bernoulli' subsample=0.7 (per-tree row
            subsample) + random_seed varied per bag. Average the 5 bags -> the bagged base. Early-stop
            on an INNER split carved from train-fold rows only (val_idx never in any fit — fixes n039's
            early-stop-on-OOF-fold leak).
hypothesis: a true-bagged CatBoost is more DECORRELATED from cat-3/node_0039 than another seed, so it
            may add signal the redundant single CatBoosts cannot. Honest EV LOW (drop study shows our
            CatBoosts already redundant w/ cat-3; NCL cliff says no strong base decorrelates <~0.72).
target:     decisive check = err-correlation of the bagged base vs bank n070: if <0.65 AND BA>=0.966 ->
            full 5-fold + restack onto n091 (promote only if CV > 0.970355 by >2sem); else record & close.

## RESULT (2026-06-16)
CHEAP-KILL at fold-0. Fold-0 bagged BA **0.967195** (strong, tier-worthy; per-bag 0.965598-0.966614,
the bag-average lifts solo a touch). Decorrelation verdict vs bank node_0070 on fold-0 val:
**err-corr = 0.8049 overall, 0.7925 per-class** (GALAXY 0.813, QSO 0.826, STAR 0.739). That is ABOVE
the ~0.72 wall, NOT below 0.65 — the bagged CatBoost is MORE correlated with the consensus than a
single strong base. Mechanism: bagging reduces VARIANCE, which removes the idiosyncratic noise that
was a model's only source of decorrelation, so it agrees MORE with the bank. Per the gate (err-corr
not <0.65), STOPPED after fold-0: no full 5-fold, no restack. 1-bag timing was 213s -> projected full
run 88.9 min (saved by the cheap-kill).

This is the data-directed bag done correctly — right family (CatBoost, the LOO load-bearer, vs TabM the
old seed-bag n075 picked) and right method (true bagging with bootstrap+rsm+subsample, vs seed-only) —
and it STILL cannot beat the wall. Combined with the drop study (pool saturated, max single-base
contribution <1sem) and the NCL cliff (no strong-decorrelated base exists in model space), the
"add more bases / bag the top models" avenue is closed from every angle. Champion node_0091 stands.

## notes
well = data-centric (favored). Orchestrator-registered, human-directed node ("add more bases with the
top models"). solution.py is correct and leak-safe (kept the dev-agent's implementation; orchestrator
fixed the vestigial repo-root infinite-loop). probe.log holds the fold-0 run.
