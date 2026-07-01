# round_plan — playground-series-s6e6 · ROUND 2026-06-16 PM (n116–n120)

champion at round open: **node_0091** (CV 0.970355 · LB 0.97121 · Balanced Accuracy maximize)
promote bar: CV > 0.970853 (n091 + 2·sem 0.000498) AND leak-clean AND LB-consistent
mode: full_auto · budget 0/10 today (resets 00:00 UTC) · deadline 2026-06-30 (13 left)
prior rounds (n081–n115, all wash/kill, decorrelation↔strength wall confirmed) → see journal.md

Origin: propose-loop refined 5 (critic happy). 1 evidence-driven exploit + 1 finals hedge +
3 fresh-axis explorers (each with a concrete cheap-kill). Orchestrator builds ALL five.

## items
| node | op | desc | well | parent | judged on | verdict |
|------|----|------|------|--------|-----------|---------|
| node_0116 | combine | LOO-pruned restack (drop xgb-6/tabm-0/node_0042) | exploit | n091 | CV > 0.970853 → promote | NULL — best prune 0.970369 (<bar); drop-deltas don't sum under L2 refit, combine sub-axis closed |
| node_0117 | combine | max-decorrelated rank-vote finals hedge | outside | n091/n039/n070 | CV ~n091 + max row-disagreement (slot-2 hedge, no promote) | DONE — CV 0.969272, 0.76% disagreement vs n091; held as deferred finals slot-2 hedge |
| node_0118 | draft | generator decimal-fingerprint GBDT (fs_genfp) | data | root | fold-0 BA ≥0.962 → 5-fold + stack-add | NULL — solo 0.966266 (tier, leak-clean) but err-corr 0.81 + stack-add +1.5e-5 wash; decorrelation wall holds 6th time |
| node_0119 | draft | generative pretrain→finetune TabM (fs_synthpre) | wildcard | root | gen<20min + fold-0 BA ≥0.965 → 5-fold + stack-add | DEAD — kill gate 2: pretrain fold-0 0.967934 < cold 0.968562 (copula pretrain hurts); augmentation lever closed |
| node_0120 | improve | CatBoost ordered-TE + monotone-z recipe | data | n039 | solo ≥0.965 + stack-add to n091 > 2·sem | NULL — solo 0.966366 (<n039), err-corr 0.82, stack-add 0.968519 misses bar; recipe diversity adds no family value (closes the lever with n115's bag side) |

## schedule (30 GB RAM box — heavy nodes serialized)
- Wave 1 (parallel, light): n116, n117, n118
- Wave 2 (serial, each alone): n120 (CatBoost, RAM-gated 30 GB) → n119 (TabM generative, GPU + uv add sdv)

## verdicts
**Wave 1 complete (2026-06-16T13:54Z) — 3 nodes valid, NONE promote, champion n091 stands.**
- n116 NULL: LOO drop-deltas don't sum under L2 refit — pruning recovers ~nothing (best 0.970369 vs bar 0.970853). The last untried combine sub-axis is closed.
- n117 DONE: 0.969272 honest CV, 0.76% row-disagreement vs n091 — kept as a deferred finals slot-2 variance hedge (different decision rule: rank-vote, not LogReg).
- n118 NULL: generator-fingerprint axis is real (solo 0.966266, leak-clean) but correlated (err-corr 0.81) — decorrelation↔strength wall holds a 6th time.

**Wave 2 complete (2026-06-17) — both NULL/DEAD, champion n091 stands.**
- n120 NULL: CatBoost ordered-TE+monotone-z recipe — solo 0.966366 (<n039), err-corr 0.82, stack-add 0.968519 misses bar. Recipe diversity adds no family value; with n115 (bag) the "extend load-bearing CatBoost family" lever is closed both ways. (Stripped the deleted-leakage_scan.py infinite-loop bug.)
- n119 DEAD: synth-pretrain TabM — copula pretrain HURTS vs cold-start (0.967934 < 0.968562). Closes the last augmentation/synthetic-data lever (joins n065 mixup / n062 swap-DAE / jitter-aug).

**ROUND 0116-0120 CLOSED: 5 nodes, 0 promote.** No untried internal lever remains → next round must open with a genuine look-outside (fresh top notebook + late-discussion scan for any signal source not in the bank). Champion n091 (cv 0.970355 · lb 0.97121).
