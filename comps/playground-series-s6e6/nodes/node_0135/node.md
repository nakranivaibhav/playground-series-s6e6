---
id: node_0135
desc: TabPFN-FT full OOF + stack-add test
op: improve
parents: [node_0133]
uses_data: [fs_realmlp_fe]
family: nn
status: proposed
stage: proposed
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: null
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: null, no_nan: null, dist_sane: null,
        leak_clean: null, cv_too_good: null, passed: null}
gate_note: null
leak: null
lb: null
submitted: null
created: 2026-06-21T07:36Z
decided: null
tags: [nn, foundation-model, tabpfn, finetune, decorrelation, stack-add, exploit]
---

## plan
built on:   node_0133 (TabPFN-v2 FINETUNE on fs_realmlp_fe). Its fold-0 read was
            BA 0.958857 — below the 0.960 standalone tier bar, so it was filed
            below-tier and the full 5-fold OOF was never produced. The pipeline
            (nodes/node_0133/src/solution.py) stays BYTE-IDENTICAL: same
            FinetunedTabPFNClassifier, same fs_realmlp_fe FE, same frozen
            folds.json loop, same lr=3e-5 / 30ep / ctxq=10k / infer-sub=50k.
change:     RUN the full 5-fold to materialise nodes/node_0135/oof.npy
            (577347×3) + test_probs.npy (247435×3) + submission.csv, then run the
            ONE thing n133 never could: the stack-add test — does this
            decorrelated-but-weak (0.9589) foundation-transformer base LIFT the
            champion n091 mega-stack when added as a 64th base? No model change;
            this is the OOF-completion + stack-add probe that n133's fold-0 read
            deferred.
hypothesis: a gradient-finetuned transformer is a NEW REPRESENTATION — the only
            historical source of decorrelation here (flux 0.485 / RBF 0.53 / FT-T
            0.53). At 0.9589 it is STRONGER than every prior decorrelated base
            that failed to stack (RBF 0.947/corr0.53 stack-add −0.00005; flux
            0.940/corr0.485), so if its err-corr vs n070 is < ~0.65 the stack-add
            could flip positive where the 0.94-tier decorrelated bases could not —
            the one untested corner of the decorrelation wall.
target:     Balanced Accuracy maximize. Solo CV ≈ 0.9589 expected (NOT promotable
            solo). Real value = stack-add to n091: keep only if the re-fit
            mega-stack CV beats champion 0.970355 by the structural gate
            (tools/pred_diagnostic.py bootstrap P(>champ) ≥ 0.90 AND holdout-fold
            fix-block holds — NOT raw scalar). Otherwise this closes the FM family
            with a measured decorrelation+stack verdict, not just a tier read.

## build protocol (cost-staged)
- The full 5-fold is the expensive step (~3.7hr projected: ~42min/fold fit +
  144s val-predict, 12.3GB VRAM — single-fold timing already measured in n133).
  Run it BACKGROUNDED with a marker file (CLAUDE.md long-training protocol):
  `DONE=/tmp/s6e6_node_0135.done`; tail train.log filtered for `cv=|Traceback|
  Error|Killed|OOM`. No re-smoke needed — n133 already proved the pipeline.
- Once oof.npy exists, the stack-add is SECONDS: append this base's clipped
  log-probs as a column to the n091 FULL pool and re-fit the L2 LogReg meta
  (LogisticRegressionCV nested C grid) over the frozen folds.

## leakage discipline (inherited from n133 — re-verify, do not re-derive)
- Stateless fs_realmlp_fe computed once; factorize/KBins/TargetEncoder fit on the
  train fold only; FinetunedTabPFNClassifier does its OWN internal 10% val split
  FROM the train fold for early-stop and never touches our OOF val_idx; folds from
  frozen folds.json. OOF covers every train row exactly once.
- After the stack-add: this is a combine over pre-computed OOF (no target/id in
  features) — same leak profile as n091; verify the appended column is the
  fold-honest OOF (not a refit-on-full-train artifact).

## references to READ
- nodes/node_0133/src/solution.py — the verbatim TabPFN-FT pipeline to re-run.
- nodes/node_0133/node.md — fold-0 read, timing, VRAM, the "stack value = only if
  corr<0.65 AND strong enough to flip stack-add" prior.
- champion/src/solution.py + a1_full_merge.py / a1_submit.py — the n091 OOF-ingest
  (clip log-probs ±30, normalize) + balanced-LogReg meta to re-fit with the added
  column.
- tools/pred_diagnostic.py — the structural gate (err-corr vs node_0070, per-class
  recall deltas, paired bootstrap P, holdout fix-block) every base now passes.
- journal.md 2026-06-20T08:31Z (n133 fold-0 read) and 2026-06-20T14:27Z
  (FM-family-closed verdict — this node tests the ONE open question that verdict
  left: full-OOF decorrelation + stack-add, not just tier).
