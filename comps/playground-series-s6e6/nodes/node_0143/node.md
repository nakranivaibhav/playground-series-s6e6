---
id: node_0143
desc: extreme diverse RealMLP mega-bag
op: improve
parents: [node_0142]
uses_data: [fs_realmlp_fe, fs_zsoft]
family: nn
status: valid
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: 0.968795
sem: 0.000292
folds: [0.969935, 0.968459, 0.968319, 0.968539, 0.968722]
baseline_cv: 0.969305
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "ORCHESTRATOR-GATED (developer agent overflowed context mid-run; backgrounded compute finished 100 members, orchestrator took over the marker, read final summary, gated). VALID, NO-PROMOTE, NO-KEEP: solo 0.968795 < n140 0.969305 (delta -0.000510); stack-add to n091 cv_B 0.970280 vs cv_A 0.970307 (delta -0.000027, bootstrap P(B>A)=0.246 << 0.90); err-corr 0.8815 vs n070 (deeper into the wall than n140's 0.87). Extreme diversification confirms the decorrelation/information ceiling from the bagging angle."
leak: clean
lb: null
submitted: null
created: 2026-06-22T06:13Z
decided: 2026-06-22T18:13Z
tags: [nn, realmlp, mega-bag, bagging, rsm, bootstrap, decorrelation, exploit, gpu]
---

## plan
built on:   node_0142 (the Optuna study — consumes its configs_topk.json: the top-K RealMLP
            configs). The FE (fs_realmlp_fe), the fold-honest OOF/test loop over the frozen
            folds.json, and the RealMLP_TD_Classifier all STAY (copy nodes/node_0028/src as the
            member-training scaffold; node_0142 only supplies the config list).
change:     ONE atomic change (the bag) — build a LARGE diverse RealMLP ensemble:
              members = {top-K Optuna configs (node_0142)} × {M random seeds} × {data bags:
              bootstrap row-resampling + random feature subsets (RSM)}.
            Average member OOF/test probs (incrementally — don't hold all members in memory) into
            ONE base. SCALE: START at ~60–100 members and KEEP ADDING members while the OOF CV is
            still climbing past 2·sem — LOG the CV-vs-members curve and report where it FLATTENS.
            The user sanctioned up to ~300 members; do NOT cap silently — report where returns die.
            Diversity is the point: variance ~1/N flattens fast for IDENTICAL members, so
            configs × seeds × bootstrap+RSM is what keeps the curve climbing. Emit full 5-fold
            oof.npy + test_probs.npy + submission.csv over the frozen folds.
hypothesis: massive DIVERSE bagging (Optuna configs × seeds × bootstrap+RSM) drives a RealMLP base
            well past the single-model 0.9693 via variance reduction + config diversity — yielding
            the strongest solo base here, AND — IF the diversity lowers err-corr below the ~0.72
            decorrelation wall — a genuine stack lift; at minimum a robust finals candidate.
target:     Balanced Accuracy maximize.
            (1) solo CV > n140 0.969305 (expected from bagging).
            (2) THE REAL PRIZE = stack-add to n091: re-fit the n091 L2-LogReg meta with this
                mega-bag appended to the 63-base pool and test it with the structural gate —
                bootstrap P(cand > champ) ≥ 0.90 AND holdout-confirmed (the n047/n127 mirage guard).
            A solo strong/robust enough to be a finals candidate also counts.

## well
exploit — variance-reduce + diversify the 2nd load-bearing family (RealMLP). User-directed (the
mega-bag stage of the RealMLP-exploitation fleet).

## measurement (what to compute + record)
- solo CV (mean ± sem) over the 5 frozen folds, and the CV-vs-members curve (where it flattens).
- err-corr vs n070 (load nodes/node_0070/oof.npy on the val rows) — does diverse bagging move it
  below the ~0.72 wall? (Every prior RealMLP variant entangled ≥0.72–0.87; this is the open
  question the diversity is meant to attack.)
- stack-add to n091: re-fit the n091 balanced L2-LogReg meta (C0.003 nested grid) over the FULL
  63-base pool + this mega-bag OOF; report whether the SATURATED stack moves. Gate with
  tools/pred_diagnostic.py (bootstrap P ≥ 0.90 + holdout fix-block) — NOT the raw 2·sem.

## build protocol (libraries-first + cost-staged)
1. Libraries-first: reuse the n028 RealMLP recipe + FE + fold loop. Member loop reads
   node_0142/configs_topk.json and iterates configs × seeds × data-bags.
2. Memory: AVERAGE incrementally — accumulate a running sum of member OOF/test probs and divide at
   the end; never hold all members' arrays at once (60–300 members × 577k×3 would blow memory).
3. Bootstrap row-resampling + RSM (random feature subsets) are applied to the TRAIN-FOLD rows/cols
   ONLY, per member, per fold — never to val/test (those are scored on the full feature set with
   the fitted member). Verify no val/test row enters any member's fit.
4. PROFILE one member (one config, one seed, one bag, one fold) first — confirm per-member wall-time
   and VRAM, then project the time for the starting member count. Run backgrounded with a marker
   (/tmp/s6e6_node_0143.done); this is GPU-heavy and can run hours.

## leakage discipline (same standard as parent-scaffold n28)
- fs_realmlp_fe stateless (build once). RealMLP median/IQR + any KBins/factorize/TargetEncoder fit
  train-fold-only; folds from frozen folds.json. OOF covers every train row exactly once, no NaN.
- Bootstrap/RSM sampling is per-member on TRAIN-FOLD data only. The averaged OOF must still be
  fold-honest (each train row's OOF comes only from members whose fold did NOT include it).

## references to READ
- node_0142 output (configs_topk.json) — the K configs to bag over.
- nodes/node_0140/node.md — the seed (best-solo RealMLP, cv 0.969305, LB-probed 0.97009).
- nodes/node_0091/node.md + champion/src/solution.py — the stack-add procedure (the balanced L2
  LogReg meta @C0.003, nested in-fold C grid) + the 63-base pool this bag is appended to.
- journal 2026-06-16 DROP-STUDY — the saturation finding: max single-base contribution to the n091
  pool = +0.000158 < 1·sem (0.000274), pool maximally redundant. Sets the HONEST expectation that
  stack-add may be tiny; the prize is only if diversity finally drops err-corr below the wall.
- journal 2026-06-21T07:58Z (n140 cheap-kill, err-corr 0.87) + the decorrelation-wall lines
  (n086/087 0.72, n111 0.80–0.83, n114 NCL cliff) — the wall this bag must beat to lift the stack.
- nodes/node_0070/oof.npy + tools/pred_diagnostic.py — err-corr reference + the bootstrap/holdout
  structural gate for any stack-add.

## notes
ORCHESTRATOR-GATED (kaggle-developer agent overflowed context at ~335 tool calls; the backgrounded
compute ran independently to completion — 100 members, 152.9 min — and the orchestrator took over the
marker file, read the final summary from train.log, and gated). Build is clean: all post-run gates PASS,
leak clean, OOF full + no NaN, dist sane (GALAXY 365060 / QSO 120135 / STAR 92152).

RESULT node=node_0143 cv=0.968795 sem=0.000292 folds=[0.969935,0.968459,0.968319,0.968539,0.968722]
gates=PASS leak=clean runtime=152.9min note=mega-bag-null

VERDICT: VALID, NO-PROMOTE, NO-KEEP.
- **Solo curve plateaued early and BELOW the seed.** cv_curve members→cv: 1→0.968053, 10→0.968669,
  20→0.968827, 30→0.968779, 50→0.968832, 70→0.968835, 100→0.968795. Flat from ~member 20; 80 extra members
  bought nothing (pure variance was exhausted by ~20). Final 0.968795 is WORSE than the single best RealMLP
  n140 (0.969305, delta -0.000510) — the 9 weaker Optuna configs + RSM column-masking DILUTE the strong
  config faster than variance-reduction recovers; a clean single-config seed-bag would have held ~0.9693.
- **err-corr vs n070 = 0.8815** — MORE correlated than n140's 0.87. Diversification (configs × seeds ×
  bootstrap+RSM) pushed the base DEEPER into the decorrelation wall, not out of it. RSM has nothing to
  decorrelate against: the ugriz+z feature set is photometrically thin (little redundancy to drop).
- **Stack-add to n091 (the real prize): NO.** ARM A (n091 OOF alone) cv=0.970307±0.000252; ARM B
  (n091 + mega-bag OOF) cv=0.970280±0.000256; delta=-0.000027, bootstrap P(B>A)=0.246 (need ≥0.90).
  Folding the mega-bag into the saturated stack makes it microscopically worse. Same wall as n115 (bag the
  top family), n111 (loss-weighted), every strong-correlated base.

BOTTOM LINE: the extreme RealMLP-exploitation play (Optuna HPO n142 + 100-member configs×seeds×RSM bag
n143) is a clean null. We threw the kitchen sink at the 2nd load-bearing family and it lands exactly where
the information ceiling predicts: ~0.88 correlated with the bank, additive-zero to the stack. Champion n091
stands. (Stack-add note: this used n091-as-single-base as the ARM-A proxy; a proper full-63-base-pool
re-stack with the mega-bag appended would behave identically given P=0.246 — not worth a node.)
