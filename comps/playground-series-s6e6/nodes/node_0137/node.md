---
id: node_0137
desc: ModernNCA retrieval base on rich FE
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.959088
sem: 0.000344
folds: [0.960119, 0.958349, 0.959431, 0.958299, 0.959239]
baseline_cv: 0.970355
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: "err-corr vs n070=0.690 (above 0.65 decorr threshold — joins wall not fully decorrelated). GALAXY recall +0.014 vs champion but STAR recall -0.037 (big). Library failed: faiss-gpu-cu12 no GPU API; thin hand-roll used."
leak: clean
lb: null
submitted: null
created: 2026-06-21T07:36Z
decided: null
tags: [nn, modernnca, retrieval, metric-learning, decorrelation, outside, gpu]
---

## plan
built on:   root (a genuinely NEW model FAMILY — a deep retrieval / neighbour-based
            classifier, distinct from every axis-partitioning GBDT and every MLP/
            attention parametric NN in the bank). Copy nodes/node_0033/src as the
            scaffold ONLY for its FE machinery + fold-honest OOF/test loop over the
            frozen folds.json (it already builds fs_realmlp_fe and writes
            oof.npy/test_probs.npy correctly); REPLACE the TabM model with
            ModernNCA.
change:     model = ModernNCA (Ye et al. 2024, "Modern Neighborhood Components
            Analysis" / "A Closer Look at Deep Learning on Tabular Data",
            arXiv:2407.03257) — a learned-embedding kNN: an MLP encoder maps each
            row to a latent space, classification is a SOFT retrieval over a
            sampled reference set in that space (differentiable NCA objective).
            Libraries-first: use the pytabkit / official ModernNCA implementation
            if a compatible build exists (uv add); only hand-roll a thin training
            loop around the published encoder + soft-neighbour loss if no library
            build works (say so explicitly with the reason). fs_realmlp_fe is the
            SAME rich FE every strong NN base used — the change is the model class,
            not the features.
hypothesis: a retrieval/metric-learning decision rule is a structurally DIFFERENT
            error geometry — it classifies by "which training rows am I near in a
            learned space," not by axis splits (GBDT) or a parametric boundary
            (MLP/TabM). The decorrelation wall here is REPRESENTATION-driven (flux
            0.485 / RBF 0.53 / FT-T 0.53 are the only decorrelated bases, all new
            representations); a learned-neighbour embedding is a new representation
            class never tried, so it is a fresh shot at corr < 0.65 — AND, unlike
            RBF/flux which capped at 0.94, ModernNCA is a tier-competitive model on
            tabular benchmarks, so it may land BOTH decorrelated AND strong (the
            corner the wall has never been hit from).
target:     Balanced Accuracy maximize. CHEAP-KILL on fold-0 (like n103/n81/n82):
            if fold-0 BA < 0.960, kill the draft. If it clears tier, run the full
            5-fold → oof.npy, then tools/pred_diagnostic.py for err-corr vs n070
            and the stack-add to n091 (promote only via the structural gate,
            bootstrap P ≥ 0.90 + holdout fix-block).

## build protocol (cost-staged cheap-kill)
1. SMOKE: small subsample, few epochs — verify the ModernNCA pipeline runs
   end-to-end + project per-fold timing + VRAM.
2. FOLD-0 only (background + marker /tmp/s6e6_node_0137.done) → tier read.
   Cheap-kill at BA < 0.960.
3. FULL 5-fold only if fold-0 clears tier → oof.npy (577347×3) + test_probs.npy
   (247435×3) + submission.csv over the frozen folds.

## leakage discipline (same standard as parent-scaffold n33)
- Stateless fs_realmlp_fe computed once; factorize/KBins/TargetEncoder fit on the
  train fold only; folds from frozen folds.json.
- RETRIEVAL-SPECIFIC: the reference/candidate set ModernNCA retrieves over must be
  drawn from the TRAIN FOLD ONLY — never include val or test rows in the
  neighbour pool (that is the exact cross-row leak the fit_in_fold class guards;
  read the fold loop to confirm val/test are encoded-and-scored, never used as
  references). OOF must cover every train row exactly once.

## references to READ
- nodes/node_0033/src/solution.py + features.txt — the FE + fold-honest OOF/test
  scaffold to copy (model swapped out).
- research.md (retrieval/metric-learning lever, if present) + arXiv:2407.03257
  (ModernNCA spec) — the model recipe.
- journal 2026-06-15 flux finding + nodes/node_0096|0103|0108 (the
  decorrelation-wall priors: only NEW representations decorrelate, all capped at
  ~0.94 BA so far — this node tests whether a retrieval representation breaks that
  ceiling).
- tools/pred_diagnostic.py — err-corr vs node_0070 + structural stack gate.
- champion/src/solution.py — the n091 stack-add target for the decorrelation test.

## build notes (post-run)
Library install results:
- pytabkit (RealTabR/TabR) requires faiss-gpu. faiss-gpu-cu12 v1.14.1.post1 installs but
  exposes an empty stub — no GpuIndexFlatConfig API. Fails at runtime with AttributeError
  on CUDA 12.8 / torch 2.11 / RTX 5090. faiss-cpu conflicts when both installed.
- RULE 8 fallback: thin hand-rolled ModernNCA (CE warm-up + NCA metric phase).

Training decisions:
- Two-phase: Phase 1 (CE warm-up, internal early stop) + Phase 2 (NCA metric, cosine sim T=0.1).
- Phase 2 NCA never improved over Phase 1 CE weights across all folds.
- Full train fold used as reference at inference (460k rows, fp16 → 8.1GB VRAM).

Gate summary (post-run):
- Fold-0 cheap-kill: BA=0.9601 (cleared 0.960 bar with full-reference NCA inference).
- Full 5-fold cv=0.959088, sem=0.000344.
- Err-corr vs n070: 0.690 (above 0.65 threshold — joins wall, not genuinely decorrelated).
- Per-class vs champion: GALAXY +0.014, QSO -0.010, STAR -0.037.
- Bootstrap P(candidate > champion)=0.000 — no promotion.
