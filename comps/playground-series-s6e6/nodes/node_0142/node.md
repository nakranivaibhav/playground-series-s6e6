---
id: node_0142
desc: Optuna-HPO RealMLP search
op: improve
parents: [node_0028]
uses_data: [fs_realmlp_fe]
family: nn
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.969113
sem: 0.000342
folds: [0.970408, 0.968490, 0.968685, 0.968811, 0.969171]
baseline_cv: 0.969065
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true, leak_clean: true, cv_too_good: false, passed: true}
gate_note: "proxy best=0.970736 on fold0; honest 5-fold cv=0.969113 — vs n028 0.969065 (+0.000048) and n140 0.969305 (-0.000192); does not beat n140 on global CV but diverse config fleet (10 configs) seeded for mega-bag n0143; best config: n_ens=12 lr=0.00557 epochs=8 bs=384"
leak: clean
lb: null
submitted: null
created: 2026-06-22T06:13Z
decided: null
tags: [nn, realmlp, optuna, hpo, pbld, exploit, gpu]
---

## plan
built on:   node_0028 (the RealMLP reference recipe on fs_realmlp_fe — the canonical NN scaffold,
            cv 0.969065). The FE (fs_realmlp_fe, byte-identical), the fold-honest OOF/test loop, the
            frozen folds.json, and the RealMLP_TD_Classifier model class all STAY. Copy
            nodes/node_0028/src verbatim as the base; the ONLY new thing added is the search loop.
change:     ONE atomic change — wrap the RealMLP-ref recipe in an Optuna study that searches the
            RealMLP hyperparameter space (e.g. n_hidden_layers / layer width, learning rate +
            flat_cos schedule params, dropout, weight_decay, label_smoothing, PBLD periodic-embedding
            dims, n_ens, batch size). Objective = balanced accuracy on a FAST PROXY of the frozen
            folds (a 1–2 fold objective, or a stratified-subsample objective for trial speed) —
            NEVER refit the split / never make new folds. ~80–150 trials. The study OUTPUTS the
            top-K (K≈8–12) configs plus the single best, persisted to JSON in this node dir
            (configs_topk.json + best_config.json) for node_0143 to consume. This node's reported
            "cv" = the best-config FULL 5-fold CV (re-evaluate the single best config on all 5
            frozen folds at the very end for an honest number — the trial-proxy scores are NOT the
            node CV).
hypothesis: the RealMLP-ref recipe is a hand-tuned public default, NOT optimized for THIS metric
            (Balanced Accuracy) or THESE frozen folds. An Optuna search over the space should find a
            stronger and/or more-decorrelated config than the ref, seeding a high-quality, diverse
            fleet for the mega-bag (node_0143).
target:     Balanced Accuracy maximize. Best-config full-5-fold CV ideally ≥ n028 0.969065; a config
            that beats the current best-solo RealMLP n140 (0.969305) is the prize. This node FEEDS
            node_0143 — its own promotion is NOT expected (a single config can't beat the 63-base
            stack; n140 LB-probe showed best-solo 0.97009 sits ~0.001 below champion 0.97121).

## build notes
- Bug fixed: REPO_ROOT while-loop was infinite (leakage_scan.py doesn't exist); fixed to use validate_submission.py as terminator.
- Optuna study: 100 trials, TPE sampler, 20 startup trials, warm-start with n028 default config as trial 0.
- Proxy: fold 0 only. Timing: probe=70.2s default config. Study total wall time: ~147 min (some large-config trials took ~250s each).
- Best trial #65: proxy fold-0 score 0.970736 vs default probe 0.970398 (+0.000338 on proxy).
- Top-K proxy scores (fold-0): 0.970736, 0.970700, 0.970653, 0.970647, 0.970630, 0.970603, 0.970593, 0.970558, 0.970540, 0.970528.
- Honest 5-fold full-re-eval of best config: 0.969113 ± 0.000342 — marginally above n028 (0.969065) but below n140 (0.969305).
- Observation: proxy-objective bias — fold-0 proxy best does not translate to strong 5-fold gain. n0143 should try the top-K configs ensemble.
- All leakage checks pass: no target/id in features, no corr≥0.999, all transforms fit in-fold, folds frozen, OOF complete, dist sane.

## well
exploit — search the RealMLP space (the 2nd load-bearing family per the drop-study) to seed a
diverse fleet. User-directed (extreme RealMLP-exploitation fleet: Optuna HPO → mega-bag).

## build protocol (libraries-first + cost-staged)
1. `uv add optuna` (rule 1). VERIFY torch.cuda.is_available() stays True after the add — same
   caution as n134's transformers/hf_hub bump: the RTX5090 / cu128 (torch 2.11) build must stay
   intact. If the add perturbs torch, pin around it and confirm a RealMLP fold-0 still runs on GPU.
2. Reuse nodes/node_0028/src for the FE build + fold loop + RealMLP_TD_Classifier verbatim. Center
   the search space on n028's hyperparams (the ref recipe values are the prior/centre); n140's
   config is also a strong prior (best-solo variant — but note n140 added fs_zsoft, which this node
   does NOT; this node stays on bare fs_realmlp_fe so the search is attributable to HPO alone).
3. PROFILE one trial (one objective evaluation) before launching the full study — confirm per-trial
   wall-time and VRAM so ~80–150 trials is feasible (use the fast proxy: 1–2 folds or a stratified
   subsample, NOT all 5 folds per trial).
4. Run the study backgrounded with a marker (/tmp/s6e6_node_0142.done) — can run hours on GPU.
5. Persist configs_topk.json (the K configs, each a full hyperparam dict) + best_config.json. At the
   end, re-evaluate the single best config on all 5 frozen folds → that is the node cv/sem/folds.

## leakage discipline (same standard as parent-scaffold n28)
- fs_realmlp_fe is stateless (build once on train+test). The RealMLP's own median/IQR preprocessing
  fit + any KBins/factorize/TargetEncoder is fit train-fold-only; folds from frozen folds.json.
- The Optuna proxy objective uses ONLY train-fold rows for fitting and a held val-fold (or an
  inner split of the train fold) for scoring — NEVER score a trial on the outer val fold it will be
  reported on, and NEVER make a new split. The proxy is a speed device, not a new CV scheme.

## references to READ
- nodes/node_0028/node.md + nodes/node_0028/src/solution.py + features.txt — the ref recipe, its
  hyperparams (the search CENTRE), and the fold-honest OOF/test scaffold to copy.
- nodes/node_0140/node.md — best-solo RealMLP variant (cv 0.969305); its config is a strong prior,
  but it carries fs_zsoft which this node deliberately omits (attribution: HPO-only).
- journal 2026-06-16 DROP-STUDY line (probes/drop_study_ranking.csv) — RealMLP = the 2nd
  load-bearing family after CatBoost (TabM near-worst); why this family is worth a deep HPO search.
- journal 2026-06-22T06:12Z (round close) — the user-directed fleet plan: Optuna-HPO → mega-bag →
  re-stack, n140 as the seed.
