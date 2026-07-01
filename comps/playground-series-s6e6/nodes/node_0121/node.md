---
id: node_0121
desc: SDR sharpened-manifold kNN base
op: draft
parents: [root]
uses_data: [fs_sdr]
family: nn
status: dead
stage: decided
metric: Balanced Accuracy Score
direction: maximize
cv: null
sem: null
folds: [0.897402]
baseline_cv: 0.970355
gates: {schema_ok: null, oof_full: false, no_nan: null, dist_sane: null,
        leak_clean: true, cv_too_good: false, passed: false}
gate_note: "fold-0 BA=0.8974 < kill threshold 0.94; pySDR not available (wrong PyPI package); fell back to umap+kNN; representation sub-tier vs photometric space"
leak: clean
lb: null
submitted: null
created: 2026-06-17T09:44Z
decided: 2026-06-17T10:07Z
tags: [sdr, manifold, outside, draft, fs_sdr]
---

## plan
built on:   root (a fresh base — a new representation CLASS, not a reparametrization of the 26-D feature space)
change:     Sharpened Dimensionality Reduction (SDR) base. Iteratively shift points toward higher-density
            regions (mean-shift-like local-gradient density sharpening) BEFORE a DR projection (UMAP/LMDS),
            then kNN-classify in the sharpened embedding. New feature-set fs_sdr (leak class: **fit_in_fold**).
hypothesis: Classifying by manifold geometry after density sharpening produces errors driven by projection
            topology + density estimation, NOT axis-aligned/margin partitioning of the original space — so it
            may decorrelate from the ENTIRE bank (all of which partition the original 26-D space and share the
            physically-ambiguous GALAXY↔QSO overlap hard zone) while staying tier-strength.
target:     Balanced Accuracy maximize. Cheap-kill if fold-0 solo BA < 0.94. The PAYOFF is decorrelation:
            valuable if err-corr vs the bank (node_0070) < 0.65 AND solo BA is tier-ish, OR if stack-add to
            champion n091 (cv 0.970355) beats it by > 2·sem (0.000498). Honest: if it lands ≥0.72 err-corr at
            strength like the 6 prior decorrelation attempts, it confirms the wall a 7th time and gets killed.

This is a method-literature look-outside lever (well = outside). Source: "Supervised star, galaxy,
and QSO classification with sharpened dimensionality reduction," A&A 2024
(https://www.aanda.org/articles/aa/full_html/2024/10/aa50214-24/aa50214-24.html) — the SAME task
(star/galaxy/QSO). Reported precision 99.7/98.9/98.5% = tier-comparable to RF/XGB, NOT sub-tier.

Family note: labelled `nn` as the closest valid frontmatter family value, but it is really a
manifold-geometry base (mean-shift density sharpening + DR embedding + kNN), not a standard neural net.

LIBRARIES FIRST (hard rule 8): try the authors' pySDR/SHARC via `uv add` first. If they do not install
cleanly, fall back to a small mean-shift sharpening loop over `umap-learn` + sklearn kNN — and say so
EXPLICITLY in the node prose / journal if you fall back, with the reason.

Input features: the BASE photometric set (u, g, r, i, z, redshift, colors) — the same numeric inputs as
fs_realmlp_fe's base columns (see data.md fs_realmlp_fe recipe). The SDR EMBEDDING is the new feature-set
fs_sdr.

LEAK CLASS = **fit_in_fold** (CRITICAL): the UMAP/SDR embedding + the density sharpening + the kNN are
ALL cross-row stats → they must be fit on the TRAIN FOLD ONLY each fold, then val/test rows
transformed/embedded via the fitted train-fold embedding. NEVER fit the embedding on full train or test.
The node's self-gate must VERIFY this by reading its own fold loop (mirror the fold-local cross-row-stat
pattern used by node_0096 / node_0106 — read those node.md files for the embedding-fit-inside-fold shape).
Load folds from the frozen folds.json.

CPU, minutes-to-~1-2h. Emit OOF (n_train×k) + test_probs (n_test×k) aligned to the frozen folds so the
restack / err-corr study can consume it.

READ pointers for the developer:
- research.md tail entry "2026-06-17T09:44Z — METHOD look-outside" (full lever spec + paper URL)
- data.md fs_realmlp_fe recipe (for the base numeric columns)
- nodes/node_0096/node.md and nodes/node_0106/node.md (prior fit_in_fold cross-row-stat bases — the
  fold-local-embedding pattern to mirror)
- the frozen folds.json

## notes
pySDR on PyPI (pysdr==1.2) is a software-defined RADIO library — not the A&A 2024
astronomy SDR paper's code. No installable version of the paper's code exists on PyPI.
Fell back to mean-shift sharpening (2 iter, k=15) + UMAP (n_components=8, n_neighbors=15) + kNN.
Torch cuda remained True after umap-learn install.
Fold-0 BA = 0.8974 < kill threshold 0.94 → killed. This representation is sub-tier vs the
photometric feature space: the mean-shift embedding loses discriminative structure that raw
log-mag/color space preserves, especially the redshift~0 STAR separation.
Err-corr vs n070 was NOT computed (node killed before full OOF).
