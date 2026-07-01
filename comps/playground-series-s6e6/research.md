# research — playground-series-s6e6 (Predicting Stellar Class)

Synthesis of background research (web). Used to seed feature/model nodes. Each idea
becomes an experiment node, scored on the frozen CV — research proposes, CV decides.

## Anchor facts
- **Redshift dominates** (~65% of importance in SDSS DR17 studies); achievable accuracy 97–98%. STAR z≈0, GALAXY low-mod, QSO high. Confirms our EDA.
- **Hard confusion = QSO↔GALAXY** (and STAR↔GALAXY at low z), worst around the 2.5<z<3 stellar-locus crossing where QSO colors overlap stars.
- Our two categoricals (`spectral_type`,`galaxy_population`) are NOT in the standard SDSS Kaggle set — engineered, high-signal but not deterministic (EDA: M→95% GALAXY, Red_Sequence→90% GALAXY). Watch the CV-too-good tripwire, but they're legitimate (present identically in test).

## Feature engineering — ranked by expected value
1. **Base color indices** u-g, g-r, r-i, i-z — DONE (clean.py add_color_features, + u-z).
2. **QSO color-box flag** (highest-leverage new feature): binary "in QSO region" from literature cuts — low-z QSOs at `u-g < 0.6 AND g-r > 0`; UV-excess boundary `u-g > 0.4`. Directly attacks QSO↔GALAXY/STAR.
3. **Redshift transforms**: `log1p(redshift)` (compress QSO tail), coarse bins / "is z≈0 (STAR)" flag, "is high-z" flag.
4. **Full pairwise + second-order colors**: u-r, u-i, g-i, r-z, and curvature terms (u-g)-(g-r), (g-r)-(r-i).
5. **Extinction/dereddening proxy**: galactic latitude `b` (from RA/Dec) — extinction strongest near galactic plane. Moderate EV.
6. **Sky-position**: convert alpha,delta → galactic (l,b). b doubles as extinction proxy. Lower EV, cheap.
7. **Brightness anchor**: keep raw i (or r) magnitude + a couple magnitude ratios. Modest.
8. **Categoricals**: native (CatBoost/LightGBM) or inside-fold target-encoding (leakage risk). Probe importance.
> Skip quartile-binning of magnitudes (GBDTs bin internally).

## Models / ensemble
- GBDTs dominate (LightGBM, XGBoost, CatBoost). Expect 97%+. ✅ building all three.
- **Stack/blend the 3 GBDTs** (OOF-stack or hill-climb weights) — the real gain over any single model. Optionally add a decorrelated NN (RealMLP/TabM) arm — only as diversifier, NN rarely beats tuned GBDT here.
- Optuna ~100-150 trials on the best family.

## The highest-leverage structural idea — One-vs-Rest decomposition
Mirrors the s6e4 1st-place solution. Because STAR (z≈0) is cleanly separable:
1. model A: **STAR-vs-rest** → P1
2. model B: **QSO-vs-GALAXY** on non-star objects → P2
3. recombine: P(STAR)=P1, P(GALAXY)=(1-P1)(1-P2), P(QSO)=(1-P1)P2
Concentrates capacity on the hard QSO↔GALAXY boundary + a dedicated tunable threshold. **Try this as a draft node.**

## Balanced-accuracy levers
1. Post-hoc per-class threshold/prior calibration on OOF — biggest metric lever per literature. (We tried node_0002 → within-noise here because class_weight='balanced' already calibrates; revisit ON the blend.)
2. class_weight='balanced' baseline ✅.
3. Multiclass **focal loss** custom objective — boosts minority recall.
4. Optimize balanced accuracy directly in the threshold search.

## R2 addendum — GBDT tuning & stacking (refines the plan)
- **node_0001 is UNDER-REGULARIZED**: it used LightGBM defaults (min_data_in_leaf=20). At 577k rows that's the classic mistake — use **min_data_in_leaf 100–2000**. Highest-value cheap tuning fix.
- **Real tuning lever = low LR + early stopping**: learning_rate 0.02–0.05, n_estimators high (3000–10000), early_stopping 50–200 picks the count. Don't tune n_estimators directly.
- Capacity: num_leaves 31–255 (center 63–127) / max_depth 6–10. Sampling: feature_fraction 0.5–0.9, bagging_fraction 0.6–0.9 (+bagging_freq). L1/L2 last.
- **Optuna only marginally worth it** — one moderate pass (40–80 trials), then stop; spend budget on diversity + stack instead.
- **Stacking = the real win (~0.3–1.5% on metric)**: L0 = LGBM/XGB/CatBoost OOF prob matrices under the SAME folds.json; L1 meta = multinomial **logistic regression on OOF probs** (edging out hill-climb) AND hill-climb weights — run both, pick by OOF balanced acc. Combine PROBABILITIES not labels, then recalibrate thresholds on the BLENDED OOF. GBDT-meta L2 only if LR-meta OOF says it helps.
- Diversity that decorrelates errors (feeds blend): different libraries (have 3, free), **DART boosting**, alt categorical encodings, feature-subset variants, multi-seed bagging (5–10). Measure pairwise OOF error correlation; add the LEAST-correlated valid member.
- **NN verdict: skip TabNet/FT-Transformer (weak); TabPFN out (577k >> 50k cap). One TabM/MLP arm late, only as a blend diversifier if it lifts blended OOF.** Threshold calibration reportedly 1.5–4% — but we measured within-noise (node_0002) because class_weight already calibrates; revisit ON the blend.

## Planned node queue (research-seeded, CV decides)
- **node_0005**: tuned LightGBM = node_0001 + min_data_in_leaf~200 + lr 0.03 + n_estim 5000 + early_stopping 100 + feature/bagging fraction 0.8. (improve on node_0001 — fixes under-regularization; expected real lift)
- **node_0006**: feature-rich LightGBM = best-LGBM + QSO color-box flag + log/binned redshift + full pairwise colors + galactic (l,b). (improve)
- **node_0007**: one-vs-rest decomposition (STAR-vs-rest, then QSO-vs-GALAXY). (draft — highest structural idea)
- **node_0008**: LGBM+CatBoost+XGBoost OOF stack (LR-meta vs hill-climb), probability blend, recalibrate thresholds. (draft off family — the real win)
- maybe later: DART LightGBM; focal-loss; Optuna pass on leader; multi-seed; one TabM/MLP diversifier.

## 2026-06-12 — Deotte LR-stacker diff + the "look outside" mandate (post-saturation)
Pulled cdeotte/gpu-logistic-regression-stacker (refs/cdeotte_lr_stacker/, VER 9). His stacker vs our champion node_0063:
- HIS recipe: PyTorch multinomial logreg (Adam, C=0.1, weight_decay=1/(C·n), 1000 epochs), **RAW probabilities** (no logit), **5-seed bag** (SEEDS 42..46 × 5 folds = 25 models, averaged), **plain argmax** (NO threshold), class_weight='balanced' + BOOST knob (=1.0). Bases = same 19 we have.
- OUR recipe (n63): sklearn balanced LogReg C=1.0, **clipped log-probs**, **single seed-42**, **DE per-class threshold** for balanced accuracy.
- Two substantive levers pulling opposite ways: HIS 5-seed bag (variance/stability) vs OUR DE threshold (metric edge). node_0069 combines both.
- His current notebook has NO models we lack — "bigger stack from Deotte" is exhausted.

MANDATE for proposals now (8+ consecutive non-promoting nodes; in-house levers all wash): **stop drafting in-house model variants.** The stack only grows from bases that are strong AND decorrelated from the existing 17 — every in-house add has washed (n52/n64/n67, and merging our 15 diluted to 0.970025). The live levers are:
1. NEW EXTERNAL OOF: source other authors' shared OOF/test preds for s6e6 (not in Deotte's bank) — different families (AutoGluon, TabPFN-v2 big-ctx, FT-Transformer, a strong CNN/MLP, a different GBDT recipe) — add as new decorrelated bases. (A sourcing agent is pulling candidates into refs/ext_oof/.)
2. RECIPE on the bank: seed-bag (n69), raw-prob vs log-prob, C sweep, BOOST knob for STAR recall, threshold-after-bag — small but the only honest CV levers left on the meta.
3. A genuinely new external method from discussions/arXiv that produces a decorrelated base.
Anything that just re-adds/re-derives a model we already have = predicted wash, do not propose.

## 2026-06-13T11:50Z — LOOK-OUTSIDE pass (post-plateau, champion n63 0.970153 / LB 0.97073)
Web/arXiv sweep of SDSS star/galaxy/QSO ML. The marginal facts only re-confirm our anchors
(redshift dominates ~65%; colors u-g/g-r…; GBDT≈RF≈NN all converge ~0.965-0.97 OOF — no family ahead).
**Nothing new in the "more models / more colors" direction** — those are all in-bank and washing.

THE ONE NEW ANGLE — **redshift-CONDITIONAL photometric features** (different inductive bias):
- arXiv/ScienceDirect "Disentangling conditional redshift effects in Quasar–Galaxy photometric
  classification" (S2213133726000466, 2026) + photo-z CDE literature (FlexCoDE; Bovy 2012
  XDQSOz generative flux-redshift density). Core idea: every model in our bank leans on the
  MARGINAL redshift as its dominant global splitter; almost none is structurally forced to learn
  the *residual photometric interaction AT FIXED redshift* — the QSO↔GALAXY (and STAR↔GALAXY)
  signal that survives after redshift is "explained away." A feature-set that exposes that
  residual produces errors decorrelated from the tree/NN bank, because trees split on z first
  and then on raw colors in coarse z-regions, never on a continuous z-conditional color anomaly.
- CONCRETE RECIPE (all fit-inside-fold; `fit_in_fold` leak class):
  1. Bin redshift into ~30-50 quantile bins (edges fit on train fold only).
  2. Within each z-bin, fit on the train fold the per-bin MEAN and STD of each color
     (u-g, g-r, r-i, i-z, u-z) and of each magnitude. Transform every row (train+val+test)
     into its **z-conditional color RESIDUAL / z-score**: (color − mean_zbin) / std_zbin.
     → "how anomalous is this object's color GIVEN its redshift." QSOs and galaxies that share
     a redshift but differ in SED shape now separate on a feature trees don't construct.
  3. Smooth alternative (no hard bins, lower variance): regress each color on a spline/poly of
     redshift (fit on train fold), feed the **regression residual** as the feature.
  4. Optionally add the local z-conditional density: kNN in color-space restricted to a z-window
     (instance-based, also decorrelated) — secondary.
- AS A NEW PRIMARY BASE (the only thing that has ever moved honest CV): train ONE strong model
  (RealMLP-ref recipe or a LightGBM, our existing best families) on a feature set that is
  z-residual-DOMINATED — i.e. DROP raw redshift+raw colors and feed mostly the z-conditional
  residuals (+ keep z itself for STAR z≈0). Forcing the model onto the conditional manifold is
  what makes its OOF errors decorrelate from the bank; if it merely ADDS residuals to the full
  raw set it will collapse back onto the same z-split and wash (that is why n6's "redshift
  transforms + QSO box" and n11 already washed — they were ADDITIVE, never conditional-residual,
  never the model's primary signal). Gate: pairwise OOF error-correlation vs the 17-bank BEFORE
  judging — accept only if decorrelated AND blended OOF BA > champion by >2·sem.
- Honesty: this is a HYPOTHESIS for decorrelation, not a guaranteed lift. But it is the only
  genuinely UN-tried inductive bias surfaced — a model whose primary signal is "color anomaly
  at fixed z" rather than "z then color." Everything else outside is already in-bank or clout.

Sources: arXiv 2205.10745 (multimodal SGQ), MNRAS 518/2/3123 & 527/3/4677 (multi-NN photometric
SGQ), A&A aa36770-19 (RF 111M SDSS), arXiv 1105.3975 (Bovy XDQSOz flux-z density),
ScienceDirect S2213133726000466 (conditional redshift disentangling), arXiv 2207.01848/2410.24210
(TabPFN/TabM diversity — already exhausted in-bank).

---
## 2026-06-14 — fresh top-kernel look-outside pass (board climbed to ~0.972 today)
Pulled 5 of the freshest/highest public kernels (snapshotted in `refs/pull_*`), read code, diffed
against our bank. **VERDICT: nothing genuinely new — every technique is either already in-bank,
already-falsified, or pure post-processing clout.** Details by kernel:

- **zoli800/s6e6-0-97184-external-qso-patch** (LB 0.97184, the new top): NOT a model. The notebook
  is literally 2 cells — a markdown blurb + one code cell that base64-gzip-decodes a *precomputed
  submission.csv*. By its own description: "strong user-owned anchor submission (0.97181) + four
  high-confidence GALAXY→QSO corrections" selected by a CV-logistic stacker over the SDSS17 external
  set. The +0.00003 over the anchor = 4 flipped rows scored on the 20% public slice. **CLOUT / LB
  post-processing** (same family as fachri00 ridge-flips). The "external QSO patch" = the same
  fedesoriano SDSS17 dataset whose coord-match we already FALSIFIED in node_0083 (coords ⊥ labels,
  ~50% agreement). Not private-robust, not a lever.

- **barbagrande007/data-augmentation|NN|PseudoLabeling|TTA** (no LB stated; single Keras MLP on a
  30% holdout, no K-fold): three "augmentation" tricks, ALL already exhausted for us —
  (1) **Gaussian jitter** of the train rows ×4 at σ=0.01 → this is exactly the additive-noise
  augmentation family; our n65 mixup already showed confusion-zone augmentation tanks STAR recall,
  and a DAE swap-noise rep (n62) confirmed the 26-dim tabular space has no within-row manifold to
  exploit. (2) **Pseudo-labeling** test rows at p>0.99 then retrain → we did self-distill (n67),
  disjoint-teacher (n74/n79) and self-train GBDT (n46); all wash in-stack. (3) **TTA** = average 20
  jittered copies of each test row at inference → a single-model variance-reduction trick that is
  strictly dominated by our 5-fold + multi-seed bagging (which already washes). No new
  base/feature/representation. **Would NOT decorrelate** (same features, same softmax MLP class as
  our n8/n36).

- **shamanthakreddymallu/s6e6-dcn** (standalone DCN, GPU; CV in the 0.966 band): explicitly "adapted
  from the public NN-2 recipe" — i.e. the SAME `refs/nn-v2-for-s6e6.py` our **n55 DCN was ported
  from**. Same CrossNet+deep-tower, class-mean CE, EMA, qbin(16/32/64) tokens. Two differences vs
  our n55, both already-tested dead ends for us: (i) it **appends external SDSS17 rows at weight
  0.12** with threshold-reconstructed categoricals — our n55 deliberately set `append_original=False`
  and n83 falsified the external labels; (ii) it adds **synthetic-artifact floor/freq tokens**
  (`art_*_floor` + log-count/frequency over train+test) — this is exactly our n60 generator-forensics
  provenance-flag node (solo −0.00033, stack washes). Same architecture → same ~0.70-0.79 err-corr
  band as n55 (which itself only moved an LB *probe*, never honest CV). **Not a new family.**

- **shamanthakreddymallu/s6e6-realmlp-lgbm-catb-xgb-dcn** (the 0.972-band blend kernel): pure
  **combine** — a Nelder-Mead simplex weight-blend over FIVE of the author's own OOF banks
  (LGBM 0.96594 / XGB 0.96586 / CatB 0.96753 / RealMLP 0.96611 / DCN) + a per-class weight grid
  (0.6-1.6, the same "QSO/STAR multiplier" DE-threshold we do as node_0017/0020). Every base is a
  GBDT or RealMLP/DCN we already hold a stronger version of; the external SDSS17 rows are appended
  to the GBDT training pools (n64/n44 territory — washes). **No new model/feature/representation;
  a blend of in-bank families.** Their FE function is a strict subset of our `fs_realmlp_fe`.

- **yekenot/ps-s6-e6-realmlp-pytorch** (80 votes, refreshed today; the *reference* RealMLP impl):
  this is the canonical hand-written-PyTorch RealMLP (PBLD periodic embeddings, n_ens=8, flat_cos
  LR, EMA, label smoothing) — i.e. **the exact recipe our n28 already reproduced** (n28 CV 0.969065,
  our breakthrough base). Diff vs n28: only light hyperparameter cosmetics (epochs=6 vs ours,
  pbld_freq_scale=5.0, a TargetEncoder on factorized-float "combo" categoricals). The combo-TE is
  the same target-encoding family that washed for us (n18). **No recipe improvement that we lack** —
  n28 is already at/above this kernel's level, and seed-bagging n28 (n32/n35) already washed.

**Net:** the public board's climb to 0.972 is 100% (a) simplex/vote blends of the same GBDT+RealMLP+
DCN zoo we hold and (b) LB-slice post-processing (anchor sub + hand-picked GALAXY→QSO flips, ridge
label-flip probing) that overfits the 20% public split and will shake on private. **No new inductive
bias, no decorrelated base, surfaced on the public LB.** The only un-tried decorrelation hypothesis
remains the z-conditional-residual *primary base* idea above — and our own n86 (z-resid TabM) already
came back err-corr 0.72 / wash, so even that needs the *primary-signal* (drop-raw-z) variant, not an
additive one, to have any chance.

## 2026-06-15 — LINEAR FLUX-SPACE is the most decorrelated representation found (node_0103)
node_0103 trained TabM on fs_flux (linear flux f_b=10^(−0.4(mag_b−mag_mean)), pairwise flux RATIOS,
unit-sum SED simplex, + raw redshift; NO magnitudes/log-colors). Result: fold-0 solo BA 0.940
(cheap-killed <0.965) BUT **err-corr vs the 17-bank = 0.485** — the LOWEST of ANY base we have ever
built (RBF-Nystroem n096 was 0.53, FT-T ~0.53, everything else 0.70-0.79). The linear-flux geometry
makes TabM partition the SED-shape space along genuinely different directions than log-color splits,
so its errors decorrelate HARD — exactly the property the saturated stack still lacks.

THE LEVER: n103 died only on STRENGTH, not decorrelation, and the weakness was FEATURE POVERTY — fs_flux
had just 21 features, no categoricals (spectral_type/galaxy_population), no rich-FE aggregates. The
representation is right; the feature set was stripped. CONCRETE NEXT NODE (high priority for next round):
a TabM/RealMLP base on a RICH flux-space FE — flux ratios + unit-sum simplex + flux-space analogues of
the fs_realmlp_fe aggregates (flux means/ranges, flux×redshift interactions) + the two categoricals
(native or PLR-embedded) + raw redshift kept. Target: push solo BA back to ≥0.965 (tier) while KEEPING
err-corr <0.65. If it clears both, stack-add to n091 — this is the first genuine shot at a
strong-AND-decorrelated base since FT-Transformer. (Decorrelation already proven at 0.485; the only
open question is whether richer flux-FE recovers the ~2.5pp BA that the sparse 21-feat set lost.)

## 2026-06-16T08:57Z — Drop-model (LOO) study: the pool is saturated, CatBoost is the load-bearing family
Leave-one-base-out over the champion node_0091 FULL pool (63 bases, 5-fold OOF, fixed C=0.003,
delta = cv_full - cv_without; positive = base helps). cv_full=0.970316, per-fold sem=0.000274.
- MAX contribution = cat-3 (+0.000158), the only base whose removal even approaches 1*sem. Everything
  else <= +0.00006. So NO single base is individually significant => the pool is maximally redundant
  (every base backed up by correlated peers). Quantitative confirmation of the NCL-cliff saturation.
- |coef| != importance: node_0039 (our CatBoost) carries the LARGEST coefficient (0.977) yet its causal
  LOO delta is NEGATIVE (-6e-6) — cat-3 fully covers it. Use LOO-delta, not coef size, to rank contribution.
- Family ranking by causal contribution: CatBoost (cat-3 #1 by 4x) >> RealMLP (realmlp-2) > LGBM
  (node_0003, lgbm-5). TabM is near the BOTTOM (tabm-0 rank 62, slightly harmful). xgb-6/tabm-0/node_0042
  are mildly harmful (removing them lifts CV within noise).
- Implication for "bag the top models": more SEEDS of any family will wash (a 0.99-correlated copy of a
  base whose total contribution is <1*sem adds ~0). The one untried, data-directed shot is TRUE bagging
  (bootstrap rows + feature subsampling, for real decorrelation) of the CatBoost family specifically — the
  prior seed-bag n075 bagged TabM, the wrong (near-worst) family. -> node_0115.

## 2026-06-16T12:20Z — Look-outside (user-directed): exhausted, no new signal
- Fresh top-kernel re-scan (2 days after 06-14): top = cdeotte GPU-LR stacker (122 votes), yekenot
  RealMLP-PyTorch (80), philippsinger TabPFN-3, pilkwang — all already in our bank. New since 06-14:
  kospintr baseline (CatB/HGBC/XGB/LGBM/RealMLP), meenalsinha ensemble, abbas829 stacking-ensemble,
  aarishasifkhan FT-T-from-scratch — every one a standard GBDT/stack/blend over the same 8 features.
  No new signal source, no novel representation, no decorrelated base. Standing read re-confirmed.
- ORIGINAL-DATASET FEATURE-MATCH (the one untested join; n083 only did coords). Matched comp train+test
  to fedesoriano star_classification.csv (100k) on key (u,g,r,i,z,redshift), rounded dec 6..2:
  0 matches at ALL precisions. Comp marginals are SHIFTED vs original (u in [-0.14,28.25] vs [9.82,32.78];
  redshift mean 0.723 vs 0.577) — the synthetic generator reshaped the distributions (matches the
  high train-vs-SDSS17 adversarial AUC the discussions reported). The generator preserves NO row
  identity → no real-label recovery from the original. External-data avenue CLOSED (both coord + feature).
- NET: every lever is now closed with evidence. Champion n091 (CV 0.970355 / LB 0.97121) is the genuine
  information ceiling. The board 0.972 cluster is clout (vote-blends + LB-flip-probing over the same
  public zoo), below private sigma and private-fragile. The rational move is finals selection, not more nodes.

## 2026-06-17T09:44Z — METHOD look-outside (arXiv/web, post round-0116-0120): ONE new lever found — SDR
After the Kaggle ecosystem re-confirmed exhausted, searched the METHOD literature (the avenue never tried this comp). Result: ONE concrete buildable lever that is NOT wall-bound, plus several confirmed wall-bound.

**LEVER — Sharpened Dimensionality Reduction (SDR) base.** Source: "Supervised star, galaxy, and QSO classification with sharpened dimensionality reduction," A&A 2024 (aanda.org/articles/aa/full_html/2024/10/aa50214-24/aa50214-24.html) — SAME task (star/galaxy/QSO). Method: iteratively shift points toward higher-density regions (mean-shift-like local-gradient sharpening) BEFORE a DR projection (UMAP/LMDS), then kNN-classify in the sharpened embedding. Reported precision 99.7/98.9/98.5% = tier-comparable to RF/XGB, NOT sub-tier. Libraries-first: authors ship pySDR/SHARC; if those don't uv-install, the sharpening is a small mean-shift loop over umap-learn + sklearn kNN.
WHY IT MAY BEAT THE WALL: every existing base partitions the ORIGINAL 26-D feature space, so all their errors concentrate on the same physically-ambiguous GALAXY↔QSO overlap → that shared hard zone is exactly why they correlate ≥0.72 at strength. SDR classifies by MANIFOLD GEOMETRY after density sharpening → errors driven by projection topology + density estimation, not axis-aligned/margin partitioning. A representation-CLASS change (not a reparametrization), which is the specific gap the 6 prior decorrelation attempts (flux/RBF/z-resid/NCL/STAR-gate/fingerprint — all still feature-space partitioners or reparametrizations) never filled. Build: one node, fit UMAP/SDR INSIDE each fold (embedding is a cross-row stat → fit_in_fold, fold-local), sharpen, kNN, emit OOF; CPU, minutes-to-~1-2h. Cheap to falsify: if OOF err-corr ≥0.72 vs bank like everything else, kill — wall confirmed a 7th time.

**Wall-bound / rejected (checked, not drafted):**
- TabPFN-2.5 / TabICL-v2 / LimiX-16M (arXiv 2511.08667, 2509.03505; pip tabicl/tabtune): stronger TFMs but same in-context-transformer paradigm as TabPFN-v2 already in bank → will correlate with existing transformer bases. Only worth a champion-tier base SWAP (replace TabPFN-v2), not a decorrelating add.
- ModernNCA / TabDPT (retrieval/learned-metric kNN, LAMDA TALENT toolkit): borderline, but kNN-density signal already probed + capped sub-tier → likely wall-bound.
- Synthetic-data latent recovery beyond decimal-fingerprint (n118): no published buildable method for a single static synthetic table.
- Correlation-penalized meta-learners (XStacking arXiv 2507.17650, diversity-guided stacking): only re-confirm the LOO saturation finding; nothing rescues an already-saturated combiner. Covered.

## 2026-06-17T18:27Z — METHOD look-outside #2 (autonomous): PHYSICS feature lever for low-z GALAXY↔STAR — STRONG, NEW
Targeted search on the dominant remaining error (low-z GALAXY↔STAR confusion, GALAXY recall the bottleneck). Found a genuinely new, physically-grounded, STATELESS feature block (ugriz + z only, row-wise deterministic, no fit → leak-safe stateless) NOT on the ruled-out list (not flux/RBF/SDR rep, not focal/reweight/threshold, not redshift-removal):

**fs_physloc block** (add to a strong GBDT base):
1. **[Fe/H]_phot — Ivezić 2008 (ApJ 684,287) eq.4**, photometric metallicity cubic in (u−g,g−r). Stars land at real [Fe/H] (−2.5..+0.5); low-z galaxies (composite SED, 4000Å break) fall OUT-OF-RANGE → a calibrated stars-vs-galaxy axis trees can't synthesize from raw colors. Formula:
   x = (u−g) if (g−r)<=0.4 else (u−g) − 2*(g−r) + 0.8 ;  y = (g−r)
   [Fe/H] = -13.13 +14.09*x +28.04*y −5.51*x*y −5.90*x^2 −58.68*y^2 +9.14*x^2*y −20.61*x*y^2 +0.00*x^3 +58.20*y^3  (Bond 2010 signed coeffs)
   Do NOT clip to physical range (out-of-range = the galaxy signal). Optional bool feh_in_range=(-3<feh<0.6).
2. **P2s — stellar-locus PERPENDICULAR distance** (Ivezić 2008 principal colors): P2s = −0.249*(u−g) + 0.545*(g−r) + 0.234. Single stars cluster |P2s|≈0 (MS band −0.06..0.06); low-z galaxies off-locus → |P2s| large. The cleanest single "on the stellar track?" scalar. Companion P1s = 0.910*(u−g)+0.415*(g−r)−1.28 (along-locus, metallicity/temp axis).
3. **z_warp = log10(z + 3e-4)** (ε≈SDSS z error floor; or asinh(z/ε)) — re-express (NOT remove) redshift to EXPAND the z≈0 neighbourhood: stellar-noise-z cloud vs smallest real galaxy z (~0.02) get a wide stable margin instead of a sub-0.01 crush. Attacks the exact bottleneck boundary.
Calibrated for F/G dwarfs (0.2<g−r<0.6); extrapolation outside is intended (galaxies/M-dwarfs → out-of-range = signal). Synthetic data has no extinction → apply to raw ugriz. Sources: Ivezić 2008 tomographyII.pdf eq.4 + §3; Bond 2010 signed coeffs; Gu 2015 (1507.02054) validity. DRAFT: add fs_physloc to a strong GBDT (copy n030), report solo BA + err-corr vs n070 + stack-add to n091 + pred_diagnostic low-z fix-block. If it wins, ABLATE the block (feh_phot & P2s likely carry it). → node_0124.

## 2026-06-18 — arXiv FEATURE pull (user-directed): 3 genuinely-new axes exploiting OUR edges (z + synthetic)
Honest frame: the canonical paper on our exact 3-class SDSS problem (Clarke 2020 A&A 639 A84) gets its lift from WISE mid-IR (w1−w2) + morphology — neither of which we have, and its largest residual is STILL "QSO missed as galaxy at z<1". So a new ugriz color won't move it. BUT two edges we have, that the photometry-only literature lacks, open real new axes: (a) we HAVE spectroscopic redshift; (b) our data is SYNTHETIC. Three buildable, leak-safe feature sets NOT on our tried list:

★1 (TOP PICK) — PER-CLASS TEMPLATE-FIT χ², z-gated [fs_tmplchi2, STATELESS]. Fit each row's 5 ugriz fluxes to 3 template families redshifted to the KNOWN z + integrated through SDSS bandpasses; closed-form amplitude α*=Σ(f·T/σ²)/Σ(T²/σ²), χ²_t=Σ(f−α*T)²/σ². Features: chi2_{gal,qso,star} + pairwise diffs (chi2_star−chi2_gal, chi2_qso−chi2_gal) + argmin class + softmax posterior. WHY NEW AXIS: a star fits a 1-param stellar template tightly but galaxy/QSO poorly; z≈0.9 QSO fits power-law+lines well, galaxy poorly; galaxy fits CWW (4000Å break) well — DIFFERENT 5-band residual structures, NOT any single color. Knowing z removes the redshift-degeneracy that hobbles photometric-only template fitting = our genuine edge. Hits BOTH confusion channels. BUILD: `uv add speclite` (SDSS sdss2010 Doi 2010 filters); templates CWW (E/Sbc/Scd/Im), Vanden Berk 2001 QSO composite (astro-ph/0105231), Pickles 1998 stellar (VizieR VI/61); precompute synth ugriz on Δz≈0.01 grid → per-row linear algebra, NO training. STATELESS (no fit/target/cross-row). Source: CPz Fotopoulou&Paltani 2018 A&A619 A14 / arXiv:1808.04977; Fadely-Hogg-Willman 2012 arXiv:1206.4306. CAVEAT: CPz says NIR carries dominant power (optical star-score ~96.5%), but its galaxy half IS the 4000Å-break + star half IS the blackbody-locus — build #1 first; ★2 may be redundant if #1 wins.

★2 — z-GATED CONTINUUM-RELATIVE SED-SHAPE [fs_sedshape, STATELESS] (distinct from our z-conditional color RESIDUAL — that subtracted POPULATION-median color in a z-bin; THIS subtracts each object's OWN interpolated continuum at the physical line/break position). 2a MgII bump-excess (z≈0.9 QSO marker): at z≈0.9 rest-2800Å lands in r (obs≈6175Å); bump_excess = mag_r − (w·mag_g+(1−w)·mag_i), w≈0.55 = r minus log-flux continuum interpolated from non-line neighbors g,i; sign-definite for QSOs. z-gated form: band B nearest 2800·(1+z), emit mag_B − continuum_interp(neighbors). Source: Richards 2001 AJ121 2308 / arXiv:astro-ph/0012449 (quasar λ_eff 3651/4679/6175/7494/8873). 2b z-gated D4000 break proxy (galaxy>1.3, QSO≈1): mag offset across the 2 bands straddling 4000·(1+z) (low-z→(u−g,g−r); z≈0.9→(r−i,i−z)). Sources: PAU arXiv:2201.04411, Beck 2016 arXiv:1603.09708. NOTE low-z 2b ≈ curvature u−2g+r; build ONE per node for attribution; LEAD with 2a (surgical on the z≈0.9 channel).

★3 — PER-CLASS AUTOENCODER reconstruction-error GAPS [fs_aerecon, fit_in_fold] — exploits SYNTHETIC data. Train 1 tiny AE per class on standardized (ugriz,z,colors); per row emit the 3-vector of recon errors + DIFFS (err_STAR−err_GAL, err_QSO−err_GAL). A low-z galaxy confusable with a star reconstructs almost as well under STAR-AE as GAL-AE → the gap = "which manifold am I closer to", a GBDT-splittable signal + a difficulty/label-noise score. fit_in_fold (each AE on train-fold rows of its class only). Source: Marks/Griffin/Corso 2024 arXiv:2412.02596 (per-class recon-error-ratio feature). Weaker cousins: local-covariance anomaly (Frobenius dist of per-row kNN-ball covariance to global — generators distort joint structure most in transition zones = our confusion regions; arXiv:2603.17041/2503.20903; distinct from our kNN class-fraction = geometry not labels); per-class density-ratio log p(x|GAL)−log p(x|STAR) via KDE/flow.

REJECTED (info ceiling): WISE w1−w2/z−W1 (the actual workhorse, Clarke2020/Zeraatgari2024 — NO ugriz proxy, it's 3-5µm hot dust); morphology/resolved_r (breaks low-z star/gal, we lack it); KX/UVX/XDQSO boxes (QSO-vs-STAR not QSO-vs-galaxy, KX needs K-band); power-law slope α (collapses to u−z color). VERDICT: broadband-optical alone can't FULLY break the 2 degeneracies, but ★1/★2/★3 are real untried axes via the z + synthetic edges. Build order: ★1 → ★3 → ★2a; ablate ★2 vs ★1 to avoid double-counting the break.
