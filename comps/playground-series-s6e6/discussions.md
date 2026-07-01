# Kaggle discussions — playground-series-s6e6 (pulled 2026-06-06)

Source: `kaggle competitions topics list/show`. The three that matter for us:

## 1. spectral_type & galaxy_population are DETERMINISTIC color cuts (topic 703535, 36 votes)
`broccoli beef` showed the two categoricals are just thresholded color indices:
```python
spectral_type    = cut(r-g, [-inf,-1,-0.5,0,inf], ['M','G/K','A/F','O/B'])
galaxy_population = cut(u-r, [-inf, 2.2, inf], ['Blue_Cloud','Red_Sequence'])
# verified exact: spectral_type(X.g,X.r).equals(X.spectral_type)  etc.
```
**Implication for us:** these categoricals are REDUNDANT with our color features (we already have
r-g, u-r). This explains why node_0018's target-encoding of them REGRESSED — there was no new signal.
- Concatenating the "original" SDSS17 dataset is possible (apply the same cuts), BUT a commenter
  (`nybbler`) reports the train↔SDSS17 **adversarial AUC is high** → distribution drift, so appending
  likely won't help. (Matches our suspicion that train/test drift caps us.)

## 2. The LB is NOISE-LIMITED — and the recipe that reaches the ceiling (topic 704512)
`Siddhesh Sathe`'s variance analysis (Chris Deotte concurs in comments):
- Per-model OOF balanced-accuracy lands in a tight **0.9632–0.9688** band; no family is ahead.
  (Exactly what we found — our arms are 0.957–0.965.)
- **The ceiling recipe (what the top uses), honest nested CV:**
  1. base zoo: XGBoost×2, CatBoost, RealMLP, TabM (+ public blends)
  2. **STACK** = `LogisticRegression(class_weight='balanced')` on the **log-probs** of each base's OOF
     (`log(clip(p,1e-7,1))`, concatenated). The balanced weighting is ESSENTIAL — a plain multinomial
     meta scored 0.9627 (worse than its bases) from majority-class bias.
  3. **threshold calibration:** `differential_evolution` over 3 per-class multipliers (bounds 0.1–5),
     maximize BA on OOF, then `argmax(prob * weights)`. Fit INSIDE the training folds only.
  - **Result: honest OOF BA ≈ 0.9697 → ~0.9707 public.**
- Metric resolution: public (20%, ~49.5k) 1σ ≈ **0.00087**; private (80%, ~198k) 1σ ≈ **0.00035**.
  Top ~25 teams sit inside a 0.0002 window — below private σ. The ordering is draw-dominated.
- **"Pick 2 decorrelated" does NOT help** (4000-resample Monte-Carlo: uplift ≈ 0). Submit your two
  highest-expected-BA blends and stop. → directly informs /kaggle-final.
- Only structural rule: `redshift == 0.0001 → QSO` (100% pure in train), but a competent model
  already predicts those QSO, so forcing it adds zero. (Worth a one-line verify we get them.)

## 3. The stacker starter (topic 704014, Chris Deotte GM, 26 votes)
- A **GPU multinomial Logistic Regression stacker** (hand-written in PyTorch for speed + true
  multinomial calibration). "Save OOF + test PREDS for each model, give the RAW predictions to the
  LogReg stacker — it ensembles and calibrates at the same time. No pre-calibration needed."
- This is his April-playground 2nd-place stacker. Alternatives mentioned: **TabPFN-3 as a stacker**
  (Psi/philippsinger), RealMLP base.

---
## ACTIONABLE for us (current state: blend 0.965889 CV / 0.96704 LB; node_0017 thresh 0.966084)
1. **Build a balanced multinomial LogReg stacker** on our base OOF log-probs (n6/n4/n1/n9 + others),
   class_weight='balanced', + **DE per-class threshold** — all under honest nested CV. This is the
   lever we have NOT pulled: node_0017 did grid-threshold on a probability-AVERAGE blend; the top does
   a balanced-LogReg STACK + DE threshold. Their 0.9697 OOF vs our 0.965889 is the gap to chase.
2. node_0018 (target-enc) failure is now EXPLAINED (redundant deterministic cuts) — don't retry FE on
   spectral_type/galaxy_population.
3. For /kaggle-final: the "two decorrelated picks" Monte-Carlo says it's futile — lock the two
   highest-CV blends, don't chase decorrelation.
4. Optional verify: confirm our champion already predicts redshift==0.0001 rows as QSO (should be free).

---
## 2026-06-13T11:50Z — LB + top-kernel re-scan (champion n63 0.970153 / LB 0.97073)
LB top: 0.97173 (Optimistix), then a DENSE cluster — ~20 teams pinned at 0.97146 incl. Chris
Deotte 0.97148. This flat shelf IS the draw-dominated vote-blend band the variance analysis
(topic 704512) predicted; below private σ, ordering is noise.
Pulled the 3 newest top public kernels we lacked (snapshotted in refs/):
- **fachri00/ridge-flips-0-9717** (#2-near-top LB, 0.9717): pure PUBLIC-LB-PROBING. Regresses
  per-row label-flip vectors of historical submissions against their PUBLIC scores with a Ridge
  model, then applies the highest-coef flips to an anchor sub. No model, no feature — it overfits
  the 20% public slice and will shake out on private. CLOUT, not a lever. Do NOT replicate.
- **makthanithin/2-97146-…-featur**: 5-way hard majority VOTE over CdeOtte public subs
  (cat-v3/realmlp-v5/nn-v2/xgb-v5 + Kazeminia binary-chain), splits test by vote agreement.
  Bases all already in our bank. No new model/feature. Vote-blend.
- **mehrankazeminia/s6e6-…-deeper-look**: narrative "ambiguous rows can't be stacked away"
  diagnostic over the same public subs. Confirms the plateau; no new signal.
VERDICT: the entire top of the LB is vote-blends + LB-flip-probing over the SAME public base zoo
we already hold. There is NO new modelling recipe on the public LB. The only honest lever lives
in research.md (z-conditional residual base) — outside the public-notebook ecosystem entirely.

---
## 2026-06-14 — fresh top-kernel re-scan (LB top climbed to ~0.972; champion n63 0.970153 / LB 0.97073)
Pulled the 5 freshest/highest public kernels (`refs/pull_aug_tta`, `pull_qso_patch`, `pull_dcn`,
`pull_realmlp_dcn`, `pull_realmlp_torch`) and diffed against our 88-node graph. **What's driving the
board to 0.972 is CLOUT, not new modelling.** Clout vs genuine, per kernel:

- **zoli800 0.97184 "external QSO patch" (current #-near-top) — CLOUT.** It is a precomputed
  submission.csv embedded as base64-gzip in a 2-cell notebook: an anchor sub (0.97181) + **4
  hand-picked high-confidence GALAXY→QSO label flips**. The +0.00003 is 4 rows on the 20% public
  slice. Will shake on private. The "external" source is the same fedesoriano SDSS17 set we already
  falsified (n83). Do NOT replicate; do NOT spend an honest slot chasing it.
- **shamanthakreddymallu realmlp-lgbm-catb-xgb-dcn (0.972 band) — GENUINE-but-in-bank.** Honest
  simplex weight-blend of 5 of his own OOF banks (LGBM/XGB/CatB/RealMLP/DCN) + per-class weight grid.
  All families + the per-class multiplier are things we hold a stronger version of. The 0.972 LB is
  blend+threshold variance over the same zoo, not a new signal.
- **shamanthakreddymallu s6e6-dcn — in-bank.** The nn-v2 DCN our n55 was ported from; adds external
  SDSS17 rows (n83-falsified) + artifact floor/freq tokens (= our n60, washed). Same architecture.
- **barbagrande007 aug|NN|pseudolabel|TTA — already-exhausted.** Jitter-aug (our n65/n62 territory),
  p>0.99 pseudo-label (n46/n67/n74/n79), test-time jitter-averaging (dominated by our bag). No K-fold.
- **yekenot RealMLP-PyTorch (80 votes, refreshed today) — = our n28.** The canonical RealMLP recipe
  we already reproduced (n28 0.969065). Only cosmetic HP diffs + a combo TargetEncoder (our n18, washed).

**Standing read (unchanged, now re-confirmed at 0.972):** the entire top of the LB is (1) simplex/vote
blends of the GBDT+RealMLP+DCN public zoo we already hold and (2) public-LB post-processing (anchor +
GALAXY→QSO flips, ridge flip-probing). There is STILL no new public modelling recipe and STILL no
decorrelated base on the board. The CV↔LB gap to the leaders is base-set + LB-overfit, not a stacker
we're missing (n53/n72/n80 already confirmed the stacker is saturated). Hold the two honest finals
candidates (n70 FT-T base 0.97087 honest-best; n76/n84 stacks); treat the 0.972 cluster as the
draw-dominated, private-fragile shelf the variance analysis predicted.

---
## 2026-06-16T12:20Z — fresh re-scan + original-dataset feature-match (look-outside, user-directed)
Top kernels unchanged from 06-14 (cdeotte stacker 122, yekenot RealMLP 80, TabPFN-3, pilkwang); new
ones (kospintr baseline, meenalsinha/abbas829 ensembles, aarishasifkhan FT-T) are all standard
GBDT/stack/blend over the same 8 features — no new signal/representation. NEW TEST this round: matched
comp rows to the original fedesoriano SDSS17 (100k) on photometric FEATURE values (u/g/r/i/z/redshift),
not coords (n083 did coords). Result: 0 matches at any precision + shifted marginals → fully-synthetic
generation, no row identity preserved. The external-data join is dead in BOTH forms. No honest lever
remains; the 0.972 top stays clout (vote/flip), private-fragile.

## 2026-06-17T09:41Z — look-outside (round 0116-0120 close): Kaggle ecosystem RE-confirmed exhausted
mehrankazeminia "a deeper look at the results" (2026-06-17, refs/pull_mehrankazeminia_s6e6-stellar-a-deeper-look-at-the-results/): NOT a new signal source — a hard-VOTE post-processing kernel. Mode-vote over 7 in-bank public subs (cdeotte cat-v3/realmlp-v5/nn-v2/xgb-v5, omidbaghchehsaraei lgbm+flaml, binary-chain=our node_0049), bucket test rows by agreement count 3-7, and on ONLY the maximally-ambiguous count==3 rows OVERWRITE the anchor's label with cdeotte xgb-v5. Anchor = zoli800/s6e6-0-97209-clean-final (LB 0.97209) → final 0.97216 (+0.00007, a few flipped rows on the 20% public slice). Same LB-flip clout family as fachri00 ridge-flips / zoli800 QSO-patch — public-slice tuning, NOT private-robust, NOT a CV lever. Its "ambiguous rows can't be stacked away" thesis = our saturation finding restated. zoli800 0.97209-clean-final = iteration of the external-SDSS17 GALAXY→QSO flip sub (coord-match falsified in node_0083). Every voter in-bank; binary-chain built (n049, stack-neutral). VERDICT: public ecosystem exhausted again; no new signal source. CLI note: `competitions discussions`/per-comp topic listing unavailable in kaggle 2.2.1 — only `topics list/show`.

## 2026-06-17T18:27Z — intel scan #3 (autonomous): one borderline new feature (omadon spatial kNN), rest exhausted
voteCount board UNCHANGED (same zoo). 3 fresh kernels pulled (refs/scan_2026-06-17b/). 2 = nothing (vladstud error-geometry EDA = our decorrelation-wall hard zone restated; omkarkashid EDA runs on the synthetic-fallback branch with spectral_type/galaxy_population cols that DON'T exist in real comp data → fabricated quirk). 1 BORDERLINE: omadon/s6e6-spatial-knn-class-fraction-features (2026-06-17) — fold-safe kNN CLASS-FRACTION on (alpha,delta): per row, fraction of K sky-neighbours in each class, KD-tree on TRAIN-FOLD only, self-excluded, K=10/50/200. Claims +0.003 BA. NEW vs our bank (n013 used label-FREE kNN DISTANCE, not neighbour-label fractions = a spatial target-encode). SKEPTIC PRIOR (n083: class ~independent of coords, ~50% match < base rate; n060: generator REUSES alpha/delta values 32-38% → kNN reads coord-reuse as in-fold structure, overfits; n013 leak-safe positional regressed −6σ/5 folds) → almost certainly a coordinate-value-reuse MIRAGE that overfits in-fold + shakes on private. omadon validates with ONE 5-fold + 1 base, no holdout/LB. WORTH one cheap FALSIFICATION node (fit_in_fold class-fraction block, judge by pred_diagnostic holdout + bootstrap, LB-probe ONLY if it survives the holdout). Expect death like n013. → node_0125 (falsification). danushkumar "ridge-flip refinement" = known LB-flip clout family (fachri00/zoli800). discussions CLI still unavailable in kaggle 2.2.1.
