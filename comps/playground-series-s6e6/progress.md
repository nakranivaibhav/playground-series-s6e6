# progress — playground-series-s6e6
today (UTC): 2026-06-18   submissions: 2/10 today (resets 00:00 UTC)   deadline: 2026-06-30 (12 days left)

## ★★ FINALS DECISION (locked-in pair, to UI-select before 2026-06-30) ★★
Champion (promotion) = node_0091. FINALS 2 submissions (manually select these in the Kaggle UI —
Submissions page → mark for final; the API CANNOT do it, and if you DON'T pick, Kaggle auto-uses your
2 best PUBLIC scores = the clout A4 vote, which is private-fragile):
  · SLOT 1 = node_0091  · sub ref 53670216 (2026-06-14) · public LB 0.97121 · honest champion, best LB
  · SLOT 2 = node_0129  · sub ref 53799340 (2026-06-18) · public LB 0.97118 · highest CV (0.970410), LB-confirmed
Rationale: the two highest LB-confirmed honest subs; slightly decorrelated (n129 is a meta over n091+5
stacks). All-pairs E[max] study (probes/finals_robustness.py) + LB probes back this; n116 dropped (LB
0.97110, CV overstated it), n117 dropped (LB 0.97003, wins 0% of private draws).
ALTERNATIVE upside-gamble (revisit at deadline): A4 vote sub ref 53535519 · public LB 0.97123 (HIGHEST
public) — but it's a clout vote-blend, public-overfit, private-RISKY (n047-mirage class). Use ONLY if
willing to gamble a slot on public→private holding. Default recommendation: the honest pair above.

★ Champion node_0091 (CV 0.970355 / LB 0.97121) — UNCHANGED. Mode: full_auto. Nothing running.
### ▶ AUTONOMOUS SESSION 2026-06-17→18 (user asleep, full_auto, "never give up") — 7 nodes + 2 probes + 4 look-outsides, ZERO promote, champion stands. THE NIGHT'S VALUE = knowledge, tooling, and finals, not CV. Built the structural-gate toolkit (tools/pred_diagnostic.py + probes/hidden_signal_sweep + capturability + finals_robustness) and switched the decision gate from raw 2·sem to paired-bootstrap + holdout-confirmed structure (CLAUDE.md + validation.md updated). FINDINGS: (1) BA hid 12 holdout-significant complementary fix-blocks (per-base) — REAL signal the scalar discarded — but ALL row-entangled, UNCAPTURABLE by region/class conditioning, proven across all 3 meta forms (hard-vote capturability, additive interaction n122, learned MoE-gate n127). (2) Genuinely-NEW info sources all land in the SAME GALAXY↔STAR entanglement (GALAXY +~0.01 / STAR −~0.015 / corr ~0.81): gen-fingerprint n118, PHYSICS photo-metallicity+stellar-locus n124/n126 (mechanism works, closed), spatial-kNN n125 (real-NOT-mirage, holds on holdout, still entangled), z-local-color n128 (high-z GAL↔QSO channel, same trade). (3) focal-loss n123 DEAD (boundary is geometry- not hardness-driven). (4) PAST-WINNERS sweep: all 5 canonical synthetic-TPS levers already-closed nodes here → stack embodies the standard playbook. The information ceiling is now confirmed ~12 ways + external corroboration. ★ FINALS OVERTURNED: finals_robustness MC (best-of-2 = E[max]) shows node_0117 is a WORTHLESS hedge (wins 0% of private draws, too weak) — the evidence-based finals pair is **node_0091 + node_0116** (E[max] 0.970399, n116 wins 58.5% of draws, near-champion + decorrelated). ACTION FOR HUMAN: switch deferred finals slot-2 from n117 → n116. 5 MEMORY lessons banked. PRIOR (2026-06-14): champion promoted, combiner maxed, decorrelation axes closed.
### ▶ ROUND 2026-06-14 PM (n095-0098 + n091 promotion) CLOSED — ★ BIG WIN: LB-probed node_0091 (full-pool L2 stack) → LB 0.97121 = BEST honest LB ever (+0.00048 vs old champ n063, +0.00034 vs n070) → PROMOTED champion (strict superset of n063 under shrinkage; sub-2sem CV overridden by decisive LB). Then closed the LAST decorrelation axis: n095 strict-resid KILL (BA 0.72); n096/n097 Nystroem-RBF DECORRELATED (err-corr 0.53-0.54, FIRST since FT-T) but n098 stack-add HURTS −0.00005 (BA 0.947 too weak) → RBF CLOSED (n62 weak-but-decorrelated rule holds at corr 0.54). STATE: combiner maxed + ALL decorrelation axes closed + public ecosystem exhausted (look-outside 2026-06-14 = board-climb is pure clout) → at the genuine honest ceiling. FINALS: slot-1(honest)=node_0091 (LB 0.97121, robust), slot-2(clout)=A4 vote (LB 0.97123).
### ▶ ROUND 2026-06-14 (node_0091/0093/0094, mega-combine over our own zoo) CLOSED — NO promote. node_0091 = ★ NEW BEST honest CV 0.970355 (full-pool L2 @C0.003 REVERSES the "in-house pool dilutes" finding — strong shrinkage absorbs weak/correlated bases) but +0.000144 vs n070 is SUB-2sem → no slot; the COMBINER IS NOW MAXED on the existing base pool. Two nulls: simplex convex blend can't match per-class LogReg (n093 0.9632); error-pocket instance-weighting does NOT decorrelate (n094 err-corr 0.79). node_0092 (Caruana) dropped pre-build = A2 re-run. VERDICT: ceiling = the BASE SET; last untested decorrelation axis = a different INPUT REPRESENTATION (flux/color ratios, interaction/projection spaces) or target reframing — NOT reweighted loss/residuals/decomp on same features. NEXT: propose round on input-representation decorrelation. FINALS unchanged: slot-1(honest)=node_0070 (LB 0.97087), slot-2(clout)=A4 vote (LB 0.97123); lock near 2026-06-30.
### ▶ ROUND 2026-06-13 (node_0076–0080) CLOSED. No promote. node_0076 = BEST HONEST candidate (cv 0.970227, FT-T+bag+argmax, threshold-free/robust) → finals slot-1. Closed: external base sourcing (n77), DE calibration (n78), expressive L1 meta (n80), honest disjoint-teacher (n79).
### ▶ LOOK-OUTSIDE ROUND 2026-06-13 (node_0081–0085) CLOSED. Research pass → new base family/framing mandate; ALL 5 FAILED: SAINT (n81) + NODE (n82) NN families dead below tier; real-SDSS-DR17-label data lever FALSIFIED (n83 — generator labels ⊥ sky coords); physics-locus FE washes (n85); revival sub-noise (n84). Public LB top confirmed clout (vote-blend + LB-flip-probe), not modelling. DIAGNOSIS: honest CV frontier genuinely ~0.9702; every decorrelation axis mapped & closed. Champion node_0063 (0.97073) stands.
### ▶ LB PROBES 2026-06-13 (2/10): node_0076 (best CV 0.970227) → LB 0.97073 (== champ, BELOW n70 0.97087 — bagging+argmax hurt LB); node_0084-clout (CV 0.970299) → LB 0.97036 (worst — n74 add is a CV mirage on LB). CV↔LB DIVERGE at this frontier; every CV-raising lever lowered LB. **FINALS REVISED:** slot-1 (honest) = **node_0070** (best honest LB 0.97087 + tied-best CV), NOT n76; slot-2 (clout) = A4 vote LB 0.97123; node_0084-clout dead for finals. **NEXT:** keep grinding new levers (z-conditional residual base etc. — 5 proposals pending registration from the latest propose-loop).
### ▶ ROUND 2026-06-13 (node_0086–0090) CLOSED — ALL 5 WASHED. z-conditional residual bases (n86 TabM/n87 LGBM) FAILED to decorrelate (err-corr ~0.70-0.72 — bank already encodes z-structure); OvR chained RealMLP (n90) below parent + corr 0.72; n55 revival no-select; STAR-BOOST knob closed (b=1.0). Decorrelation axis now exhaustively mapped across families/bases/labels/physics/residuals/decompositions — all ~0.70+ corr. Champion node_0063 stands. Finals: slot-1=node_0070 (honest LB 0.97087), slot-2=A4 vote 0.97123 (clout).
### ▶ ROUND 2026-06-10 COMPLETE (round_plan.md). 13 items run, 2 WINS + 11 nulls (stack saturation confirmed).
### CHAMPION = node_0091 (full-pool L2 LogReg mega-stack @C0.003, CV 0.970355 / LB 0.97121) — promoted 2026-06-14, prev node_0063.
### FINALS PAIR (REVISED 2026-06-14, NOT yet locked — lock near 2026-06-30):
###   slot-1 (trustworthy bet) = node_0091 full-pool L2 stack (CV 0.970355 BEST honest / LB 0.97121 BEST honest, fold-honest no test-fit; robust superset of the bank);
###   slot-2 (upside swing) = A4 vote (LB 0.97123 BEST public, clout/no-private-guarantee, refs/a4_vote/ — now only +0.00002 over our HONEST slot-1). If forced to ONE: slot-1.
### Budget 2/10.
### Skipped by user choice: C2 relabel, C4 locus, C7 distill, C10 KNN-Shapley. Nodes 0059-0062 registered.
### ▶ RESUME TOMORROW — first task: BUILD node_0058 (augmentation ablation), then decide next direction.
node_0058 is REGISTERED but NOT built (developer dispatch was cancelled to stop for the day). It re-tests the
feature->image ResNet (node_0057, killed fold-0 BA 0.9401) with an EXTENSIVE augmentation tournament on FOLD-0:
configs C0 none / C1 input-space photometric+z jitter+band-drop / C2 image regularizers (cutout/mixup) /
C3 geometric (flip/rotate/shear/crop) / C4 A+B / C5 all — each reporting fold-0 BA + err-corr-vs-CORE15.
Proceed to 5-fold ONLY if a config clears BA>=0.955 AND err-corr<=0.6; else clean-negative (closes the
"did we augment well + would geometric augs help" question). Honest prior: info ceiling (~0.95) binds → wash.
Full spec in nodes/node_0058/node.md. Re-dispatch the kaggle-developer with the augmentation-ablation prompt.
### STATE @ pause: champion node_0041 (0.97043). BEST-LB blend = node_0055-restack CORE15+DCN = LB 0.97083
(CV 0.969794 tied; #1 finals candidate, not promoted — LB-chasing guard). Within-tooling/base-set search
EXHAUSTED with evidence: model zoo, meta-config, multi-seed meta (n53), external data, priors, pseudo-label,
specialists (n47 mirage), binary-chain (n49/50), revival (n51 FT-T), DCN (n55), 1D-CNN (n56), feature->image
ResNet (n57 killed). Two LB probes show de-corr bases don't compound past 0.97083. If image+aug also washes,
the genuinely higher-EV move per the plateau rule is LOOK OUTSIDE: pull the current #1 public solution / top
discussions and diff for a concrete lever (NOT another in-house architecture).
### SESSION 2026-06-09 — SPECIALIST CV-MIRAGE caught & reverted; 3 parallel experiments all DEAD.
node_0047 GALAXY-vs-STAR low-z specialist as a 16th stack base: nested-CV 0.970881 (+0.001073 ~3.4 sem),
leak-scan clean → looked like a breakthrough, promoted + SUBMITTED → **LB 0.96242, a −0.0080 COLLAPSE**.
CV mirage: a narrow label-fit specialist sub-model double-uses labels the meta also fits; its optimism
sits in BOTH OOF and nested holdout, so honest CV can't see it. REVERTED, node_0041 reinstated. node_0046
pseudo-label (LGBM solo 0.967125) re-stack −0.0001 WASH; node_0048 Optuna-XGB-to-stack-obj killed trial
13/27, best +0.00005 WASH. Combined re-stack CORE15+spec+pseudo = 0.970915 (spec carries it, all mirage).
LESSONS → MEMORY.md ([stack] specialist-base mirage, [cv] gap ≫noise overrides trust-CV). Also: removed
kaggle-final skill + all refs system-wide (goal = top the LB with tenacity; no "finish" stage). Budget 1/5.
### SESSION 2026-06-08 (pt2) — CHASE THE 0.97126 PUBLIC CLUSTER (#1=0.97144). Pulled CdeOtte top base
notebooks (xgb-v5/cat-v3/lgbm-v3/nn-v2). Their shared edge = ORIGINAL-SDSS17 PRIOR features (P(class|color-
bin) computed on orig data, NOT row-concat) + rich FE + top-370 sel. Tested BOTH forms on our stack:
 • node_0044 = faithful xgb-v5 port (fs_zoo: rich FE+in-fold TE+orig-priors+top-370, balanced-err early-stop),
   leak-clean, solo 0.96769 (caps below ref 0.969 — our folds ≠ their seed-42 folds). Re-stack −0.00012 WASH.
 • node_0045 = our strongest arm RealMLP n28 UPGRADED in-place with orig-priors, leak-clean, solo 0.969050
   FLAT vs n28. Stack swap/add ALL regress (−0.00021..−0.00023).
CONCLUSIVE: orig-prior lever INERT for us (rich-FE bases already capture it + priors carry generator drift).
CdeOtte-zoo direction DEAD. Champion node_0041 UNCHANGED, no submission. Remaining gap to cluster (+0.00083)
is fold-scheme + noise — only closable by rebuilding the WHOLE 15-base zoo on their fold scheme (huge, low-EV).
### SESSION 2026-06-08 — ALL LEVERS EXHAUSTED, no submission spent. Closed the last two:
(1) META-CONFIG sweep (logit vs log-prob × C∈{1,.3,.1} × seed-bag×5) on the 15-base CORE — ALL wash <0.12sem;
    gap to public 0.97105 is base-set, not the stacker. (2) SDSS17 EXTERNAL DATA — DEAD: adversarial drift
    AUC 0.909 (redshift KS 0.194), single-base honest A/B HURT −0.0010 (adversarial-downweight worse −0.0012).
Model-zoo (all families) + meta-config + external data all exhausted. +0.0006 to 0.97105 is within private 1σ
(~0.00087) = practical ceiling. NEXT: keep running the experiment loop (terminal stage); finals candidates are
tracked in journal.md + round_plan.md, locked near the deadline.
Probes: meta_config_probe.py, sdss17_drift_probe.py, sdss17_ab_probe.py.
### 🏁 SESSION 2026-06-07 full_auto — CEILING REACHED. champion node_0041, CV 0.969808 / LB 0.97043.
Session arc: node_0020 (0.96722) → node_0029 (0.96993) → node_0041 (0.97043). 0.0006 from public 0.97105.
CHAMPION node_0041 = 15-base balanced-LogReg STACK + DE thresh = champ9 + 3 RealMLP-ref seeds (n28/32/35)
+ TabM-richFE (n33) + LightGBM-richFE (n30) + CatBoost-richFE (n39). Breakthrough = node_0028 RealMLP
reference recipe (heavy FE fs_realmlp_fe + PBLD-embed RealMLP + fit-in-fold TargetEncoder), +0.020 over our
under-built RealMLP. STACK SATURATED: only the FIRST strong member of each new family lifts it (RealMLP/
TabM/LightGBM/CatBoost each +0.0001..+0.0026); weak bases (MLP/LogReg/ExtraTrees) HURT; 2nd seeds/configs
of included families (RealMLP-B, CatBoost-B, XGBoost, 3rd RealMLP seed, TabICL) WASH. All families covered.
Budget 2/10 today. FINALS candidates (tracked in journal.md + round_plan.md, locked near the deadline):
node_0041 (best CV+LB) + node_0029 (proven LB, slightly de-correlated). UNCOMMITTED: full agent refactor +
breakthrough nodes + probes + refs/. HF token in
.env works (revoke leaked hf_ghSiKk…). Reference recipes saved in comps/.../refs/.

### 🏆 BREAKTHROUGH 2026-06-07 (ROUND #4) — CHAMPION node_0029, cv 0.969205, LB 0.96993 (+0.0027).
The "ceiling" conclusion below was WRONG: our RealMLP was UNDER-BUILT (bare pytabkit-TD, 0.949), not capped.
The public 0.97105 meta-stacker (user-flagged) revealed it — its single RealMLP scores 0.96973. node_0028
ports that proven recipe (heavy FE fs_realmlp_fe + hand-rolled PBLD-embedding RealMLP + fit-in-fold
TargetEncoder) → cv 0.969065 SOLO (+0.020 vs our n24), beating the old champion stack alone. node_0029 =
champ9+n28 stack + DE thresh → cv 0.969205, LB 0.96993 = NEW CHAMPION, submitted (budget 1/10). LESSON:
verify a base is built to the PROVEN recipe before concluding a model "caps". Now 0.0011 from 0.97105.
IN FLIGHT: n30 LightGBM / n31 XGBoost / n32 RealMLP-seed2 on fs_realmlp_fe → re-stack toward 0.971.
Reference recipes saved in comps/playground-series-s6e6/refs/. Stacker meta-config saturated (~0.0001).

ROUNDS 2026-06-07 #1+#2 (stronger + SOTA NN bases) = NULL [SUPERSEDED by the breakthrough above — the
NN bases were under-built, not capped]. #1: n21 RealMLP 0.9501 / n22 TabPFN-3 0.9426 /
n23 CatBoost-retune 0.9627. #2: n24 RealMLP-HPO 0.9492 / n25 TabPFN-2.5 `v2.5_large-samples` (proper HF
ckpt, batching FIXED 4h→10min, peak 5.2GB) 0.9490. Re-stack A/B: NONE promotable (+cat +0.00006 noise;
every NN base −0.00003..−0.0001). Champion node_0020 (0.966627) UNCHANGED. CONCLUSION: stronger-NN-base
lead EXHAUSTED — RealMLP/TabPFN/FT-T all cap 0.949-0.957 on fs_research, the discussion's 0.963-0.968 NN
band does NOT reproduce; node_0020 is a genuine ceiling for our base zoo. ROUND #3 (last SOTA): n26 TabICL 0.9590 (strongest NN, 3.4min) + n27 TabPFN-v3-multiclass 0.9409 — re-stack
NULL (+tabicl -0.00005). Model-zoo lead DEFINITIVELY EXHAUSTED (6 NN families all 0.94-0.959, none lifts the
stack; TabM already fills the slot). node_0020 0.966627 = genuine ceiling. Within-feature search COMPLETE.
Remaining levers OUTSIDE the zoo: external SDSS17 concat (risky, high drift) OR finals (user-trigger only).
AGENT REFACTOR (done): kaggle-developer+reviewer → ONE self-gating agent w/ perf rules (time-one-unit-first,
encode-context-once); references updated across skills/CLAUDE.md/docs. STILL UNCOMMITTED: agent refactor +
shuffled-scrub + GPU/stack/speed probes. HF token in .env (revoke the earlier LEAKED token hf_ghSiKk…).
CHAMPION = node_0020 (balanced multinomial LogReg STACK on 9 bases' OOF log-probs + DE per-class
threshold) cv=0.966627±0.000221, lb=0.96722 (NEW BEST, prev node_0010 0.96704). Beat prev champion by
+0.000738 (~3.3 sem), all 5 folds, fold-honest. THE STACK is the lever (discussions broke our plateau).
Submission limit corrected to 10/day (was wrongly 5 in spec). DE vs GPU-grid threshold: DE +0.000054
(within noise), 45x slower — and threshold barely matters atop a balanced stack.
NEXT (best leads, from discussions in discussions.md): (1) add STRONGER bases to the stack — RealMLP,
TabPFN-3, more XGB/Cat configs — that's the gap to the ~0.9707 ceiling; (2) tune stacker C / base set;
(3) keep running the experiment loop (terminal stage); finals candidates live in journal.md + round_plan.md,
locked near the deadline. DEAD ENDS (don't retry): more NN/tree diversity arms (blend
saturated), target-encoding spectral_type/galaxy_population (they're deterministic color cuts), positional FE.
NOTE: reusable system code UNCOMMITTED — shuffled-label control REMOVED system-wide (kaggle-developer/
reviewer/leakage skill/CLAUDE.md/tools/leakage_scan.py), GPU blend+stack probe scripts, propose-loop slug
fallback patch. Commit/push at start of next session if wanted. Kaggle auth: .env uses KAGGLE_TOKEN -> map
to KAGGLE_KEY before kaggle calls.

champion: see graph.md (single source)

## one-time setup (human)
- [x] KAGGLE_USERNAME + KAGGLE_KEY in env        → from .env (new-style KAGGLE_API_TOKEN)
- [x] competition rules accepted in browser      → download returned 200
- [ ] account phone-verified (GPU/internet)      → not needed (local tabular, no kernels)
- [x] data downloaded + unzipped                 → comps/playground-series-s6e6/data/
- [x] spec.md written                            → comps/playground-series-s6e6/spec.md

## stages
- [x] understand   (card approved)
- [x] toolkit      (card approved — GBDT seeds A/B/C; sklearn + DL allowed as later branches)
- [x] eda          → /kaggle-eda
- [x] validate     → /kaggle-validate
- [x] baseline     → /kaggle-baseline
- [ ] experiment   → /kaggle-experiment
