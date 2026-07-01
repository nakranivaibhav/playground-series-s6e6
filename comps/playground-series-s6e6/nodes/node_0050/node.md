---
id: node_0050
desc: symmetric one-vs-rest 3-binary heads
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: gbdt
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.968134
sem: 0.000128
folds: [0.968189, 0.968482, 0.967738, 0.967974, 0.968287]
baseline_cv: 0.969808
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-09
decided: null
tags: [one-vs-rest, symmetric, re-stack-candidate, controlled-ab, lb-gate]
---

## plan
built on:   root (fresh draft on fs_realmlp_fe). Reuses the stateless RealMLP
            reference FE feature-set (fs_realmlp_fe) byte-identical; reuses the
            frozen folds.json — no refit across folds. Standalone reference bar
            is node_0030 (LightGBM richFE, cv 0.966952); champion 15-base stack
            is node_0041 (cv 0.969808). NEW base for the re-stack, and the
            controlled symmetric companion/A-B to node_0049's asymmetric chain.
change:     Replace the single 3-class GBDT head with a SYMMETRIC one-vs-rest set
            of THREE independent binary LightGBM heads on fs_realmlp_fe: GALAXY
            vs rest, QSO vs rest, STAR vs rest. Each trained train-fold-only; the
            three raw binary scores are normalized (per-row sum-to-1) into 3-class
            OOF + test probs emitted as a stack base. Identical decomposition idea
            as node_0049 but symmetric (no ordering/conditioning) — the controlled
            A/B isolating whether the asymmetric chain's ordering helps.
hypothesis: One-vs-rest gives each class its own decision surface and a different
            error structure than the joint softmax head, adding de-correlated
            signal to the meta-stack; symmetry avoids the chain's error-propagation.
target:     Balanced Accuracy (maximize) · solo beats parent bar if cv > 0.966952
            (node_0030); the real test is the re-stack A/B vs node_0041 (0.969808),
            also compared head-to-head against node_0049. LB-GATE before promotion
            — do NOT promote on CV alone even at >2σ (node_0047 was a CV mirage:
            cv 0.970881 but LB crashed −0.0080).

## notes
### Run results (2026-06-09)

Timing: fold0=82.8s, projected 5-fold=414s (~7min); actual total=763s (12.7min including DE+re-stack).

**STANDALONE (symmetric OvR, DE threshold):**
- cv = 0.968134 ± 0.000128
- per-fold: [0.968189, 0.968482, 0.967738, 0.967974, 0.968287]
- raw-argmax (no DE): 0.960640 — DE heavily downscales GALAXY (~0.21) and QSO (~0.68), STAR anchor 1.0
- vs node_0030 (parent LightGBM): 0.966952 → +0.001182 (beats parent)
- vs node_0049 (asymmetric chain): 0.968020 → +0.000114 (symmetric marginally beats chain, within 1 sem)

**RE-STACK CORE15 + node_0050 (16 bases):**
- cv = 0.969781 ± 0.000292
- per-fold: [0.970851, 0.969145, 0.969506, 0.969520, 0.969885]
- vs champion node_0041 (0.969808): −0.000027 (essentially neutral, within noise)

**RE-STACK CORE15 + node_0049 + node_0050 (17 bases — both chain framings):**
- cv = 0.969665 ± 0.000292
- per-fold: [0.970666, 0.969008, 0.969194, 0.969585, 0.969871]
- Adding both together is WORSE than CORE15+n50 alone → n49 and n50 are not jointly additive

**Interpretation:**
- Symmetric OvR (0.968134) ≈ asymmetric chain (0.968020) — ordering/conditioning in the chain gives no meaningful benefit; both improve on the 3-class parent via binary decomposition + DE.
- As a stack base: CORE15+n50 (0.969781) essentially tied with champion (0.969808); LB-gate required before promotion.
- The standalone and re-stack results suggest the OvR framing adds de-correlated signal but the meta-learner cannot extract more than CORE15 already captures.
