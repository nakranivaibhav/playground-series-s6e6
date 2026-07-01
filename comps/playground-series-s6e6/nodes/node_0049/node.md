---
id: node_0049
desc: asymmetric binary chain GALAXY-then-QSO/STAR
op: draft
parents: [root]
uses_data: [fs_realmlp_fe]
family: gbdt
status: valid
stage: reviewed
metric: Balanced Accuracy Score
direction: maximize
cv: 0.968020
sem: 0.000159
folds: [0.967773, 0.968434, 0.968308, 0.968117, 0.967489]
baseline_cv: 0.969808
gates: {schema_ok: true, oof_full: true, no_nan: true, dist_sane: true,
        leak_clean: true, cv_too_good: false, passed: true}
gate_note: null
leak: clean
lb: null
submitted: null
created: 2026-06-09
decided: null
tags: [binary-chain, asymmetric, re-stack-candidate, lb-gate]
---

## plan
built on:   root (fresh draft on fs_realmlp_fe). Reuses the stateless RealMLP
            reference FE feature-set (fs_realmlp_fe) byte-identical; reuses the
            frozen folds.json — no refit across folds. Standalone reference bar
            is node_0030 (LightGBM richFE, cv 0.966952); champion 15-base stack
            is node_0041 (cv 0.969808). This is a NEW base for the re-stack.
change:     Replace the single 3-class GBDT head with an ASYMMETRIC binary CHAIN
            of two GBDTs on fs_realmlp_fe: (1) GALAXY vs not-GALAXY, then (2) on
            the not-GALAXY rows only, QSO vs STAR. Each stage is its own
            in-fold-trained LightGBM classifier; final 3-class OOF probs are
            composed P(GALAXY)=p1, P(QSO)=(1-p1)·p2, P(STAR)=(1-p1)·(1-p2). Both
            stages fit train-fold-only; emit full OOF + test probs as a stack base.
hypothesis: The dominant GALAXY class is cleanly separable; isolating it first lets
            the second head specialize on the harder QSO/STAR boundary, yielding a
            de-correlated error structure the meta-stack can exploit.
target:     Balanced Accuracy (maximize) · solo beats parent bar if cv > 0.966952
            (node_0030); the real test is the re-stack A/B vs node_0041 (0.969808).
            LB-GATE before promotion — do NOT promote on CV alone even at >2σ
            (node_0047 was a CV mirage: cv 0.970881 but LB crashed −0.0080).

## notes
re-stack A/B: CORE15+n49 = 0.969765 vs champ 0.969808
  Per-fold re-stack: [0.970938, 0.969136, 0.969443, 0.969431, 0.969878]  sem=0.000316
  Delta vs champ: -0.000043 — within fold noise (1 sem = 0.000316). Neutral graft.
  STANDALONE composed-OOF DE-threshold BA: 0.968020 +/- 0.000159
    Per-fold DE: [0.967773, 0.968434, 0.968308, 0.968117, 0.967489]
  STANDALONE raw-argmax BA: 0.960537 +/- 0.000363
  Timing: fold0=44.1s, 5-fold chain+DE+restack total=6.8min.
  LB-GATE before any promotion (node_0047 mirage precedent).
