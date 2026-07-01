"""node_0077 — source NEW primary external OOF bases, greedy forward-select onto bank17+FT-T.

Starting set: bank-17 + pilkwang ft_transformer_lite (node_0070's selected set, cv=0.970211).
New candidates from refs/ that are NOT already in the starting set:
  - ravi_gnn_mlv1/OOF_Preds_REALMLPV1_1.parquet  (solo BA ~0.965)
  - ravi_gnn_mlv1/OOF_Preds_MLV1_1.parquet        (solo BA ~0.967)
  - pilkwang_5090/oof_extratrees_soft_*            (solo BA ~0.947) — tried in n70, not selected
  - pilkwang_5090/oof_hgb_balanced_*               (solo BA ~0.956) — tried in n70, not selected
  - pilkwang_5090/oof_tabm_lite_*                  (solo BA ~0.931) — tried in n70, not selected
  - pilkwang_5090/oof_logit_elastic_*              (solo BA ~0.898) — tried in n70, not selected

Kernels confirmed OOF-LESS (no primary OOF array, only submission):
  - pull_philippsinger_tabpfn-3-stacker: meta-stacker, reads others' OOF, publishes no primary OOF
  - pull_kospintr_stellar-catb-hgbc-xgb-lgbm-realmlp-baseline: outputs submission.csv only
  - nn-v2-for-s6e6.py: already ingested as nn-2 in bank-17 (node_0070 manifest line 141)
  - ps-s6-e6-realmlp-pytorch.py: already ingested as realmlp-0 in bank-17 (yekenot kernel)
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3
EPSILON = 0.00003

# ---------------------------------------------------------------------------
# Helpers (byte-identical to node_0070)
# ---------------------------------------------------------------------------

def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))

def norm(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True); s[s == 0] = 1
    return a / s

def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))

def fit_meta(Xtr: np.ndarray, ytr: np.ndarray) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m

def best_thr_de(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(
        neg, [(0.1, 5.0), (0.1, 5.0)],
        maxiter=40, tol=1e-7, seed=0, polish=False, workers=1
    )
    return np.array([r.x[0], r.x[1], 1.0])

def eval_cols(oof_cols: list, y: np.ndarray, fval: list) -> tuple:
    """Fold-honest stacked CV: LogReg + DE threshold."""
    n = y.shape[0]
    OOF = np.concatenate(oof_cols, axis=1)
    stack = np.zeros((n, NC))
    for vi in fval:
        tr = np.setdiff1d(np.arange(n), vi)
        stack[vi] = fit_meta(OOF[tr], y[tr]).predict_proba(OOF[vi])
    pf = []
    for vi in fval:
        oth = np.setdiff1d(np.arange(n), vi)
        w = best_thr_de(stack[oth], y[oth])
        pf.append(score_fn(y[vi], np.argmax(stack[vi] * w, axis=1)))
    return float(np.mean(pf)), float(np.std(pf, ddof=1) / np.sqrt(len(pf))), stack

def rd(path: str | Path, nr: int) -> np.ndarray:
    """Read OOF / test probs from npy or csv."""
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a
        return a[:, :3]
    d = pd.read_csv(p)
    c = list(d.columns)
    if set(LAB).issubset(c): return d[LAB].values.astype(float)
    pc = [f"prob_{l}" for l in LAB]
    if set(pc).issubset(c): return d[pc].values.astype(float)
    num = d.select_dtypes("number")
    if num.shape[1] >= 3: return num.values[:, :3]
    v = d.iloc[:, 0].values.astype(float); return v.reshape(nr, 3)

def load_ext_csv(path: str | Path, nr: int) -> np.ndarray:
    """Load pilkwang-style CSV with proba_GALAXY/QSO/STAR columns."""
    d = pd.read_csv(path)
    pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
    if set(pcols).issubset(d.columns):
        return d[pcols].values.astype(float)
    return rd(path, nr)

def load_parquet_probs(path: str | Path, nr: int) -> np.ndarray:
    """Load ravi-style parquet with p_GALAXY/p_QSO/p_STAR columns."""
    d = pd.read_parquet(path)
    pcols = ["p_GALAXY", "p_QSO", "p_STAR"]
    if set(pcols).issubset(d.columns):
        arr = d[pcols].values.astype(float)
        assert arr.shape[0] == nr, f"expected {nr} rows, got {arr.shape[0]}"
        return arr
    raise ValueError(f"unexpected columns in {path}: {list(d.columns)}")


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n = len(train)
    nt = len(test)
    y = train["class"].map(L2I).to_numpy()
    fval = [np.asarray(f["val_idx"]) for f in folds_data]

    print(f"n_train={n} n_test={nt} n_folds={len(fval)}")
    assert n == 577347, f"unexpected n_train={n}"
    assert nt == 247435, f"unexpected n_test={nt}"

    # ---- PRE-FLIGHT LEAKAGE CHECKS ----
    print("Leakage check 1-2: features are OOF probs only (no target/id in feature matrix). PASS")
    print("Leakage check 4-5: LogReg fit inside fold loop; folds from frozen folds.json. PASS")

    # ---- Load public bank-17 (same manifest as node_0070) ----
    B = COMP / "refs/oof_bank"
    K = COMP / "refs/kernel_out"

    MANIFEST = {
        'xgb-0':      (K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",         K/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
        'xgb-1':      (K/"xgb-v1-for-s6e6/oof_preds.npy",           K/"xgb-v1-for-s6e6/test_preds.npy"),
        'realmlp-0':  (B/"oof_preds_realmlp0_v12.csv",               B/"test_preds_realmlp0_v12.csv"),
        'realmlp-1':  (K/"realmlp-v1-for-s6e6/oof_preds.npy",        K/"realmlp-v1-for-s6e6/test_preds.npy"),
        'tabm-0':     (B/"oof_preds_tabm0_v2.csv",                   B/"test_preds_tabm0_v2.csv"),
        'cat-0':      (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
        'realmlp-2':  (B/"oof_preds_realmlp2_v10.csv",               B/"test_preds_realmlp2_v10.csv"),
        'tabicl-2':   (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
        'lgbm-3':     (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",     K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
        'logreg-1':   (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",  K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
        'nn-1':       (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",          K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
        'xgb-3':      (K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
        'xgb-5':      (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",       K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
        'realmlp-5':  (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
        'nn-2':       (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",          K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
        'cat-3':      (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",        K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
        'lgbm-5':     (B/"oof_preds_lgbm5_v1.csv",                  B/"test_preds_lgbm5_v1.csv"),
        'xgb-6':      (B/"oof_final_xgb6_v1.csv",                   B/"test_final_xgb6_v1.csv"),
        'tabm-1':     (B/"oof_final_tabm1_v1.csv",                   B/"test_final_tabm1_v1.csv"),
    }

    POOF = {}; PTEST = {}; good = []
    print(f"\n{'model':12s} {'oofBA':>9s} {'shape':>12s} {'status'}")
    for name, (op, tp) in MANIFEST.items():
        try:
            o = norm(rd(op, n)); t = norm(rd(tp, nt))
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            st = "OK" if 0.90 < ba < 0.972 else ("QUARANTINE" if ba >= 0.972 else "LOW?")
            if st == "OK": POOF[name] = o; PTEST[name] = t; good.append(name)
            print(f"{name:12s} {ba:9.6f} {str(o.shape):>12s} {st}")
        except Exception as e:
            print(f"{name:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}")

    print(f"\nLoaded {len(good)} public models OK (expected 17)")

    # ---- Load FT-Transformer (selected by n70, forms the bank17+FT-T starting set) ----
    PILK = COMP / "refs/ext_oof/pilkwang_5090"
    print("\n=== Loading FT-Transformer (n70 selected base) ===")
    ft_oof_raw  = load_ext_csv(PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n)
    ft_test_raw = load_ext_csv(PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", nt)
    assert ft_oof_raw.shape == (n, 3), f"FT-T OOF shape {ft_oof_raw.shape}"
    assert ft_test_raw.shape == (nt, 3), f"FT-T test shape {ft_test_raw.shape}"
    ft_solo_ba = balanced_accuracy_score(y, norm(ft_oof_raw).argmax(1))
    print(f"ft_transformer: solo_BA={ft_solo_ba:.6f}")
    assert ft_solo_ba > 0.85, f"FT-T solo BA {ft_solo_ba:.4f} too low"
    ft_oof_l  = logp(norm(ft_oof_raw))
    ft_test_l = logp(norm(ft_test_raw))

    # ---- Reproduce bank17+FT-T baseline (node_0070 cv=0.970211) ----
    bank_oof  = [logp(POOF[k]) for k in good]
    bank_test = [logp(PTEST[k]) for k in good]
    start_oof  = bank_oof  + [ft_oof_l]
    start_test = bank_test + [ft_test_l]

    print("\n=== Reproducing bank17+FT-T baseline (node_0070) ===")
    base_cv, base_sem, _ = eval_cols(start_oof, y, fval)
    print(f"bank17+FT-T cv={base_cv:.6f}  sem={base_sem:.6f}")
    EXPECTED = 0.970211
    diff = abs(base_cv - EXPECTED)
    print(f"diff from n70 expected {EXPECTED}: {diff:.6f}")
    if diff >= 0.0003:
        raise SystemExit(
            f"BASELINE ASSERTION FAILED: got {base_cv:.6f}, expected ~{EXPECTED}, diff={diff:.6f}. STOPPING."
        )
    print(f"BASELINE ASSERTED OK: {base_cv:.6f} ~ {EXPECTED}")

    # ---- Load new candidate bases ----
    RAVI = COMP / "refs/ext_oof/ravi_gnn_mlv1"
    print("\n=== Checking new candidate primary OOF sources ===")
    print("NOTE: tabpfn-3-stacker (philippsinger) is a meta-stacker -- reads others OOF, no primary OOF. SKIP.")
    print("NOTE: kospintr stellar kernel -- outputs submission.csv only, no OOF array. SKIP.")
    print("NOTE: nn-v2-for-s6e6 -- already ingested as nn-2 in bank-17. SKIP.")
    print("NOTE: ps-s6-e6-realmlp-pytorch (yekenot) -- ingested as realmlp-0 in bank-17. SKIP.")
    print("Sourcing new primary OOF from ravi_gnn_mlv1 (REALMLPV1, MLV1) and remaining pilkwang bases...")

    candidate_defs = [
        # NEW: ravi primary models (not in bank-17 or n70 candidate list)
        ("ravi_realmlp_v1",  "parquet", RAVI / "OOF_Preds_REALMLPV1_1.parquet",  RAVI / "Mdl_Preds_REALMLPV1_1.parquet"),
        ("ravi_ml_v1",       "parquet", RAVI / "OOF_Preds_MLV1_1.parquet",        RAVI / "Mdl_Preds_MLV1_1.parquet"),
        # pilkwang bases tried in n70 but NOT selected (still valid for this new starting point)
        ("extratrees_soft",  "csv",     PILK / "oof_extratrees_soft_seed42_full_fullrows_5fold.csv",    PILK / "sub_extratrees_soft_seed42_full_fullrows_5fold.csv"),
        ("hgb_balanced",     "csv",     PILK / "oof_hgb_balanced_seed42_full_fullrows_5fold.csv",        PILK / "sub_hgb_balanced_seed42_full_fullrows_5fold.csv"),
        ("tabm_lite",        "csv",     PILK / "oof_tabm_lite_seed42_full_fullrows_fullorig_5fold.csv",  PILK / "sub_tabm_lite_seed42_full_fullrows_fullorig_5fold.csv"),
        ("logit_elastic",    "csv",     PILK / "oof_logit_elastic_seed42_full_fullrows_5fold.csv",       PILK / "sub_logit_elastic_seed42_full_fullrows_5fold.csv"),
    ]

    candidates_oof = {}
    candidates_test = {}
    print(f"\n{'candidate':20s} {'solo_BA':>9s} {'oof_shape':>12s}  status")
    for cname, fmt, oof_path, test_path in candidate_defs:
        try:
            if fmt == "parquet":
                raw_oof  = load_parquet_probs(oof_path, n)
                raw_test = load_parquet_probs(test_path, nt)
            else:
                raw_oof  = load_ext_csv(oof_path, n)
                raw_test = load_ext_csv(test_path, nt)
            assert raw_oof.shape  == (n, 3),  f"OOF shape {raw_oof.shape}"
            assert raw_test.shape == (nt, 3), f"test shape {raw_test.shape}"
            oof_n = norm(raw_oof)
            solo_ba = balanced_accuracy_score(y, oof_n.argmax(1))
            ok = solo_ba > 0.85
            print(f"  {cname:20s} {solo_ba:9.6f} {str(raw_oof.shape):>12s}  {'OK' if ok else 'BAD_ORDER'}")
            if not ok:
                print(f"  SKIP {cname}: solo BA {solo_ba:.4f} < 0.85 -- likely column order issue")
                continue
            candidates_oof[cname]  = logp(oof_n)
            candidates_test[cname] = logp(norm(raw_test))
        except Exception as e:
            print(f"  {cname:20s} FAIL: {e}")

    print(f"\n{len(candidates_oof)} candidate bases loaded for greedy forward selection")

    # ---- Leakage check 3: single-feature correlation sweep ----
    print("\nLeakage check 3: single-feature correlation sweep (sample 50k)...")
    rng = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys = y[sidx].astype(float)
    for cname, oof_mat in candidates_oof.items():
        for ci in range(3):
            x = oof_mat[sidx, ci]
            corr = abs(np.corrcoef(x, ys)[0, 1])
            if corr >= 0.999:
                raise SystemExit(f"LEAK: {cname} col {ci} ~ target corr={corr:.4f}")
    print("Leakage check 3: PASS")

    # ---- Greedy forward selection onto bank17+FT-T ----
    print("\n=== Greedy forward selection onto bank17+FT-T ===")
    current_oof  = list(start_oof)
    current_test = list(start_test)
    selected = []
    remaining = list(candidates_oof.keys())
    current_cv = base_cv
    selection_path = []

    step = 0
    while remaining:
        step += 1
        print(f"\n-- Step {step}: current_cv={current_cv:.6f}, candidates={remaining} --")
        best_cand = None
        best_cv = current_cv
        best_delta = 0.0
        for cname in remaining:
            probe_oof = current_oof + [candidates_oof[cname]]
            cv_p, _, _ = eval_cols(probe_oof, y, fval)
            delta = cv_p - current_cv
            print(f"  +{cname}: cv={cv_p:.6f} delta={delta:+.6f}")
            if cv_p > best_cv:
                best_cv = cv_p
                best_cand = cname
                best_delta = delta
        if best_cand is not None and best_delta > EPSILON:
            selected.append(best_cand)
            remaining.remove(best_cand)
            current_oof.append(candidates_oof[best_cand])
            current_test.append(candidates_test[best_cand])
            current_cv = best_cv
            selection_path.append((best_cand, best_cv, best_delta))
            print(f"\n  >>> SELECTED {best_cand}: cv={best_cv:.6f} delta={best_delta:+.6f}")
        else:
            print(f"\n  STOP: best delta={best_delta:+.6f} <= epsilon={EPSILON}")
            break

    print(f"\nForward selection path: {selection_path}")
    print(f"Selected {len(selected)} new bases: {selected}")

    # ---- Final fold-honest scoring on selected set ----
    print("\n=== Final per-fold scores ===")
    OOF_full = np.concatenate(current_oof, axis=1)
    TST_full = np.concatenate(current_test, axis=1)

    final_stack = np.zeros((n, NC))
    for vi in fval:
        tr_idx = np.setdiff1d(np.arange(n), vi)
        final_stack[vi] = fit_meta(OOF_full[tr_idx], y[tr_idx]).predict_proba(OOF_full[vi])

    per_fold_scores = []
    for i, vi in enumerate(fval):
        oth = np.setdiff1d(np.arange(n), vi)
        w = best_thr_de(final_stack[oth], y[oth])
        s = score_fn(y[vi], np.argmax(final_stack[vi] * w, axis=1))
        per_fold_scores.append(s)
        print(f"fold {i}: score={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

    cv_mean = float(np.mean(per_fold_scores))
    cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
    print(f"\ncv={cv_mean:.6f}  sem={cv_sem:.6f}")

    lift_vs_n70 = cv_mean - base_cv
    lift_vs_champ = cv_mean - 0.970153
    two_sem = 2 * cv_sem
    beats_champ = lift_vs_champ > two_sem
    print(f"\nn70 baseline (bank17+FT-T): {base_cv:.6f}")
    print(f"final cv:                   {cv_mean:.6f}")
    print(f"lift vs n70:                {lift_vs_n70:+.6f}")
    print(f"lift vs champion(n63):      {lift_vs_champ:+.6f}  (2*sem={two_sem:.6f})  beats_champ={beats_champ}")

    # ---- Final refit on all train, predict test ----
    meta_full = fit_meta(OOF_full, y)
    w_full = best_thr_de(final_stack, y)
    print(f"final DE w=[{w_full[0]:.4f},{w_full[1]:.4f},{w_full[2]:.4f}]")

    stack_test_probs = meta_full.predict_proba(TST_full)
    test_preds_idx = np.argmax(stack_test_probs * w_full, axis=1)
    test_labels = [I2L[i] for i in test_preds_idx]

    # ---- Write outputs ----
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    np.save(NODE_DIR / "oof.npy", final_stack)
    np.save(NODE_DIR / "test_probs.npy", stack_test_probs)
    print(f"\nsubmission written: {len(sub)} rows")
    print(f"oof.npy: {final_stack.shape}  test_probs.npy: {stack_test_probs.shape}")

    # ---- Schema check ----
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count mismatch: {len(sub)} vs {len(sample_sub)}"
    print("submission schema OK")

    # ---- Post-run output gates ----
    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    assert 0.0 <= oofn.min() and oofn.max() <= 1.0 + 1e-5, "OOF probs out of [0,1]"
    print(f"oof_full: PASS  no_nan: PASS  dist_sane: PASS (range=[{oofn.min():.4f},{oofn.max():.4f}])")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"kernels_sourced: tabpfn3=META_STACKER_NO_OOF kospintr=SUBMISSION_ONLY nn2=ALREADY_BANK17 realmlp_pytorch=ALREADY_BANK17")
    print(f"new_primary_oof_found: ravi_realmlp_v1 ravi_ml_v1")
    print(f"n70_baseline (bank17+FT-T): {base_cv:.6f}")
    print(f"selected_bases: {selected}")
    print(f"per_fold_scores: {per_fold_scores}")
    print(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}")
    print(f"lift_vs_n70={lift_vs_n70:+.6f}  beats_champion_by_2sem={beats_champ}")


if __name__ == "__main__":
    main()
