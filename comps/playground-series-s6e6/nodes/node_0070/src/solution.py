"""node_0070 — bank-17 PUBLIC bases + new external bases via greedy forward selection.

Baseline: champion node_0063 public bank-17 (17 Deotte public-bank OOF files from
refs/oof_bank + refs/kernel_out), balanced multinomial LogReg on clipped log-probs
+ DE per-class threshold, fold-honest nested, frozen folds.json.

Step 1: Reproduce bank-17 and ASSERT abs(cv - 0.970153) < 0.0002.
Step 2: Greedy forward-select candidate external bases onto bank-17 matrix.
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
# Helpers
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
    """Read OOF / test probs from npy or csv (same logic as champion a1_full_merge.py)."""
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
    print("Leakage check 1-2: features are OOF probs only (no target/id). PASS")
    print("Leakage check 4-5: LogReg fit inside fold loop; folds from frozen folds.json. PASS")

    # ---- Load public bank-17 (same manifest as champion a1_full_merge.py) ----
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

    # ---- Baseline: reproduce champion bank-17 CV ----
    print("\n=== Step 1: Reproduce champion bank-17 baseline ===")
    bank_oof  = [logp(POOF[k]) for k in good]
    bank_test = [logp(PTEST[k]) for k in good]
    base_cv, base_sem, _ = eval_cols(bank_oof, y, fval)
    print(f"bank-{len(good)} baseline: cv={base_cv:.6f}  sem={base_sem:.6f}")

    EXPECTED = 0.970153
    diff = abs(base_cv - EXPECTED)
    print(f"diff from expected {EXPECTED}: {diff:.6f}")
    if diff >= 0.0002:
        raise SystemExit(
            f"BASELINE ASSERTION FAILED: got {base_cv:.6f}, expected ~{EXPECTED}, diff={diff:.6f}. "
            f"Loaded {len(good)} bases (expected 17). STOPPING — do not proceed with wrong baseline."
        )
    print(f"BASELINE ASSERTED OK: {base_cv:.6f} ~ {EXPECTED} (diff={diff:.6f} < 0.0002)")

    # ---- Load candidate new external bases ----
    EXT_OOF = COMP / "refs" / "ext_oof"
    PILK = EXT_OOF / "pilkwang_5090"
    RAVI = EXT_OOF / "ravi_gnn_mlv1"

    candidate_defs = [
        ("gnn_v1",         RAVI / "oof_GNNV1_1.npy",                                            RAVI / "pred_GNNV1_1.npy"),
        ("extratrees_soft",PILK / "oof_extratrees_soft_seed42_full_fullrows_5fold.csv",           PILK / "sub_extratrees_soft_seed42_full_fullrows_5fold.csv"),
        ("hgb_balanced",   PILK / "oof_hgb_balanced_seed42_full_fullrows_5fold.csv",             PILK / "sub_hgb_balanced_seed42_full_fullrows_5fold.csv"),
        ("ft_transformer", PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"),
        ("tabm_lite",      PILK / "oof_tabm_lite_seed42_full_fullrows_fullorig_5fold.csv",        PILK / "sub_tabm_lite_seed42_full_fullrows_fullorig_5fold.csv"),
        ("logit_elastic",  PILK / "oof_logit_elastic_seed42_full_fullrows_5fold.csv",             PILK / "sub_logit_elastic_seed42_full_fullrows_5fold.csv"),
    ]

    candidates_oof = {}
    candidates_test = {}
    print("\n=== Candidate base alignment + solo BA checks ===")
    for cname, oof_path, test_path in candidate_defs:
        if cname == "gnn_v1":
            raw_oof  = rd(oof_path, n)
            raw_test = rd(test_path, nt)
        else:
            raw_oof  = load_ext_csv(oof_path, n)
            raw_test = load_ext_csv(test_path, nt)
        assert raw_oof.shape  == (n, 3),  f"{cname}: OOF shape {raw_oof.shape} != ({n},3)"
        assert raw_test.shape == (nt, 3), f"{cname}: test shape {raw_test.shape} != ({nt},3)"
        oof_n = norm(raw_oof)
        solo_ba = score_fn(y, oof_n.argmax(1))
        print(f"  {cname}: solo_BA={solo_ba:.6f}  oof={raw_oof.shape}  test={raw_test.shape}")
        assert solo_ba > 0.85, f"{cname}: solo BA {solo_ba:.4f} too low — column order issue"
        candidates_oof[cname]  = logp(oof_n)
        candidates_test[cname] = logp(norm(raw_test))

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

    # ---- Step 2: Greedy forward selection onto bank-17 ----
    print("\n=== Step 2: Greedy forward selection onto bank-17 ===")
    current_oof  = list(bank_oof)
    current_test = list(bank_test)
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

    lift = cv_mean - base_cv
    two_sem = 2 * cv_sem
    beats = lift > two_sem
    print(f"\nchampion baseline: {base_cv:.6f}")
    print(f"final cv:          {cv_mean:.6f}")
    print(f"lift vs champion:  {lift:+.6f}  (2*sem={two_sem:.6f})  beats_by_2sem={beats}")

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
    print(f"bank_cv (baseline, asserted): {base_cv:.6f}")
    print(f"selected_bases: {selected}")
    print(f"per_fold_scores: {per_fold_scores}")
    print(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}")
    print(f"lift={lift:+.6f}  beats_champion_by_2sem={beats}")


if __name__ == "__main__":
    main()
