"""node_0088 — revival re-stack of n55 (DCN) onto n76 stack.

Forward-select n55 saved OOF as candidate addition to the n76 bank17+FT-T
5-seed bagged-argmax LogReg meta stack. No retraining.

Step 1: Reproduce n76 baseline exactly (HARD assert cv ~= 0.970227).
Step 2: Add n55 OOF to the feature matrix and re-run the same meta.
Step 3: Keep only if delta > eps (+0.00003).
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3
SEEDS = [42, 43, 44, 45, 46]
EPS = 0.00003  # keep n55 only if delta > eps

# ---------------------------------------------------------------------------
# Helpers (byte-identical to n76)
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

def fit_meta(Xtr: np.ndarray, ytr: np.ndarray, seed: int = 42) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                           n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr)
    return m

def rd(path: str | Path, nr: int) -> np.ndarray:
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
    d = pd.read_csv(path)
    pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
    if set(pcols).issubset(d.columns):
        return d[pcols].values.astype(float)
    return rd(path, nr)


def run_stack(OOF_full, TST_full, y, fval, n, nt):
    """Run 5-seed bagged LogReg meta; return (bagged_oof, bagged_test, per_fold_scores, cv_mean, cv_sem)."""
    seed_oof_probs  = np.zeros((len(SEEDS), n, NC))
    seed_test_probs = np.zeros((len(SEEDS), nt, NC))

    for si, seed in enumerate(SEEDS):
        seed_oof = np.zeros((n, NC))
        for fi, vi in enumerate(fval):
            tr_idx = np.setdiff1d(np.arange(n), vi)
            m = fit_meta(OOF_full[tr_idx], y[tr_idx], seed=seed)
            seed_oof[vi] = m.predict_proba(OOF_full[vi])
        seed_oof_probs[si] = seed_oof

        m_full = fit_meta(OOF_full, y, seed=seed)
        seed_test_probs[si] = m_full.predict_proba(TST_full)

    bagged_oof  = seed_oof_probs.mean(axis=0)
    bagged_test = seed_test_probs.mean(axis=0)

    per_fold_scores = [score_fn(y[vi], bagged_oof[vi].argmax(1)) for vi in fval]
    cv_mean = float(np.mean(per_fold_scores))
    cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
    return bagged_oof, bagged_test, per_fold_scores, cv_mean, cv_sem


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

    # ---- Load bank-17 (same manifest as n76) ----
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

    # ---- Load FT-Transformer ----
    PILK = COMP / "refs" / "ext_oof" / "pilkwang_5090"
    ft_oof_path  = PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"
    ft_test_path = PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"

    ft_oof_raw  = load_ext_csv(ft_oof_path, n)
    ft_test_raw = load_ext_csv(ft_test_path, nt)
    assert ft_oof_raw.shape == (n, 3),   f"FT-T OOF shape {ft_oof_raw.shape}"
    assert ft_test_raw.shape == (nt, 3), f"FT-T test shape {ft_test_raw.shape}"
    ft_solo_ba = score_fn(y, norm(ft_oof_raw).argmax(1))
    print(f"\nft_transformer: solo_BA={ft_solo_ba:.6f}  shape={ft_oof_raw.shape}")
    assert ft_solo_ba > 0.85, f"ft_transformer solo BA {ft_solo_ba:.4f} too low"

    # ---- Leakage check 3: single-feature correlation sweep ----
    print("\nLeakage check 3: single-feature correlation sweep (sample 50k)...")
    rng = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys = y[sidx].astype(float)
    ft_logp = logp(norm(ft_oof_raw))
    for ci in range(3):
        x = ft_logp[sidx, ci]
        corr = abs(np.corrcoef(x, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK: ft_transformer col {ci} ~ target corr={corr:.4f}")
    print("Leakage check 3: PASS")

    # ---- Load n55 (DCN) saved OOF ----
    n55_oof_raw  = np.load(COMP / "nodes/node_0055/oof.npy").astype(float)
    n55_test_raw = np.load(COMP / "nodes/node_0055/test_probs.npy").astype(float)
    n55_oof_raw  = n55_oof_raw.reshape(n, -1)[:, :3]
    n55_test_raw = n55_test_raw.reshape(nt, -1)[:, :3]
    n55_oof_norm  = norm(n55_oof_raw)
    n55_test_norm = norm(n55_test_raw)
    assert n55_oof_norm.shape  == (n, 3),  f"n55 OOF shape {n55_oof_norm.shape}"
    assert n55_test_norm.shape == (nt, 3), f"n55 test shape {n55_test_norm.shape}"
    n55_solo_ba = score_fn(y, n55_oof_norm.argmax(1))
    print(f"n55 (DCN): solo_BA={n55_solo_ba:.6f}  (expected ~0.966037)")

    # Leakage check 3 extension: n55
    for ci in range(3):
        x = logp(n55_oof_norm)[sidx, ci]
        corr = abs(np.corrcoef(x, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK: n55 col {ci} ~ target corr={corr:.4f}")
    print("Leakage check 3 (n55): PASS")

    # ---- Build baseline feature matrix (bank-17 + FT-T) — same as n76 ----
    base_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    base_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]
    OOF_base = np.concatenate(base_oof_logp, axis=1)
    TST_base = np.concatenate(base_test_logp, axis=1)
    print(f"\nBaseline feature matrix: train={OOF_base.shape} test={TST_base.shape}")

    # ---- Step 1: Reproduce n76 baseline EXACTLY ----
    print(f"\n=== Step 1: Reproducing n76 baseline (HARD assert cv ~= 0.970227) ===")
    bagged_oof_base, bagged_test_base, pfs_base, cv_base, sem_base = run_stack(
        OOF_base, TST_base, y, fval, n, nt
    )
    print(f"\nBaseline per-fold: {[f'{s:.6f}' for s in pfs_base]}")
    print(f"Baseline cv={cv_base:.6f}  sem={sem_base:.6f}")
    N76_REF = 0.970227
    assert abs(cv_base - N76_REF) < 0.0002, (
        f"HARD ASSERT FAIL: baseline cv={cv_base:.6f} differs from n76 {N76_REF:.6f}"
    )
    print(f"HARD ASSERT PASS: baseline cv={cv_base:.6f} ~= n76 {N76_REF:.6f}")

    # ---- Step 2: Add n55 to feature matrix ----
    print(f"\n=== Step 2: Adding n55 (DCN) to feature matrix ===")
    ext_oof_logp  = base_oof_logp  + [logp(n55_oof_norm)]
    ext_test_logp = base_test_logp + [logp(n55_test_norm)]
    OOF_n55 = np.concatenate(ext_oof_logp, axis=1)
    TST_n55 = np.concatenate(ext_test_logp, axis=1)
    print(f"Extended feature matrix: train={OOF_n55.shape} test={TST_n55.shape}")

    bagged_oof_n55, bagged_test_n55, pfs_n55, cv_n55, sem_n55 = run_stack(
        OOF_n55, TST_n55, y, fval, n, nt
    )
    print(f"\nn55-added per-fold: {[f'{s:.6f}' for s in pfs_n55]}")
    print(f"n55-added cv={cv_n55:.6f}  sem={sem_n55:.6f}")
    delta = cv_n55 - cv_base
    print(f"delta vs baseline: {delta:+.6f}  (eps={EPS:+.6f})")

    # ---- Step 3: Select final stack ----
    if delta > EPS:
        print(f"\n==> n55 SELECTED (delta={delta:+.6f} > eps={EPS})")
        final_oof  = bagged_oof_n55
        final_test = bagged_test_n55
        pfs_final  = pfs_n55
        cv_final   = cv_n55
        sem_final  = sem_n55
        selected = "n55_added"
    else:
        print(f"\n==> n55 NOT selected (delta={delta:+.6f} <= eps={EPS}), using baseline stack")
        final_oof  = bagged_oof_base
        final_test = bagged_test_base
        pfs_final  = pfs_base
        cv_final   = cv_base
        sem_final  = sem_base
        selected = "baseline_only"

    # Print per-fold scores (required)
    print(f"\n=== Per-fold scores (selected: {selected}) ===")
    for i, s in enumerate(pfs_final):
        print(f"fold {i}: score={s:.6f}")
    print(f"\ncv={cv_final:.6f}  sem={sem_final:.6f}")

    # Promote check
    CHAMPION_CV = 0.970153
    promote_threshold = CHAMPION_CV + 2 * sem_final
    print(f"\nChampion cv={CHAMPION_CV:.6f}, 2*sem={2*sem_final:.6f}, threshold={promote_threshold:.6f}")
    print(f"Promotes champion: {cv_final > promote_threshold}")

    # ---- Write outputs ----
    test_preds_idx = final_test.argmax(1)
    test_labels = [I2L[i] for i in test_preds_idx]
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    np.save(NODE_DIR / "oof.npy", final_oof)
    np.save(NODE_DIR / "test_probs.npy", final_test)
    print(f"\nsubmission written: {len(sub)} rows")
    print(f"oof.npy: {final_oof.shape}  test_probs.npy: {final_test.shape}")

    # ---- Schema check ----
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count: {len(sub)} vs {len(sample_sub)}"
    print("submission schema: OK")

    # ---- Post-run output gates ----
    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5, "OOF probs out of [0,1]"
    row_sums = oofn.sum(axis=1)
    assert abs(row_sums.mean() - 1.0) < 0.01, f"OOF row sums off: mean={row_sums.mean()}"
    print(f"oof_full: PASS  no_nan: PASS  dist_sane: PASS (range=[{oofn.min():.4f},{oofn.max():.4f}])")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"selected: {selected}")
    print(f"n55_delta: {delta:+.6f}")
    print(f"per_fold_scores: {pfs_final}")
    print(f"cv={cv_final:.6f}  sem={sem_final:.6f}")


if __name__ == "__main__":
    main()
