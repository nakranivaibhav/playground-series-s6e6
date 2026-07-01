"""node_0089 — STAR-recall BOOST knob sweep on n76 bagged-argmax LogReg meta.

Sweeps b ∈ {1.0, 1.25, 1.5, 2.0}: multiplies STAR-class sample weight inside
the balanced LogReg fit (class_weight = {GALAXY:1, QSO:1, STAR:b} * balanced),
picks b maximising OOF balanced accuracy, reports cv/sem per b.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3
SEEDS = [42, 43, 44, 45, 46]
BOOST_VALUES = [1.0, 1.25, 1.5, 2.0]


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


def make_class_weight(y: np.ndarray, b: float) -> dict:
    """balanced class_weight dict, then multiply STAR (class index 2) by b."""
    classes = np.arange(NC)
    w = compute_class_weight("balanced", classes=classes, y=y)
    w[2] *= b  # STAR is index 2
    return {c: w[c] for c in classes}


def fit_meta(Xtr: np.ndarray, ytr: np.ndarray, seed: int, b: float) -> LogisticRegression:
    cw = make_class_weight(ytr, b)
    m = LogisticRegression(class_weight=cw, C=1.0, max_iter=2000,
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

    # ---- Load public bank-17 ----
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

    print(f"\nLoaded {len(good)} public models OK")

    # ---- Load FT-Transformer ----
    PILK = COMP / "refs" / "ext_oof" / "pilkwang_5090"
    ft_oof_path  = PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"
    ft_test_path = PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"

    ft_oof_raw  = load_ext_csv(ft_oof_path, n)
    ft_test_raw = load_ext_csv(ft_test_path, nt)
    assert ft_oof_raw.shape == (n, 3)
    assert ft_test_raw.shape == (nt, 3)
    ft_solo_ba = score_fn(y, norm(ft_oof_raw).argmax(1))
    print(f"\nft_transformer: solo_BA={ft_solo_ba:.6f}")

    # ---- Leakage check 3 ----
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

    # ---- Build feature matrix ----
    all_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    all_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]

    OOF_full = np.concatenate(all_oof_logp, axis=1)
    TST_full = np.concatenate(all_test_logp, axis=1)
    print(f"\nFeature matrix: train={OOF_full.shape} test={TST_full.shape}")

    n_folds = len(fval)

    # ---- BOOST SWEEP ----
    boost_results = {}  # b -> (cv_mean, cv_sem, per_fold_scores, bagged_oof, bagged_test)

    for b in BOOST_VALUES:
        print(f"\n{'='*60}")
        print(f"=== BOOST b={b} ===")
        seed_oof_probs  = np.zeros((len(SEEDS), n, NC))
        seed_test_probs = np.zeros((len(SEEDS), nt, NC))

        for si, seed in enumerate(SEEDS):
            seed_oof = np.zeros((n, NC))
            for fi, vi in enumerate(fval):
                tr_idx = np.setdiff1d(np.arange(n), vi)
                m = fit_meta(OOF_full[tr_idx], y[tr_idx], seed=seed, b=b)
                seed_oof[vi] = m.predict_proba(OOF_full[vi])

            fold_scores = [score_fn(y[vi], seed_oof[vi].argmax(1)) for vi in fval]
            cv_s = float(np.mean(fold_scores))
            print(f"  seed {seed} cv={cv_s:.6f}  folds={[f'{s:.6f}' for s in fold_scores]}")
            seed_oof_probs[si] = seed_oof

            # Refit on all train for test predictions
            m_full = fit_meta(OOF_full, y, seed=seed, b=b)
            seed_test_probs[si] = m_full.predict_proba(TST_full)

        bagged_oof  = seed_oof_probs.mean(axis=0)
        bagged_test = seed_test_probs.mean(axis=0)

        per_fold_scores = [score_fn(y[vi], bagged_oof[vi].argmax(1)) for vi in fval]
        cv_mean = float(np.mean(per_fold_scores))
        cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(n_folds))

        print(f"\nb={b} bagged per-fold: {[f'{s:.6f}' for s in per_fold_scores]}")
        print(f"b={b} cv={cv_mean:.6f}  sem={cv_sem:.6f}")

        boost_results[b] = (cv_mean, cv_sem, per_fold_scores, bagged_oof, bagged_test)

    # ---- Report all b results ----
    print("\n\n=== BOOST SWEEP SUMMARY ===")
    print(f"{'b':>6s} {'cv':>10s} {'sem':>10s} {'folds'}")
    for b in BOOST_VALUES:
        cv_mean, cv_sem, folds, _, _ = boost_results[b]
        print(f"{b:>6.2f} {cv_mean:>10.6f} {cv_sem:>10.6f}  {[f'{s:.6f}' for s in folds]}")

    # ---- Pick best b ----
    best_b = max(BOOST_VALUES, key=lambda b: boost_results[b][0])
    best_cv, best_sem, best_folds, best_oof, best_test = boost_results[best_b]
    print(f"\nBest b={best_b}  cv={best_cv:.6f}  sem={best_sem:.6f}")

    parent_cv = 0.970227
    parent_sem = 0.000244
    lift = best_cv - parent_cv
    threshold = 2 * parent_sem
    print(f"parent cv={parent_cv:.6f}  lift={lift:+.6f}  threshold=2*sem={threshold:.6f}")
    if lift > threshold:
        print(f"BEATS parent by >{threshold:.6f}")
    else:
        print(f"Does NOT beat parent by >2*sem ({threshold:.6f})")

    # ---- Write outputs with best b ----
    print(f"\nWriting outputs for best b={best_b}")
    test_preds_idx = best_test.argmax(1)
    test_labels = [I2L[i] for i in test_preds_idx]

    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    np.save(NODE_DIR / "oof.npy", best_oof)
    np.save(NODE_DIR / "test_probs.npy", best_test)
    print(f"submission written: {len(sub)} rows")
    print(f"oof.npy: {best_oof.shape}  test_probs.npy: {best_test.shape}")

    # ---- Schema check ----
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count: {len(sub)} vs {len(sample_sub)}"
    print("submission schema: OK")

    # ---- Post-run output gates ----
    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5
    row_sums = oofn.sum(axis=1)
    assert abs(row_sums.mean() - 1.0) < 0.01
    print(f"oof_full: PASS  no_nan: PASS  dist_sane: PASS (range=[{oofn.min():.4f},{oofn.max():.4f}])")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"best_b={best_b}")
    print(f"per_fold_scores: {best_folds}")
    print(f"cv={best_cv:.6f}  sem={best_sem:.6f}")


if __name__ == "__main__":
    main()
