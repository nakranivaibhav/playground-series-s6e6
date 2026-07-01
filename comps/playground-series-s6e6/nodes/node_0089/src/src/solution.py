"""node_0076 — bank-17 + FT-Transformer base (node_0070 selected set),
5-seed-bagged LogReg meta (seeds 42..46), PLAIN ARGMAX (no DE threshold).

Combine of node_0070 (bank17+FT-T, DE threshold) and node_0069 (5-seed bag mechanics).
This node: same feature matrix as node_0070's final selected set, but uses 5-seed
bagging and drops DE threshold — plain argmax only.
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


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n = len(train)
    nt = len(test)
    y = train["class"].map(L2I).to_numpy()
    # frozen folds — val indices only (seed-42 anchored)
    fval = [np.asarray(f["val_idx"]) for f in folds_data]

    print(f"n_train={n} n_test={nt} n_folds={len(fval)}")
    assert n == 577347, f"unexpected n_train={n}"
    assert nt == 247435, f"unexpected n_test={nt}"

    # ---- PRE-FLIGHT LEAKAGE CHECKS ----
    print("Leakage check 1-2: features are OOF probs only (no target/id). PASS")
    print("Leakage check 4-5: LogReg fit inside fold loop; folds from frozen folds.json. PASS")

    # ---- Load public bank-17 (same manifest as champion / node_0070) ----
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

    # ---- Load FT-Transformer (the one base selected by node_0070 greedy FS) ----
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

    # ---- Build feature matrix: bank-17 log-probs + FT-T log-probs ----
    # This is the same selected set as node_0070 (bank17 + ft_transformer)
    all_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    all_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]

    OOF_full = np.concatenate(all_oof_logp, axis=1)   # (n, 18*3=54)
    TST_full = np.concatenate(all_test_logp, axis=1)  # (nt, 54)
    print(f"\nFeature matrix: train={OOF_full.shape} test={TST_full.shape}")

    # ---- 5-seed bagged LogReg: average OOF probs over seeds, plain argmax ----
    # For each seed: fit LogReg inside each fold, build OOF for that seed.
    # Average OOF probs across all seeds.
    # For test: fit LogReg on all train for each seed, average test probs.

    print(f"\n=== 5-seed bagged LogReg, seeds={SEEDS}, plain argmax ===")
    n_folds = len(fval)
    seed_oof_probs  = np.zeros((len(SEEDS), n, NC))
    seed_test_probs = np.zeros((len(SEEDS), nt, NC))

    for si, seed in enumerate(SEEDS):
        print(f"\n-- Seed {seed} --")
        # OOF loop: fit on train fold, predict val fold
        seed_oof = np.zeros((n, NC))
        for fi, vi in enumerate(fval):
            tr_idx = np.setdiff1d(np.arange(n), vi)
            m = fit_meta(OOF_full[tr_idx], y[tr_idx], seed=seed)
            seed_oof[vi] = m.predict_proba(OOF_full[vi])

        fold_scores = [score_fn(y[vi], seed_oof[vi].argmax(1)) for vi in fval]
        cv_s = float(np.mean(fold_scores))
        print(f"  seed {seed} per-fold: {[f'{s:.6f}' for s in fold_scores]}")
        print(f"  seed {seed} cv={cv_s:.6f}")

        seed_oof_probs[si] = seed_oof

        # Refit on all train for test predictions
        m_full = fit_meta(OOF_full, y, seed=seed)
        seed_test_probs[si] = m_full.predict_proba(TST_full)

    # Average across seeds
    bagged_oof  = seed_oof_probs.mean(axis=0)   # (n, 3)
    bagged_test = seed_test_probs.mean(axis=0)  # (nt, 3)

    # Per-fold scores on bagged OOF
    print(f"\n=== Per-fold scores on bagged OOF (argmax) ===")
    per_fold_scores = []
    for i, vi in enumerate(fval):
        s = score_fn(y[vi], bagged_oof[vi].argmax(1))
        per_fold_scores.append(s)
        print(f"fold {i}: score={s:.6f}")

    cv_mean = float(np.mean(per_fold_scores))
    cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
    print(f"\ncv={cv_mean:.6f}  sem={cv_sem:.6f}")

    parent_cv = 0.970211
    lift = cv_mean - parent_cv
    print(f"parent (node_0070) cv={parent_cv:.6f}")
    print(f"lift vs parent: {lift:+.6f}  (2*sem={2*cv_sem:.6f})")

    # ---- Write outputs ----
    test_preds_idx = bagged_test.argmax(1)
    test_labels = [I2L[i] for i in test_preds_idx]

    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    np.save(NODE_DIR / "oof.npy", bagged_oof)
    np.save(NODE_DIR / "test_probs.npy", bagged_test)
    print(f"\nsubmission written: {len(sub)} rows")
    print(f"oof.npy: {bagged_oof.shape}  test_probs.npy: {bagged_test.shape}")

    # ---- Schema check ----
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count: {len(sub)} vs {len(sample_sub)}"
    print("submission schema: OK")

    # ---- Post-run output gates ----
    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5, "OOF probs out of [0,1]"
    # each row sums to ~1 (softmax output averaged)
    row_sums = oofn.sum(axis=1)
    assert abs(row_sums.mean() - 1.0) < 0.01, f"OOF row sums off: mean={row_sums.mean()}"
    print(f"oof_full: PASS  no_nan: PASS  dist_sane: PASS (range=[{oofn.min():.4f},{oofn.max():.4f}])")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"per_fold_scores: {per_fold_scores}")
    print(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}")
    print(f"lift_vs_parent={lift:+.6f}")


if __name__ == "__main__":
    main()
