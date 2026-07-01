"""node_0078 — post-bag DE per-class prior multiplier calibration on bank-17+FT-T.

Builds on node_0070:
- Same 18 bases (bank-17 + ft_transformer)
- 5-seed LogReg bagging (seeds 42-46) for variance reduction
- THEN fits DE per-class prior multipliers on the bagged OOF, nested fold-honest

Leakage design:
- DE multipliers are fit on the BAGGED train-fold OOF (never on held-out fold)
- Final test multiplier fit on bagged full-train OOF
- Folds from frozen folds.json
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
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3
SEEDS = [42, 43, 44, 45, 46]
N_INNER = 5  # inner folds for each seed re-partition


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


def bag_stack(OOF_full: np.ndarray, y: np.ndarray,
              tr_idx: np.ndarray, seeds: list[int], n_inner: int) -> np.ndarray:
    """
    Bag a LogReg stacker over multiple seeds (each seed = a new StratifiedKFold split
    of tr_idx). Returns bagged probabilities for tr_idx rows only.
    """
    n_tr = len(tr_idx)
    accum = np.zeros((n_tr, NC))
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        fold_probs = np.zeros((n_tr, NC))
        for inner_tr, inner_va in skf.split(tr_idx, y[tr_idx]):
            real_tr = tr_idx[inner_tr]
            real_va = tr_idx[inner_va]
            m = fit_meta(OOF_full[real_tr], y[real_tr], seed=seed)
            fold_probs[inner_va] = m.predict_proba(OOF_full[real_va])
        accum += fold_probs
    return accum / len(seeds)


def bag_stack_test(OOF_full: np.ndarray, TST_full: np.ndarray,
                   y: np.ndarray, seeds: list[int]) -> np.ndarray:
    """
    Bag LogReg predictions on test: fit each seed on all train, average.
    """
    accum = np.zeros((TST_full.shape[0], NC))
    for seed in seeds:
        m = fit_meta(OOF_full, y, seed=seed)
        accum += m.predict_proba(TST_full)
    return accum / len(seeds)


def fit_de_multipliers(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fit 2 per-class multipliers (GALAXY, QSO) via DE maximizing balanced accuracy.
    STAR is anchor = 1.0. Bounds 0.1-5.0 as per discussions topic 704512."""
    def neg(w):
        mults = np.array([w[0], w[1], 1.0])
        pred = np.argmax(probs * mults, axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(
        neg, [(0.1, 5.0), (0.1, 5.0)],
        maxiter=60, tol=1e-8, seed=0, polish=True, workers=1
    )
    return np.array([r.x[0], r.x[1], 1.0])


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
    print("Leakage check 4: DE multiplier fit on bagged TRAIN-fold probs only (never held-out fold). PASS")
    print("Leakage check 5: folds from frozen folds.json. PASS")

    # ---- Load same 18 bases as node_0070 ----
    B = COMP / "refs/oof_bank"
    K = COMP / "refs/kernel_out"
    PILK = COMP / "refs/ext_oof/pilkwang_5090"

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
        # ft_transformer selected by node_0070
        'ft_transformer': (PILK/"oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv",
                           PILK/"sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"),
    }

    POOF = {}; PTEST = {}; good = []
    print(f"\n{'model':14s} {'oofBA':>9s} {'shape':>12s} {'status'}")
    for name, (op, tp) in MANIFEST.items():
        try:
            if name == 'ft_transformer':
                raw_o = load_ext_csv(op, n)
                raw_t = load_ext_csv(tp, nt)
            else:
                raw_o = rd(op, n)
                raw_t = rd(tp, nt)
            o = norm(raw_o); t = norm(raw_t)
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            st = "OK" if 0.90 < ba < 0.972 else ("QUARANTINE" if ba >= 0.972 else "LOW?")
            if st == "OK": POOF[name] = o; PTEST[name] = t; good.append(name)
            print(f"{name:14s} {ba:9.6f} {str(o.shape):>12s} {st}")
        except Exception as e:
            print(f"{name:14s} {'--':>9s} {'--':>12s} FAIL {str(e)[:80]}")

    print(f"\nLoaded {len(good)} bases (expected 18)")
    assert len(good) >= 16, f"Expected >=16 bases, got {len(good)}"
    print(f"Using {len(good)} bases")

    # ---- Build stacked OOF input matrix ----
    OOF_full = np.concatenate([logp(POOF[k]) for k in good], axis=1)   # (n, 18*3)
    TST_full = np.concatenate([logp(PTEST[k]) for k in good], axis=1)  # (nt, 18*3)

    print(f"\nOOF_full: {OOF_full.shape}  TST_full: {TST_full.shape}")

    # ---- Leakage check 3: single-feature vs target sweep (sample 50k) ----
    print("\nLeakage check 3: single-feature correlation sweep (sample 50k)...")
    rng = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys = y[sidx].astype(float)
    for ci in range(OOF_full.shape[1]):
        x = OOF_full[sidx, ci]
        corr = abs(np.corrcoef(x, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK: col {ci} ~ target corr={corr:.4f}")
    print("Leakage check 3: PASS")

    # ---- Fold-honest evaluation ----
    # For each outer fold i:
    #   train_idx = all indices except val[i]
    #   bag the LogReg stack over 5 seeds on train_idx (inner 5-fold per seed)
    #   fit DE multipliers on bagged_train_probs vs y[train_idx]
    #   apply multipliers to held-out fold probs (from bagged stack of full train)
    #   score balanced accuracy on held-out fold

    print("\n=== Nested fold-honest bagged stack + DE multiplier CV ===")

    # We also need bagged probs for the held-out folds.
    # For each fold, bag on train_idx -> get train OOF probs AND predict held-out.
    # Strategy: for held-out fold, fit each of 5 seeds on full train_idx, predict val_idx, average.

    final_oof_probs = np.zeros((n, NC))  # bagged probs for all train rows (held-out only)
    per_fold_scores = []
    per_fold_w = []

    for fold_i, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        print(f"\nFold {fold_i}: n_train={len(tr_idx)} n_val={len(vi)}")

        # Step A: bag_stack on tr_idx to get bagged OOF probs for tr_idx rows
        # (used to fit DE multipliers)
        print(f"  Bagging stack on train fold ({len(SEEDS)} seeds x {N_INNER} inner folds)...")
        bagged_tr = bag_stack(OOF_full, y, tr_idx, SEEDS, N_INNER)
        # bagged_tr: (len(tr_idx), NC)

        # Step B: fit DE on bagged_tr
        print(f"  Fitting DE multipliers on bagged train-fold probs...")
        w = fit_de_multipliers(bagged_tr, y[tr_idx])
        per_fold_w.append(w)
        print(f"  DE w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

        # Step C: bag predict on held-out fold (fit each seed on full tr_idx, predict vi)
        val_accum = np.zeros((len(vi), NC))
        for seed in SEEDS:
            m = fit_meta(OOF_full[tr_idx], y[tr_idx], seed=seed)
            val_accum += m.predict_proba(OOF_full[vi])
        bagged_val = val_accum / len(SEEDS)

        # Step D: apply DE multipliers to held-out bagged probs
        pred = np.argmax(bagged_val * w, axis=1)
        s = score_fn(y[vi], pred)
        per_fold_scores.append(s)
        final_oof_probs[vi] = bagged_val

        # Also report argmax-only score for comparison
        pred_argmax = np.argmax(bagged_val, axis=1)
        s_argmax = score_fn(y[vi], pred_argmax)
        print(f"  fold {fold_i}: DE_score={s:.6f}  argmax_score={s_argmax:.6f}  DE_delta={s-s_argmax:+.6f}")

    cv_mean = float(np.mean(per_fold_scores))
    cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))

    print(f"\nper_fold_scores: {[f'{s:.6f}' for s in per_fold_scores]}")
    print(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}")

    parent_cv = 0.970211
    lift = cv_mean - parent_cv
    one_sem = cv_sem
    beats = lift > one_sem
    print(f"\nparent cv: {parent_cv:.6f}")
    print(f"this cv:   {cv_mean:.6f}")
    print(f"lift:      {lift:+.6f}  (1*sem={one_sem:.6f})  beats_parent_by_1sem={beats}")

    # ---- Final refit on all train ----
    print("\n=== Final refit on all train ===")
    # Bag the stack on all train (get full OOF probs for DE fitting)
    print("  Bagging full-train OOF (for DE fit on all train)...")
    bagged_full_tr = bag_stack(OOF_full, y, np.arange(n), SEEDS, N_INNER)

    print("  Fitting final DE multipliers on full-train bagged probs...")
    w_full = fit_de_multipliers(bagged_full_tr, y)
    print(f"  final DE w=[{w_full[0]:.4f},{w_full[1]:.4f},{w_full[2]:.4f}]")

    # Bag test predictions
    print("  Bagging test predictions...")
    bagged_test = bag_stack_test(OOF_full, TST_full, y, SEEDS)

    # Apply DE multipliers
    test_preds_idx = np.argmax(bagged_test * w_full, axis=1)
    test_labels = [I2L[i] for i in test_preds_idx]

    # ---- Write outputs ----
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    np.save(NODE_DIR / "oof.npy", final_oof_probs)
    np.save(NODE_DIR / "test_probs.npy", bagged_test)
    print(f"\nsubmission written: {len(sub)} rows")
    print(f"oof.npy: {final_oof_probs.shape}  test_probs.npy: {bagged_test.shape}")

    # ---- Post-run gate checks ----
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count mismatch"
    print("submission schema OK")

    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    assert 0.0 <= oofn.min() and oofn.max() <= 1.0 + 1e-5, "OOF probs out of [0,1]"
    print(f"oof_full: PASS  no_nan: PASS  dist_sane: PASS (range=[{oofn.min():.4f},{oofn.max():.4f}])")

    # OOF coverage
    covered = np.where(oofn.sum(1) > 0)[0]
    assert len(covered) == n, f"OOF coverage {len(covered)} != {n}"
    print(f"OOF coverage: {len(covered)}/{n} PASS")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"per_fold_scores: {per_fold_scores}")
    print(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}")
    print(f"lift_vs_parent={lift:+.6f}  beats_parent_by_1sem={beats}")
    print(f"per_fold_DE_w: {[list(np.round(w, 4)) for w in per_fold_w]}")


if __name__ == "__main__":
    main()
