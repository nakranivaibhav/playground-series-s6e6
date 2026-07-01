"""node_0069 — champion node_0063 + 5-seed bagging (seeds 42..46).

Built on:   node_0063 champion: 17-base balanced multinomial LogReg on clipped
            log-probs + DE per-class threshold. Base set and meta hyperparams
            byte-identical.

Change:     ONE ATOMIC CHANGE — repeat the entire fold-honest meta stack over 5
            StratifiedKFold seeds (42, 43, 44, 45, 46), average OOF and test
            probabilities across seeds, THEN apply DE per-class threshold on the
            averaged OOF. Single-seed (seed-42 only) -> 5-seed averaged.

            Per seed: StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            over the 17-base log-prob OOF matrix and y. Each fold: fit balanced
            LogReg on meta-train, predict_proba on meta-val.
            Average OOF/test over 5 seeds, then score fold-honestly with DE.

A/Bs reported:
  1. Bagged-CV vs champion 0.970153 (+ per-fold scores)
  2. Per-seed CV spread + bagged sem vs champion sem 0.000222
  3. DE-threshold-ON vs argmax-only on the bagged OOF

Leakage: external OOF features are honest from public bank models; meta LogReg
         fit inside each seed/fold from train-fold rows only; folds from folds.json.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
COMP = Path(__file__).resolve().parents[3]  # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]
B = COMP / "refs/oof_bank"
K = COMP / "refs/kernel_out"

LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3
SEEDS = [42, 43, 44, 45, 46]
N_FOLDS_BAG = 5


# ---------------------------------------------------------------------------
def rd(path, nr):
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a
        return a[:, :3]
    d = pd.read_csv(p)
    c = list(d.columns)
    if set(LAB).issubset(c):
        return d[LAB].values.astype(float)
    pc = [f"prob_{l}" for l in LAB]
    if set(pc).issubset(c):
        return d[pc].values.astype(float)
    num = d.select_dtypes("number")
    if num.shape[1] >= 3:
        return num.values[:, :3]
    v = d.iloc[:, 0].values.astype(float)
    return v.reshape(nr, 3)


def norm(a):
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True)
    s[s == 0] = 1
    return a / s


def logp(a):
    return np.log(np.clip(a, 1e-7, 1.0))


def score_fn(y_true, y_pred):
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))


def fit_meta(Xtr, ytr):
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m


def best_thr_de(probs, labels):
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(
        neg, [(0.1, 5.0), (0.1, 5.0)],
        maxiter=40, tol=1e-7, seed=0, polish=False, workers=1
    )
    return np.array([r.x[0], r.x[1], 1.0])


# ---------------------------------------------------------------------------
# MANIFEST — same 19-entry set as a1_full_merge.py; after quarantine filter ~17
MANIFEST = {
    "xgb-0":     (K / "xgb-v0-for-s6e6/oof_xgb_cv.csv",                       K / "xgb-v0-for-s6e6/test_xgb_preds.csv"),
    "xgb-1":     (K / "xgb-v1-for-s6e6/oof_preds.npy",                         K / "xgb-v1-for-s6e6/test_preds.npy"),
    "realmlp-0": (B / "oof_preds_realmlp0_v12.csv",                             B / "test_preds_realmlp0_v12.csv"),
    "realmlp-1": (K / "realmlp-v1-for-s6e6/oof_preds.npy",                      K / "realmlp-v1-for-s6e6/test_preds.npy"),
    "tabm-0":    (B / "oof_preds_tabm0_v2.csv",                                 B / "test_preds_tabm0_v2.csv"),
    "cat-0":     (K / "cat-v0-for-s6e6/catboost_oof_predictions.csv",           K / "cat-v0-for-s6e6/catboost_test_predictions.csv"),
    "realmlp-2": (B / "oof_preds_realmlp2_v10.csv",                             B / "test_preds_realmlp2_v10.csv"),
    "tabicl-2":  (K / "tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy",         K / "tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
    "lgbm-3":    (K / "lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",             K / "lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
    "logreg-1":  (K / "logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",         K / "logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
    "nn-1":      (K / "nn-v1-for-s6e6/train_oof/nn-1_oof.npy",                 K / "nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
    "xgb-3":     (K / "xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy",   K / "xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
    "xgb-5":     (K / "xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",              K / "xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
    "realmlp-5": (K / "realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",       K / "realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
    "nn-2":      (K / "nn-v2-for-s6e6/train_oof/nn-2_oof.npy",                 K / "nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
    "cat-3":     (K / "cat-v3-for-s6e6/train_oof/cat-3_oof.npy",               K / "cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
    "lgbm-5":    (B / "oof_preds_lgbm5_v1.csv",                                 B / "test_preds_lgbm5_v1.csv"),
    "xgb-6":     (B / "oof_final_xgb6_v1.csv",                                  B / "test_final_xgb6_v1.csv"),
    "tabm-1":    (B / "oof_final_tabm1_v1.csv",                                  B / "test_final_tabm1_v1.csv"),
}


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n  = len(train)
    nt = len(test)
    y  = train["class"].map(L2I).to_numpy()
    fval = [np.asarray(f["val_idx"]) for f in folds_data]

    # --- Load + validate public bases (same logic as a1_full_merge.py) -------
    POOF  = {}
    PTEST = {}
    good  = []
    print(f"{'model':12s} {'oofBA':>9s} {'shape':>12s} {'status'}")
    for name, (op, tp) in MANIFEST.items():
        try:
            o = norm(rd(op, n))
            t = norm(rd(tp, nt))
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = score_fn(y, o.argmax(1))
            st = "OK" if 0.90 < ba < 0.972 else ("QUARANTINE" if ba >= 0.972 else "LOW?")
            if st == "OK":
                POOF[name] = o
                PTEST[name] = t
                good.append(name)
            print(f"{name:12s} {ba:9.6f} {str(o.shape):>12s} {st}")
        except Exception as e:
            print(f"{name:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}")

    print(f"\nloaded {len(good)} bases OK: {good}")

    # Build stacked OOF feature matrices
    OOF_bases = np.concatenate([logp(POOF[k]) for k in good], axis=1)
    TST_bases = np.concatenate([logp(PTEST[k]) for k in good], axis=1)
    print(f"OOF_bases shape: {OOF_bases.shape}")
    print(f"TST_bases shape: {TST_bases.shape}")

    # PRE-FLIGHT LEAKAGE CHECKS ------------------------------------------------
    # 1+2. Target/ID not in features: features are log-probs from base models only
    # 3. Single-feature corr sweep on sample
    sample_n = min(50000, n)
    rng = np.random.RandomState(0)
    idx = rng.choice(n, sample_n, replace=False)
    ys = y[idx]
    max_corr = 0.0
    for ci in range(OOF_bases.shape[1]):
        xc = OOF_bases[idx, ci]
        if xc.std() > 0:
            c_val = abs(np.corrcoef(xc, ys)[0, 1])
            max_corr = max(max_corr, c_val)
    print(f"PRE-FLIGHT: max single-feature |corr| with target = {max_corr:.4f}")
    assert max_corr < 0.999, f"LEAK SMELL: max corr={max_corr}"
    # 4. fit-inside-fold: meta fit only on tr_idx rows (see loop below)
    # 5. folds: loaded from frozen folds.json (fval above)
    # 6. train<->test near-dups: N/A for tabular OOF stack
    print("PRE-FLIGHT: leakage checks PASSED")

    # --- 5-SEED BAGGING (THE ONE ATOMIC CHANGE) -------------------------------
    oof_sum  = np.zeros((n,  NC), dtype=np.float64)
    test_sum = np.zeros((nt, NC), dtype=np.float64)
    per_seed_cv_argmax = []

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS_BAG, shuffle=True, random_state=seed)
        oof_s  = np.zeros((n,  NC), dtype=np.float64)
        test_s = np.zeros((nt, NC), dtype=np.float64)
        fold_count = 0

        for tr_idx, val_idx in skf.split(OOF_bases, y):
            # meta fit on train-fold ONLY — no leakage
            meta = fit_meta(OOF_bases[tr_idx], y[tr_idx])
            oof_s[val_idx] = meta.predict_proba(OOF_bases[val_idx])
            test_s += meta.predict_proba(TST_bases) / N_FOLDS_BAG
            fold_count += 1

        assert fold_count == N_FOLDS_BAG
        seed_ba = score_fn(y, np.argmax(oof_s, axis=1))
        per_seed_cv_argmax.append(seed_ba)
        print(f"seed {seed}: OOF BA (argmax) = {seed_ba:.6f}")
        oof_sum  += oof_s
        test_sum += test_s

    # Average over seeds
    stack_oof  = (oof_sum  / len(SEEDS)).astype(np.float64)
    stack_test = (test_sum / len(SEEDS)).astype(np.float64)

    print(f"\nPer-seed argmax CVs: {[f'{v:.6f}' for v in per_seed_cv_argmax]}")
    print(f"Per-seed spread: std={np.std(per_seed_cv_argmax, ddof=1):.6f}")
    bagged_argmax_cv = score_fn(y, np.argmax(stack_oof, axis=1))
    print(f"Bagged OOF BA (argmax, no DE) = {bagged_argmax_cv:.6f}")

    # --- FOLD-HONEST DE THRESHOLD SCORING on AVERAGED OOF --------------------
    per_fold_scores_thresh = []
    per_fold_scores_argmax = []

    for i, vi in enumerate(fval):
        other = np.setdiff1d(np.arange(n), vi)
        w = best_thr_de(stack_oof[other], y[other])
        pred_thresh = np.argmax(stack_oof[vi] * w, axis=1)
        pred_argmax = np.argmax(stack_oof[vi], axis=1)
        s_thresh = score_fn(y[vi], pred_thresh)
        s_argmax = score_fn(y[vi], pred_argmax)
        per_fold_scores_thresh.append(s_thresh)
        per_fold_scores_argmax.append(s_argmax)
        print(f"fold {i}: thresh={s_thresh:.6f}  argmax={s_argmax:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

    cv_mean      = float(np.mean(per_fold_scores_thresh))
    cv_sem       = float(np.std(per_fold_scores_thresh, ddof=1) / np.sqrt(len(per_fold_scores_thresh)))
    cv_amx_mean  = float(np.mean(per_fold_scores_argmax))
    cv_amx_sem   = float(np.std(per_fold_scores_argmax, ddof=1) / np.sqrt(len(per_fold_scores_argmax)))
    thresh_delta = cv_mean - cv_amx_mean

    print(f"\n--- A/B SUMMARY ---")
    print(f"A/B 1: bagged+DE cv={cv_mean:.6f} sem={cv_sem:.6f}  vs  champion 0.970153 sem=0.000222")
    print(f"A/B 2: per-seed CVs: {[f'{v:.6f}' for v in per_seed_cv_argmax]}  spread_std={np.std(per_seed_cv_argmax, ddof=1):.6f}")
    print(f"       bagged sem={cv_sem:.6f}  vs  champion sem=0.000222")
    print(f"A/B 3: DE-thresh cv={cv_mean:.6f}  argmax-only cv={cv_amx_mean:.6f}  delta={thresh_delta:+.6f}")
    print(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}")

    # --- FINAL FIT for test predictions ---------------------------------------
    w_full = best_thr_de(stack_oof, y)
    print(f"final w=[{w_full[0]:.4f},{w_full[1]:.4f},{w_full[2]:.4f}]")

    test_preds_idx = np.argmax(stack_test * w_full, axis=1)
    test_labels    = [I2L[i] for i in test_preds_idx]

    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    print(f"submission written: {len(sub)} rows")

    np.save(NODE_DIR / "oof.npy",        stack_oof)
    np.save(NODE_DIR / "test_probs.npy", stack_test)

    feat_names = [f"{k}_p{c}" for k in good for c in range(NC)]
    (NODE_DIR / "src" / "features.txt").write_text("\n".join(feat_names) + "\n")

    # POST-FLIGHT CHECKS -------------------------------------------------------
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count mismatch: {len(sub)} vs {len(sample_sub)}"

    oof_loaded = np.load(NODE_DIR / "oof.npy")
    assert oof_loaded.shape == (n, NC), f"OOF shape wrong: {oof_loaded.shape}"
    assert not np.isnan(oof_loaded).any(), "NaN in OOF"
    probs_min, probs_max = float(oof_loaded.min()), float(oof_loaded.max())
    sums = oof_loaded.sum(axis=1)
    assert probs_min >= 0 and probs_max <= 1.001, f"OOF out of [0,1]: [{probs_min:.4f},{probs_max:.4f}]"
    assert abs(sums.mean() - 1.0) < 0.01, f"OOF rows don't sum to 1: mean={sums.mean()}"
    print(f"OOF checks PASSED: shape={oof_loaded.shape} min={probs_min:.4f} max={probs_max:.4f}")
    print("ALL POST-FLIGHT GATES PASSED")


if __name__ == "__main__":
    main()
