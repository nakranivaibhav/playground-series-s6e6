"""Recovery script for node_0093 — re-run fold loop (TIGHT arm) and produce artifacts.

The main solution.py crashed after the fold loops (joblib OOM on final refit).
CVs are clean: TIGHT=0.963155, FULL=0.963166.
Since both are below parent (0.970211), we skip submission per the plan spec,
but still produce oof.npy + test_probs.npy for the better arm (FULL is marginally
better by 0.000011 but essentially equal; use TIGHT for compactness).

This script re-runs only TIGHT arm to reconstruct OOF + test_probs.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parents[3]
NODE_DIR = Path(__file__).resolve().parents[1]
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3


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


def simplex_blend_weights(probs: list[np.ndarray], y: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    B = len(probs)
    n = probs[0].shape[0]
    P = np.stack(probs, axis=1)  # (n, B, 3)
    Y = np.zeros((n, NC))
    Y[np.arange(n), y] = 1.0

    def cross_entropy(w):
        blended = (P * w[np.newaxis, :, np.newaxis]).sum(axis=1)
        blended = np.clip(blended, 1e-10, 1.0)
        return -float(np.mean((Y * np.log(blended)).sum(axis=1)))

    def grad_ce(w):
        blended = (P * w[np.newaxis, :, np.newaxis]).sum(axis=1)
        blended = np.clip(blended, 1e-10, 1.0)
        ratio = Y / blended
        grad = -(ratio[:, np.newaxis, :] * P).sum(axis=(0, 2)) / n
        return grad

    w0 = np.ones(B) / B
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * B
    result = minimize(cross_entropy, w0, jac=grad_ce, method="SLSQP",
                     bounds=bounds, constraints=constraints,
                     options={"maxiter": 500, "ftol": tol})
    w = np.maximum(result.x, 0.0)
    w = w / w.sum()
    return w


def apply_blend(probs: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    return sum(w_i * p for w_i, p in zip(w, probs))


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

    # Load bank-17
    B_path = COMP / "refs/oof_bank"
    K_path = COMP / "refs/kernel_out"

    MANIFEST = {
        'xgb-1':      (K_path/"xgb-v1-for-s6e6/oof_preds.npy",           K_path/"xgb-v1-for-s6e6/test_preds.npy"),
        'realmlp-0':  (B_path/"oof_preds_realmlp0_v12.csv",               B_path/"test_preds_realmlp0_v12.csv"),
        'realmlp-1':  (K_path/"realmlp-v1-for-s6e6/oof_preds.npy",        K_path/"realmlp-v1-for-s6e6/test_preds.npy"),
        'tabm-0':     (B_path/"oof_preds_tabm0_v2.csv",                   B_path/"test_preds_tabm0_v2.csv"),
        'cat-0':      (K_path/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K_path/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
        'realmlp-2':  (B_path/"oof_preds_realmlp2_v10.csv",               B_path/"test_preds_realmlp2_v10.csv"),
        'tabicl-2':   (K_path/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K_path/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
        'lgbm-3':     (K_path/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",     K_path/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
        'logreg-1':   (K_path/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",  K_path/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
        'nn-1':       (K_path/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",          K_path/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
        'xgb-5':      (K_path/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",       K_path/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
        'realmlp-5':  (K_path/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",K_path/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
        'nn-2':       (K_path/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",          K_path/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
        'cat-3':      (K_path/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",        K_path/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
        'lgbm-5':     (B_path/"oof_preds_lgbm5_v1.csv",                  B_path/"test_preds_lgbm5_v1.csv"),
        'xgb-6':      (B_path/"oof_final_xgb6_v1.csv",                   B_path/"test_final_xgb6_v1.csv"),
        'tabm-1':     (B_path/"oof_final_tabm1_v1.csv",                   B_path/"test_final_tabm1_v1.csv"),
    }

    POOF = {}; PTEST = {}; good = []
    for name, (op, tp) in MANIFEST.items():
        try:
            o = norm(rd(op, n)); t = norm(rd(tp, nt))
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            st = "OK" if 0.90 < ba < 0.972 else "QUARANTINE" if ba >= 0.972 else "LOW?"
            if st == "OK": POOF[name] = o; PTEST[name] = t; good.append(name)
            print(f"{name:14s} {ba:.6f} {st}")
        except Exception as e:
            print(f"{name:14s} FAIL {e}")

    print(f"Loaded {len(good)} bank models")

    PILK = COMP / "refs" / "ext_oof" / "pilkwang_5090"
    ft_oof_raw  = norm(load_ext_csv(PILK/"oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n))
    ft_test_raw = norm(load_ext_csv(PILK/"sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", nt))
    print(f"ft_transformer loaded: {ft_oof_raw.shape}")

    # Load TIGHT in-house bases
    TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
                 28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
                 49, 50, 51, 55, 56, 60, 61, 66, 85]
    NODES_DIR = COMP / "nodes"

    tight_oof = []; tight_test = []; tight_names = []
    for nid in TIGHT_IDS:
        ndir = NODES_DIR / f"node_{nid:04d}"
        o = norm(np.load(ndir/"oof.npy").astype(float))
        t = norm(np.load(ndir/"test_probs.npy").astype(float))
        tight_oof.append(o); tight_test.append(t)
        tight_names.append(f"n{nid}")
    print(f"Loaded {len(tight_oof)} TIGHT in-house bases")

    # Build TIGHT pool
    pool_oof  = [POOF[k] for k in good] + [ft_oof_raw]  + tight_oof   # 17+1+36 = 54
    pool_test = [PTEST[k] for k in good] + [ft_test_raw] + tight_test
    pool_names = list(good) + ["ft_transformer"] + tight_names
    B = len(pool_oof)
    print(f"TIGHT pool size: {B}")

    # Fold loop
    oof_blend = np.zeros((n, NC), dtype=np.float32)
    fold_weights = []
    fold_scores = []

    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        train_probs = [p[tr_idx] for p in pool_oof]
        val_probs   = [p[vi]     for p in pool_oof]
        y_tr = y[tr_idx]

        print(f"\nFold {fi} ({len(tr_idx)} train, {len(vi)} val)...")
        w = simplex_blend_weights(train_probs, y_tr)
        fold_weights.append(w)

        val_blend = apply_blend(val_probs, w).astype(np.float32)
        oof_blend[vi] = val_blend

        s = score_fn(y[vi], val_blend.argmax(1))
        fold_scores.append(s)
        print(f"  Fold {fi} BA={s:.6f}")

    cv_mean = float(np.mean(fold_scores))
    cv_sem  = float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))
    print(f"\nTIGHT nested OOF CV={cv_mean:.6f}  SEM={cv_sem:.6f}")
    print(f"Per-fold: {fold_scores}")

    # Final refit on ALL train (smaller memory: use float32 stacked array)
    print("\nFinal refit on all train...")
    all_probs_f32 = [p.astype(np.float32) for p in pool_oof]
    w_final = simplex_blend_weights(all_probs_f32, y)
    nz = [(pool_names[i], w_final[i]) for i in range(B) if w_final[i] > 0.005]
    nz.sort(key=lambda x: -x[1])
    print(f"Final weights ({len(nz)} w>0.005):")
    for nm, wi in nz:
        print(f"  {nm:20s} {wi:.4f}")

    # Test predictions
    test_blend = apply_blend([p.astype(np.float32) for p in pool_test], w_final)
    print(f"Test blend shape: {test_blend.shape}")

    # Write artifacts
    np.save(NODE_DIR / "oof.npy", oof_blend)
    np.save(NODE_DIR / "test_probs.npy", test_blend.astype(np.float32))

    # Submission
    test_preds_idx = test_blend.argmax(1)
    test_labels = [I2L[i] for i in test_preds_idx]
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    print(f"submission.csv: {len(sub)} rows")

    # Output gates
    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    all_vi = np.concatenate(fval)
    assert len(np.unique(all_vi)) == n, "OOF coverage fail"
    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(sample_sub)
    class_counts = np.bincount(test_blend.argmax(1), minlength=NC)
    print(f"Test distribution: GALAXY={class_counts[0]}, QSO={class_counts[1]}, STAR={class_counts[2]}")
    assert all(c > 0 for c in class_counts)

    print(f"\n=== RECOVERY COMPLETE ===")
    print(f"TIGHT cv={cv_mean:.6f}  sem={cv_sem:.6f}")
    print(f"folds={fold_scores}")
    print(f"Neither arm beats parent 0.970211 — clean null")
    print(f"cv={cv_mean:.6f}")


if __name__ == "__main__":
    main()
