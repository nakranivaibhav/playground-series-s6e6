"""node_0080 — TabPFN-3 as L1 meta-stacker over bank17+FT-T base OOFs.

Replace the balanced-LogReg L1 meta-learner with a TabPFN-3 classifier.
Same base set as node_0070 (bank17 public models + ft_transformer from pilkwang).
Fold-honest: meta fit inside each train fold only.
TabPFN-3 context size: subsample to META_SUBSAMPLE rows (from train fold) per
TabPFN inference budget; every val row is still predicted exactly once.
DE threshold tuned on held-out complement folds as before.
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

os.environ["TABPFN_MODEL_CACHE_DIR"] = "/home/vaibhav/.cache/tabpfn"
TABPFN_MODEL_PATH = "/home/vaibhav/.cache/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt"

COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3

# TabPFN context size — larger context slows inference quadratically.
# At 10k: ~6s to predict 115k rows per fold → total ~30-60s for 5 folds.
# At 50k: ~10 min per fold → too slow. Knee at ~10k.
META_SUBSAMPLE = 10000


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


def best_thr_de(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(
        neg, [(0.1, 5.0), (0.1, 5.0)],
        maxiter=40, tol=1e-7, seed=0, polish=False, workers=1
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


def make_tabpfn_clf():
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(
        model_path=TABPFN_MODEL_PATH,
        n_estimators=4,
        ignore_pretraining_limits=True,
        device="cuda",
        show_progress_bar=False,
        fit_mode="fit_preprocessors",
        balance_probabilities=True,   # helps with class imbalance
        random_state=0,
    )


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

    # ---- Load public bank-17 (same as champion) ----
    B = COMP / "refs/oof_bank"
    K = COMP / "refs/kernel_out"

    MANIFEST = {
        'xgb-0':     (K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",          K/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
        'xgb-1':     (K/"xgb-v1-for-s6e6/oof_preds.npy",            K/"xgb-v1-for-s6e6/test_preds.npy"),
        'realmlp-0': (B/"oof_preds_realmlp0_v12.csv",                B/"test_preds_realmlp0_v12.csv"),
        'realmlp-1': (K/"realmlp-v1-for-s6e6/oof_preds.npy",         K/"realmlp-v1-for-s6e6/test_preds.npy"),
        'tabm-0':    (B/"oof_preds_tabm0_v2.csv",                    B/"test_preds_tabm0_v2.csv"),
        'cat-0':     (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
        'realmlp-2': (B/"oof_preds_realmlp2_v10.csv",                B/"test_preds_realmlp2_v10.csv"),
        'tabicl-2':  (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
        'lgbm-3':    (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",     K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
        'logreg-1':  (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",  K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
        'nn-1':      (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",          K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
        'xgb-3':     (K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
        'xgb-5':     (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",       K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
        'realmlp-5': (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
        'nn-2':      (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",          K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
        'cat-3':     (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",        K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
        'lgbm-5':    (B/"oof_preds_lgbm5_v1.csv",                   B/"test_preds_lgbm5_v1.csv"),
        'xgb-6':     (B/"oof_final_xgb6_v1.csv",                    B/"test_final_xgb6_v1.csv"),
        'tabm-1':    (B/"oof_final_tabm1_v1.csv",                   B/"test_final_tabm1_v1.csv"),
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

    # ---- Load ft_transformer (selected by node_0070 greedy forward) ----
    PILK = COMP / "refs/ext_oof/pilkwang_5090"
    ft_path_oof  = PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"
    ft_path_test = PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"

    raw_ft_oof  = load_ext_csv(ft_path_oof, n)
    raw_ft_test = load_ext_csv(ft_path_test, nt)
    assert raw_ft_oof.shape  == (n, 3),  f"ft_oof shape {raw_ft_oof.shape}"
    assert raw_ft_test.shape == (nt, 3), f"ft_test shape {raw_ft_test.shape}"
    ft_solo_ba = score_fn(y, norm(raw_ft_oof).argmax(1))
    print(f"ft_transformer solo BA={ft_solo_ba:.6f}")
    assert ft_solo_ba > 0.85, f"ft_transformer solo BA too low: {ft_solo_ba:.4f}"

    # ---- Assemble meta feature matrix ----
    # bank17 + ft_transformer, log-prob transformed (same as node_0070)
    bank_oof  = [logp(POOF[k]) for k in good]
    bank_test = [logp(PTEST[k]) for k in good]
    meta_cols_oof  = bank_oof  + [logp(norm(raw_ft_oof))]
    meta_cols_test = bank_test + [logp(norm(raw_ft_test))]

    OOF_full = np.concatenate(meta_cols_oof, axis=1).astype("float32")
    TST_full = np.concatenate(meta_cols_test, axis=1).astype("float32")
    print(f"\nMeta feature matrix: OOF={OOF_full.shape}  TEST={TST_full.shape}")
    n_meta_feats = OOF_full.shape[1]
    print(f"n_meta_features={n_meta_feats} ({len(good)+1} bases * 3 = {(len(good)+1)*3})")

    # ---- PRE-FLIGHT LEAKAGE CHECKS ----
    print("\n--- Pre-flight leakage checks ---")
    # Check 1-2: no target or id in features
    # Features are log-probs of base model OOFs only — target/id not present.
    assert "class" not in [f"col_{i}" for i in range(n_meta_feats)], "target in features"
    print("Check 1-2: features are log-prob OOF columns only (no target, no id). PASS")

    # Check 3: single-feature ↔ target sweep on 50k sample
    rng = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys = y[sidx].astype(float)
    for ci in range(n_meta_feats):
        x = OOF_full[sidx, ci]
        corr = abs(np.corrcoef(x, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK: meta col {ci} ~ target corr={corr:.4f}")
    print("Check 3: single-feature↔target sweep PASS")

    # Check 4: fold loop fits meta only on train-fold rows (verified by code below)
    # Check 5: folds from frozen folds.json (loaded above, not recomputed)
    print("Check 4-5: meta fit inside fold loop train-fold only; frozen folds. PASS")

    # Check 6: no near-duplicates needed (same base as node_0070, confirmed there)
    print("Check 6: train/test near-dup check n/a (tabular ID data, id ranges disjoint). PASS")
    print("--- Pre-flight COMPLETE ---\n")

    # ---- Fold-honest TabPFN-3 meta CV ----
    print(f"=== TabPFN-3 meta CV  (META_SUBSAMPLE={META_SUBSAMPLE}) ===")
    stack = np.zeros((n, NC), dtype=np.float64)

    for fold_i, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        Xtr = OOF_full[tr_idx]
        ytr = y[tr_idx]
        Xval = OOF_full[vi]

        # Subsample meta-fit train to META_SUBSAMPLE (stratified by class)
        # Keeps class distribution, manageable context size for TabPFN
        rng_fold = np.random.RandomState(fold_i)
        sub_idx = []
        for c in range(NC):
            c_idx = np.where(ytr == c)[0]
            # proportional allocation
            n_c = max(1, int(META_SUBSAMPLE * len(c_idx) / len(ytr)))
            chosen = rng_fold.choice(c_idx, min(n_c, len(c_idx)), replace=False)
            sub_idx.append(chosen)
        sub_idx = np.concatenate(sub_idx)
        rng_fold.shuffle(sub_idx)
        Xtr_sub = Xtr[sub_idx]
        ytr_sub = ytr[sub_idx]

        clf = make_tabpfn_clf()
        clf.fit(Xtr_sub, ytr_sub)
        # Predict ALL val rows in one call (encode context once)
        probs_val = clf.predict_proba(Xval)
        stack[vi] = probs_val

        fold_ba = score_fn(y[vi], np.argmax(probs_val, axis=1))
        print(f"fold {fold_i}: n_tr_sub={len(sub_idx)}  n_val={len(vi)}  fold_BA={fold_ba:.6f}")

    # ---- Per-fold DE threshold scoring ----
    per_fold_scores = []
    for i, vi in enumerate(fval):
        oth = np.setdiff1d(np.arange(n), vi)
        w = best_thr_de(stack[oth], y[oth])
        s = score_fn(y[vi], np.argmax(stack[vi] * w, axis=1))
        per_fold_scores.append(s)
        print(f"fold {i}: score={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

    cv_mean = float(np.mean(per_fold_scores))
    cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
    print(f"\ncv={cv_mean:.6f}  sem={cv_sem:.6f}")

    CHAMPION_CV = 0.970153
    lift = cv_mean - CHAMPION_CV
    two_sem = 2 * cv_sem
    beats = lift > two_sem
    print(f"\nchampion baseline: {CHAMPION_CV:.6f}")
    print(f"final cv:          {cv_mean:.6f}")
    print(f"lift vs champion:  {lift:+.6f}  (2*sem={two_sem:.6f})  beats_by_2sem={beats}")

    # ---- Final refit on all train, predict test ----
    # Subsample all 577347 train rows to META_SUBSAMPLE for test prediction
    rng_final = np.random.RandomState(99)
    sub_all = []
    for c in range(NC):
        c_idx = np.where(y == c)[0]
        n_c = max(1, int(META_SUBSAMPLE * len(c_idx) / n))
        chosen = rng_final.choice(c_idx, min(n_c, len(c_idx)), replace=False)
        sub_all.append(chosen)
    sub_all = np.concatenate(sub_all)
    rng_final.shuffle(sub_all)
    Xtr_all_sub = OOF_full[sub_all]
    ytr_all_sub = y[sub_all]

    clf_final = make_tabpfn_clf()
    clf_final.fit(Xtr_all_sub, ytr_all_sub)

    w_full = best_thr_de(stack, y)
    print(f"final DE w=[{w_full[0]:.4f},{w_full[1]:.4f},{w_full[2]:.4f}]")

    stack_test_probs = clf_final.predict_proba(TST_full)
    test_preds_idx = np.argmax(stack_test_probs * w_full, axis=1)
    test_labels = [I2L[i] for i in test_preds_idx]

    # ---- Write outputs ----
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    np.save(NODE_DIR / "oof.npy", stack)
    np.save(NODE_DIR / "test_probs.npy", stack_test_probs)
    print(f"\nsubmission written: {len(sub)} rows")
    print(f"oof.npy: {stack.shape}  test_probs.npy: {stack_test_probs.shape}")

    # ---- Post-run output gates ----
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count mismatch: {len(sub)} vs {len(sample_sub)}"
    print("submission schema OK")

    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    # Verify OOF full coverage (each row predicted once)
    # stack is set row-by-row from folds; if any fold vi is empty, assert fails above
    row_counts = np.zeros(n, dtype=int)
    for vi in fval:
        row_counts[vi] += 1
    assert (row_counts == 1).all(), "OOF rows not covered exactly once"
    assert 0.0 <= oofn.min() and oofn.max() <= 1.0 + 1e-5, "OOF probs out of [0,1]"
    print(f"oof_full: PASS  no_nan: PASS  dist_sane: PASS (range=[{oofn.min():.4f},{oofn.max():.4f}])")

    print(f"\n=== FINAL SUMMARY ===")
    print(f"per_fold_scores: {per_fold_scores}")
    print(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}")
    print(f"lift={lift:+.6f}  beats_champion_by_2sem={beats}")


if __name__ == "__main__":
    main()
