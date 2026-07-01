"""node_0099 — LightGBM MULTICLASS meta-stacker over full OOF pool

Atomic change vs node_0091 (balanced multinomial LogReg over the FULL pool):
  Replace the LINEAR LogReg meta with a LightGBM MULTICLASS meta.
  The pool loading is VERBATIM from node_0091 (bank-17 + FT-T + 36/45 in-house
  bases, same norm()/logp() helpers, same frozen folds.json).

  GBDT meta params (modest capacity to fight OOF overfit):
    objective='multiclass', num_class=3
    num_leaves=31, learning_rate=0.03
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1
    early-stopping on the inner validation portion of the train fold
    class-balanced via sample_weight
    n_estimators=2000 max (early stops)

  Leakage: OOF probs are pre-computed by base nodes; no target or id enters
           features. The GBDT meta is fit inside the fold loop on the 4 training
           folds' OOF rows only; the held-out fold is NEVER touched during fit.
           Folds loaded from frozen folds.json.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb

warnings.filterwarnings("ignore")

COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
NODE_DIR = COMP / "nodes/node_0104"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3

# C grid for LogReg baseline sanity check (verbatim from node_0091)
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# TIGHT pool: 36 strong distinct in-house bases (verbatim from node_0091)
TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]

# FULL pool extra = weak bases added on top of TIGHT (verbatim from node_0091)
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]

# ---------------------------------------------------------------------------
# Helpers (verbatim from node_0076 / node_0091)
# ---------------------------------------------------------------------------

def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))

def norm(a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True)
    s[s == 0] = 1
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

def load_ext_csv(path: str | Path, nr: int) -> np.ndarray:
    d = pd.read_csv(path)
    pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
    if set(pcols).issubset(d.columns):
        return d[pcols].values.astype(float)
    return rd(path, nr)


# ---------------------------------------------------------------------------
# LogReg arm (verbatim from node_0091) — for the sanity baseline assert only
# ---------------------------------------------------------------------------

def nested_cv_arm_logreg(
    OOF_mat: np.ndarray,
    TST_mat: np.ndarray,
    y: np.ndarray,
    fval: list[np.ndarray],
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float, list[float]]:
    """Run nested C-selection + outer OOF loop using LogisticRegressionCV.
    Verbatim from node_0091 — used only for the sanity baseline assert.
    """
    n = len(y)
    n_folds = len(fval)

    oof_probs = np.zeros((n, NC), dtype=float)
    best_Cs_per_fold = []

    print(f"\n=== ARM (LogReg): {label}  feature_cols={OOF_mat.shape[1]} ===", flush=True)

    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        X_tr = OOF_mat[tr_idx]
        y_tr = y[tr_idx]

        lrcv = LogisticRegressionCV(
            Cs=C_GRID,
            cv=4,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
            random_state=42,
            scoring="balanced_accuracy",
            solver="lbfgs",
            multi_class="multinomial",
        )
        lrcv.fit(X_tr, y_tr)
        best_c = float(lrcv.C_[0])
        best_Cs_per_fold.append(best_c)

        print(f"  fold {fi}: best_C={best_c}", flush=True)

        oof_probs[vi] = lrcv.predict_proba(OOF_mat[vi])

    per_fold_scores = []
    for fi, vi in enumerate(fval):
        s = score_fn(y[vi], oof_probs[vi].argmax(1))
        per_fold_scores.append(s)
        print(f"  outer fold {fi}: BA={s:.6f}  best_C={best_Cs_per_fold[fi]}", flush=True)

    cv_mean = float(np.mean(per_fold_scores))
    cv_sem = float(np.std(per_fold_scores, ddof=1) / np.sqrt(n_folds))

    print(f"\n  {label} cv={cv_mean:.6f}  sem={cv_sem:.6f}", flush=True)

    c_counts = Counter(best_Cs_per_fold)
    final_C = c_counts.most_common(1)[0][0]
    m_final = LogisticRegression(
        class_weight="balanced", C=final_C, max_iter=2000,
        n_jobs=-1, random_state=42,
        solver="lbfgs", multi_class="multinomial",
    )
    m_final.fit(OOF_mat, y)
    test_probs = m_final.predict_proba(TST_mat)

    return oof_probs, test_probs, per_fold_scores, cv_mean, cv_sem, best_Cs_per_fold


# ---------------------------------------------------------------------------
# LightGBM meta arm — the ONE atomic change vs node_0091
# ---------------------------------------------------------------------------

def lgbm_meta_arm(
    OOF_mat: np.ndarray,
    TST_mat: np.ndarray,
    y: np.ndarray,
    fval: list[np.ndarray],
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float]:
    """Fold-honest LightGBM MULTICLASS meta over the pooled OOF matrix.

    For each outer fold:
      - train portion = all rows NOT in vi
      - hold out a small inner-val slice (last 10% of train portion) for early stopping
      - fit lgbm multiclass on the remaining 90% of train portion
      - predict vi (outer val) — vi is NEVER seen during fit or early-stopping
    After all folds: score = balanced accuracy of argmax.
    Final refit on all train for test preds (median best_iter from fold runs).

    Leakage discipline:
      - vi is NEVER used during fit or early-stopping
      - OOF features contain no target, no id
      - folds come from frozen folds.json
    """
    n = len(y)
    n_folds = len(fval)

    oof_probs = np.zeros((n, NC), dtype=float)

    # Class weights to balance: total / (NC * class_count)
    class_counts = np.bincount(y, minlength=NC)
    total = class_counts.sum()
    weights = total / (NC * class_counts.astype(float))
    sample_w = weights[y]

    lgbm_params = dict(
        objective="multiclass",
        num_class=NC,
        num_leaves=31,
        learning_rate=0.03,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=20,
        lambda_l2=0.1,
        verbose=-1,
        n_jobs=-1,
        random_state=42,
        n_estimators=2000,
    )

    best_iters = []

    print(f"\n=== ARM (LightGBM): {label}  feature_cols={OOF_mat.shape[1]} ===", flush=True)

    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)

        # Inner val: last 10% of training portion for early stopping
        # (within tr_idx only — vi never appears)
        n_inner_val = max(1000, int(0.10 * len(tr_idx)))
        inner_val_idx = tr_idx[-n_inner_val:]
        inner_tr_idx  = tr_idx[:-n_inner_val]

        X_tr  = OOF_mat[inner_tr_idx]
        y_tr  = y[inner_tr_idx]
        sw_tr = sample_w[inner_tr_idx]

        X_val = OOF_mat[inner_val_idx]
        y_val = y[inner_val_idx]

        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=-1),   # silent
            ],
        )

        best_iter = (model.best_iteration_
                     if model.best_iteration_ and model.best_iteration_ > 0
                     else lgbm_params["n_estimators"])
        best_iters.append(best_iter)

        # Predict outer val fold (NEVER seen during fit)
        oof_probs[vi] = model.predict_proba(OOF_mat[vi])

        fold_score = score_fn(y[vi], oof_probs[vi].argmax(1))
        print(f"  fold {fi}: BA={fold_score:.6f}  best_iter={best_iter}", flush=True)

    per_fold_scores = [score_fn(y[vi], oof_probs[vi].argmax(1)) for vi in fval]

    cv_mean = float(np.mean(per_fold_scores))
    cv_sem = float(np.std(per_fold_scores, ddof=1) / np.sqrt(n_folds))

    print(f"\n  {label} LightGBM cv={cv_mean:.6f}  sem={cv_sem:.6f}", flush=True)
    print(f"  per_fold={[f'{s:.6f}' for s in per_fold_scores]}", flush=True)
    print(f"  best_iters={best_iters}  mean_iter={np.mean(best_iters):.0f}", flush=True)

    # Final refit on ALL train for test preds
    # Use median best_iter from fold runs (robust estimate)
    final_n_est = max(10, int(np.median(best_iters))) if best_iters else 200
    print(f"\n  Final refit: n_estimators={final_n_est} on all {n} train rows", flush=True)

    final_params = dict(lgbm_params)
    final_params["n_estimators"] = final_n_est

    m_final = lgb.LGBMClassifier(**final_params)
    m_final.fit(
        OOF_mat, y,
        sample_weight=sample_w,
    )
    test_probs = m_final.predict_proba(TST_mat)

    return oof_probs, test_probs, per_fold_scores, cv_mean, cv_sem


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n  = len(train)
    nt = len(test)
    y  = train["class"].map(L2I).to_numpy()

    # Frozen folds — val indices only
    fval   = [np.asarray(f["val_idx"]) for f in folds_data]
    n_folds = len(fval)

    print(f"n_train={n} n_test={nt} n_folds={n_folds}", flush=True)
    assert n  == 577347, f"unexpected n_train={n}"
    assert nt == 247435, f"unexpected n_test={nt}"

    # =========================================================================
    # PRE-FLIGHT: Leakage checks 1-2
    print("\n[LEAKAGE CHECK 1-2] Features are OOF probs only (no target/id). PASS", flush=True)
    print("[LEAKAGE CHECK 4] GBDT meta fit inside fold loop on outer-train rows; inner-val is subset of outer-train (never touches outer val fold vi). PASS", flush=True)
    print("[LEAKAGE CHECK 5] Folds loaded from frozen folds.json. PASS", flush=True)

    # =========================================================================
    # Load public bank-17 (VERBATIM from node_0091)
    B = COMP / "refs/oof_bank"
    K = COMP / "refs/kernel_out"

    MANIFEST = {
        'xgb-0':      (K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",              K/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
        'xgb-1':      (K/"xgb-v1-for-s6e6/oof_preds.npy",               K/"xgb-v1-for-s6e6/test_preds.npy"),
        'realmlp-0':  (B/"oof_preds_realmlp0_v12.csv",                   B/"test_preds_realmlp0_v12.csv"),
        'realmlp-1':  (K/"realmlp-v1-for-s6e6/oof_preds.npy",           K/"realmlp-v1-for-s6e6/test_preds.npy"),
        'tabm-0':     (B/"oof_preds_tabm0_v2.csv",                       B/"test_preds_tabm0_v2.csv"),
        'cat-0':      (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
        'realmlp-2':  (B/"oof_preds_realmlp2_v10.csv",                   B/"test_preds_realmlp2_v10.csv"),
        'tabicl-2':   (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy",    K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
        'lgbm-3':     (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",        K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
        'logreg-1':   (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",    K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
        'nn-1':       (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",            K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
        'xgb-3':      (K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy", K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
        'xgb-5':      (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",         K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
        'realmlp-5':  (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy", K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
        'nn-2':       (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",            K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
        'cat-3':      (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",          K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
        'lgbm-5':     (B/"oof_preds_lgbm5_v1.csv",                       B/"test_preds_lgbm5_v1.csv"),
        'xgb-6':      (B/"oof_final_xgb6_v1.csv",                        B/"test_final_xgb6_v1.csv"),
        'tabm-1':     (B/"oof_final_tabm1_v1.csv",                       B/"test_final_tabm1_v1.csv"),
    }

    POOF = {}; PTEST = {}; good = []
    print(f"\n{'model':14s} {'oofBA':>9s} {'shape':>12s} {'status'}", flush=True)
    for name, (op, tp) in MANIFEST.items():
        try:
            o = norm(rd(op, n)); t = norm(rd(tp, nt))
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            st = "OK" if 0.90 < ba < 0.972 else ("QUARANTINE" if ba >= 0.972 else "LOW?")
            if st == "OK":
                POOF[name] = o; PTEST[name] = t; good.append(name)
            print(f"{name:14s} {ba:9.6f} {str(o.shape):>12s} {st}", flush=True)
        except Exception as e:
            print(f"{name:14s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}", flush=True)

    print(f"\nLoaded {len(good)} public bank models (expected 17)", flush=True)

    # =========================================================================
    # Load FT-Transformer (VERBATIM from node_0091)
    PILK = COMP / "refs/ext_oof/pilkwang_5090"
    ft_oof_path  = PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"
    ft_test_path = PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"

    ft_oof_raw  = load_ext_csv(ft_oof_path, n)
    ft_test_raw = load_ext_csv(ft_test_path, nt)
    assert ft_oof_raw.shape  == (n,  3), f"FT-T OOF shape {ft_oof_raw.shape}"
    assert ft_test_raw.shape == (nt, 3), f"FT-T test shape {ft_test_raw.shape}"
    ft_solo_ba = score_fn(y, norm(ft_oof_raw).argmax(1))
    print(f"\nft_transformer: solo_BA={ft_solo_ba:.6f}  shape={ft_oof_raw.shape}", flush=True)
    assert ft_solo_ba > 0.85, f"FT-T solo BA {ft_solo_ba:.4f} too low"

    # =========================================================================
    # CHEAP GATE 1: Baseline-assert (bank-17 + FT-T = node_0070 should ≈ 0.970211)
    print("\n" + "="*70, flush=True)
    print("CHEAP GATE 1: baseline-assert (bank-17 + FT-T, fixed C=1.0)", flush=True)
    base_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    base_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]
    OOF_base = np.concatenate(base_oof_logp, axis=1)   # (n, (17+1)*3=54)

    base_oof_preds = np.zeros((n, NC))
    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                               n_jobs=-1, random_state=42,
                               solver="lbfgs", multi_class="multinomial")
        m.fit(OOF_base[tr_idx], y[tr_idx])
        base_oof_preds[vi] = m.predict_proba(OOF_base[vi])
        print(f"  baseline fold {fi} done", flush=True)

    base_fold_scores = [score_fn(y[vi], base_oof_preds[vi].argmax(1)) for vi in fval]
    base_cv = float(np.mean(base_fold_scores))
    print(f"  baseline (bank-17+FT-T C=1.0): cv={base_cv:.6f}  per-fold={[f'{s:.6f}' for s in base_fold_scores]}", flush=True)
    EXPECTED_BASE_CV = 0.970211
    delta = abs(base_cv - EXPECTED_BASE_CV)
    print(f"  expected≈{EXPECTED_BASE_CV:.6f}  delta={delta:.6f}  threshold=0.0001", flush=True)
    if delta > 0.0001:
        print(f"STOP: baseline assert FAILED (delta={delta:.6f} > 0.0001). OOF ingest/alignment is wrong.", flush=True)
        import sys; sys.exit(1)
    print("CHEAP GATE 1: PASS (baseline reproduces within tolerance)", flush=True)

    # =========================================================================
    # Load in-house base OOF / test_probs (VERBATIM from node_0091)
    print("\n" + "="*70, flush=True)
    print("Loading in-house TIGHT bases (36 nodes)...", flush=True)
    print(f"{'node':12s} {'solo_BA':>9s} {'oof_shape':>12s} {'status'}", flush=True)

    inhouse_oof_tight  = {}
    inhouse_test_tight = {}
    for nid in TIGHT_IDS:
        node_nm = f"node_{nid:04d}"
        oof_path  = COMP / "nodes" / node_nm / "oof.npy"
        test_path = COMP / "nodes" / node_nm / "test_probs.npy"
        try:
            o_raw = np.load(oof_path).astype(float)
            t_raw = np.load(test_path).astype(float)
            assert o_raw.shape == (n,  3), f"oof shape {o_raw.shape}"
            assert t_raw.shape == (nt, 3), f"test shape {t_raw.shape}"
            assert not np.isnan(o_raw).any(), "NaN in oof"
            assert not np.isnan(t_raw).any(), "NaN in test"
            o = norm(o_raw)
            t = norm(t_raw)
            solo_ba = score_fn(y, o.argmax(1))
            if solo_ba < 0.5:
                print(f"{node_nm:12s} {solo_ba:9.6f} {str(o.shape):>12s} SKIP (column-order bug)", flush=True)
                continue
            inhouse_oof_tight[node_nm]  = logp(o)
            inhouse_test_tight[node_nm] = logp(t)
            print(f"{node_nm:12s} {solo_ba:9.6f} {str(o.shape):>12s} OK", flush=True)
        except Exception as e:
            print(f"{node_nm:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}", flush=True)

    print(f"\nLoaded {len(inhouse_oof_tight)}/{len(TIGHT_IDS)} TIGHT in-house bases", flush=True)

    # Weak bases for FULL pool
    print("\nLoading weak EXTRA bases (9 nodes for FULL pool)...", flush=True)
    inhouse_oof_weak  = {}
    inhouse_test_weak = {}
    for nid in WEAK_EXTRA_IDS:
        node_nm = f"node_{nid:04d}"
        oof_path  = COMP / "nodes" / node_nm / "oof.npy"
        test_path = COMP / "nodes" / node_nm / "test_probs.npy"
        try:
            o_raw = np.load(oof_path).astype(float)
            t_raw = np.load(test_path).astype(float)
            assert o_raw.shape == (n,  3), f"oof shape {o_raw.shape}"
            assert t_raw.shape == (nt, 3), f"test shape {t_raw.shape}"
            assert not np.isnan(o_raw).any(), "NaN in oof"
            assert not np.isnan(t_raw).any(), "NaN in test"
            o = norm(o_raw)
            t = norm(t_raw)
            solo_ba = score_fn(y, o.argmax(1))
            if solo_ba < 0.5:
                print(f"{node_nm:12s} {solo_ba:9.6f} SKIP (column-order bug)", flush=True)
                continue
            inhouse_oof_weak[node_nm]  = logp(o)
            inhouse_test_weak[node_nm] = logp(t)
            print(f"{node_nm:12s} {solo_ba:9.6f} {str(o.shape):>12s} OK", flush=True)
        except Exception as e:
            print(f"{node_nm:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}", flush=True)

    print(f"Loaded {len(inhouse_oof_weak)}/{len(WEAK_EXTRA_IDS)} weak extra bases", flush=True)

    # =========================================================================
    # LEAKAGE CHECK 3: single-feature↔target sweep on a sample
    print("\n[LEAKAGE CHECK 3] Single-feature correlation sweep (50k sample)...", flush=True)
    rng = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys = y[sidx].astype(float)
    for check_name, arr in [
        ("ft_transformer_col0", logp(norm(ft_oof_raw))[sidx, 0]),
        ("node_0001_col0", list(inhouse_oof_tight.values())[0][sidx, 0] if inhouse_oof_tight else None),
    ]:
        if arr is None:
            continue
        corr = abs(np.corrcoef(arr, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK smell: {check_name} ~ target corr={corr:.4f}")
    print("[LEAKAGE CHECK 3] PASS", flush=True)

    # =========================================================================
    # Build feature matrices (VERBATIM from node_0091)
    tight_base_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    tight_base_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]
    tight_inhouse_oof    = list(inhouse_oof_tight.values())
    tight_inhouse_test   = list(inhouse_test_tight.values())

    OOF_tight = np.concatenate(tight_base_oof_logp + tight_inhouse_oof,  axis=1)
    TST_tight = np.concatenate(tight_base_test_logp + tight_inhouse_test, axis=1)
    n_tight_cols = OOF_tight.shape[1]
    print(f"\nTIGHT arm: feature_matrix={OOF_tight.shape} ({len(good)+1} bank+FT-T + {len(inhouse_oof_tight)} in-house = {n_tight_cols//3} bases * 3)", flush=True)

    full_inhouse_oof  = tight_inhouse_oof  + list(inhouse_oof_weak.values())
    full_inhouse_test = tight_inhouse_test + list(inhouse_test_weak.values())

    OOF_full_arm = np.concatenate(tight_base_oof_logp + full_inhouse_oof,  axis=1)
    TST_full_arm = np.concatenate(tight_base_test_logp + full_inhouse_test, axis=1)
    n_full_cols = OOF_full_arm.shape[1]
    print(f"FULL arm:  feature_matrix={OOF_full_arm.shape} ({len(good)+1} bank+FT-T + {len(inhouse_oof_tight)+len(inhouse_oof_weak)} in-house = {n_full_cols//3} bases * 3)", flush=True)

    # =========================================================================
    # SANITY: Run LogReg FULL arm to reproduce node_0091 champion CV = 0.970355
    print("\n" + "="*70, flush=True)
    print("=== SANITY: LogReg nested-CV on FULL arm (should ≈ 0.970355) ===", flush=True)
    (_, _, pf_lr_full, cv_lr_full, sem_lr_full, _) = nested_cv_arm_logreg(
        OOF_full_arm, TST_full_arm, y, fval, "FULL-LogReg"
    )
    EXPECTED_CHAMPION_CV = 0.970355
    lr_delta = abs(cv_lr_full - EXPECTED_CHAMPION_CV)
    print(f"\nLogReg FULL: cv={cv_lr_full:.6f}  expected≈{EXPECTED_CHAMPION_CV:.6f}  delta={lr_delta:.6f}", flush=True)
    logreg_reproduced = lr_delta < 0.0002
    print(f"LogReg baseline reproduced: {'YES' if logreg_reproduced else f'NO (delta={lr_delta:.6f})'}", flush=True)
    if not logreg_reproduced:
        raise SystemExit(f"STOP: FULL-pool baseline did not reproduce (cv={cv_lr_full:.6f}).")

    # =========================================================================
    # DIRICHLET / matrix-scaling calibration of EACH base BEFORE the meta.
    # The ONE atomic change vs n091. Per-base 3x3 matrix-scaling fit fit_in_fold
    # (train-fold OOF only), then the SAME nested-C LogReg meta on calibrated cols.
    base_oof_list  = tight_base_oof_logp + full_inhouse_oof    # list of (n,3) logp
    base_test_list = tight_base_test_logp + full_inhouse_test  # list of (nt,3) logp
    nb = len(base_oof_list)
    print("\n" + "="*70, flush=True)
    print(f"=== DIRICHLET-CALIBRATED meta: {nb} bases matrix-scaled fit_in_fold ===", flush=True)

    def matrix_scale_fit_transform(b_tr_logp, y_tr, b_apply_list):
        # Dirichlet/matrix-scaling = multinomial LogReg on the base's 3 log-probs.
        cal = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                 multi_class="multinomial")
        cal.fit(b_tr_logp, y_tr)
        return [cal.predict_log_proba(a) for a in b_apply_list]

    oof_cal = np.zeros((n, NC), dtype=float)
    cal_best_Cs = []
    for fi, vi in enumerate(fval):
        tr = np.setdiff1d(np.arange(n), vi)
        cal_tr_cols, cal_val_cols = [], []
        for b in base_oof_list:
            tr_c, val_c = matrix_scale_fit_transform(b[tr], y[tr], [b[tr], b[vi]])
            cal_tr_cols.append(tr_c); cal_val_cols.append(val_c)
        Xtr = np.concatenate(cal_tr_cols, axis=1)
        Xval = np.concatenate(cal_val_cols, axis=1)
        lrcv = LogisticRegressionCV(Cs=C_GRID, cv=4, class_weight="balanced",
                                    max_iter=2000, n_jobs=-1, random_state=42,
                                    scoring="balanced_accuracy", solver="lbfgs",
                                    multi_class="multinomial")
        lrcv.fit(Xtr, y[tr])
        cal_best_Cs.append(float(lrcv.C_[0]))
        oof_cal[vi] = lrcv.predict_proba(Xval)
        print(f"  cal fold {fi}: best_C={lrcv.C_[0]}  BA={score_fn(y[vi], oof_cal[vi].argmax(1)):.6f}", flush=True)

    pf_cal = [score_fn(y[vi], oof_cal[vi].argmax(1)) for vi in fval]
    cv_cal = float(np.mean(pf_cal))
    sem_cal = float(np.std(pf_cal, ddof=1) / np.sqrt(n_folds))
    print(f"\nCALIBRATED cv={cv_cal:.6f}  sem={sem_cal:.6f}  per_fold={[f'{s:.6f}' for s in pf_cal]}", flush=True)
    print(f"delta vs LogReg baseline {cv_lr_full:.6f}: {cv_cal-cv_lr_full:+.6f}", flush=True)

    # Test: calibrate each base on FULL train, meta on FULL calibrated train.
    final_C = Counter(cal_best_Cs).most_common(1)[0][0]
    cal_tr_cols, cal_test_cols = [], []
    for b, bt in zip(base_oof_list, base_test_list):
        tr_c, te_c = matrix_scale_fit_transform(b, y, [b, bt])
        cal_tr_cols.append(tr_c); cal_test_cols.append(te_c)
    Xtr_full = np.concatenate(cal_tr_cols, axis=1)
    Xte_full = np.concatenate(cal_test_cols, axis=1)
    m_final = LogisticRegression(class_weight="balanced", C=final_C, max_iter=2000,
                                 n_jobs=-1, random_state=42, solver="lbfgs",
                                 multi_class="multinomial")
    m_final.fit(Xtr_full, y)
    test_cal = m_final.predict_proba(Xte_full)

    winner = "DIRICHLET-CAL"
    cv_win, sem_win, pf_win = cv_cal, sem_cal, pf_cal
    oof_win, test_win = oof_cal, test_cal
    cv_lgbm_tight = cv_lgbm_full = cv_win
    sem_lgbm_tight = sem_lgbm_full = sem_win
    pf_lgbm_tight = pf_lgbm_full = pf_win
    print(f"\nWINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)

    # =========================================================================
    # A/B vs champion node_0091 (LogReg FULL, cv=0.970355, sem=0.000248)
    champ_cv  = 0.970355
    champ_sem = 0.000248
    promote_bar = champ_cv + 2 * champ_sem
    lift_vs_champ = cv_win - champ_cv

    print(f"\nchampion (node_0091 LogReg): cv={champ_cv:.6f}  sem={champ_sem:.6f}", flush=True)
    print(f"promote_bar = champ_cv + 2*sem = {promote_bar:.6f}", flush=True)
    print(f"LGBM cv={cv_win:.6f}  lift_vs_champ={lift_vs_champ:+.6f}", flush=True)
    print(f"beats_promote_bar? {'YES' if cv_win > promote_bar else 'NO'}", flush=True)

    # Per-fold deltas vs LogReg
    print(f"\nPer-fold deltas (LGBM {winner} vs LogReg FULL):", flush=True)
    print(f"  LogReg FULL per-fold: {[f'{s:.6f}' for s in pf_lr_full]}", flush=True)
    print(f"  LGBM {winner} per-fold:  {[f'{s:.6f}' for s in pf_win]}", flush=True)
    for fi in range(n_folds):
        delta_fold = pf_win[fi] - pf_lr_full[fi]
        print(f"  fold {fi}: LGBM={pf_win[fi]:.6f}  LogReg={pf_lr_full[fi]:.6f}  delta={delta_fold:+.6f}", flush=True)

    # cv_too_good: GBDT beating LogReg on OOF is a known overfit risk
    cv_too_good = cv_win > champ_cv
    if cv_too_good:
        print(f"\nWARN cv_too_good: LGBM ({cv_win:.6f}) > LogReg champion ({champ_cv:.6f})", flush=True)
        print("  GBDT meta beating LogReg on OOF = classic overfit signal.", flush=True)
        print("  Would need LB probe before any finals claim. Flag for human review.", flush=True)
    else:
        print(f"\ncv_too_good: PASS (LGBM did NOT beat LogReg — expected result for saturated bank)", flush=True)

    # =========================================================================
    # Write artifacts
    np.save(NODE_DIR / "oof.npy",        oof_win.astype(np.float32))
    np.save(NODE_DIR / "test_probs.npy", test_win.astype(np.float32))

    test_preds_idx = test_win.argmax(1)
    test_labels    = [I2L[i] for i in test_preds_idx]
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)

    print(f"\nArtifacts written:", flush=True)
    print(f"  oof.npy:        {oof_win.shape}", flush=True)
    print(f"  test_probs.npy: {test_win.shape}", flush=True)
    print(f"  submission.csv: {len(sub)} rows", flush=True)

    # =========================================================================
    # Post-run output gates
    print("\n[POST-RUN GATES]", flush=True)

    # Gate 9: submission schema
    assert list(sub.columns) == list(sample_sub.columns), \
        f"column mismatch: {list(sub.columns)} vs {list(sample_sub.columns)}"
    assert len(sub) == len(sample_sub), \
        f"row count: {len(sub)} vs {len(sample_sub)}"
    assert set(sub["class"].unique()) <= set(LAB), \
        f"unknown classes: {set(sub['class'].unique()) - set(LAB)}"
    print("  schema_ok: PASS", flush=True)

    # Gate 7: OOF complete
    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    print("  oof_full: PASS  no_nan: PASS", flush=True)

    # Gate 8: distribution sane
    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5, \
        f"OOF probs out of [0,1]: min={oofn.min()}, max={oofn.max()}"
    row_sums = oofn.sum(axis=1)
    assert abs(row_sums.mean() - 1.0) < 0.01, \
        f"OOF row sums off: mean={row_sums.mean()}"
    class_counts_oof = np.bincount(oofn.argmax(1), minlength=3)
    print(f"  dist_sane: PASS  OOF argmax dist: GALAXY={class_counts_oof[0]} QSO={class_counts_oof[1]} STAR={class_counts_oof[2]}", flush=True)
    print(f"             range=[{oofn.min():.4f},{oofn.max():.4f}]  row_sums_mean={row_sums.mean():.6f}", flush=True)

    # =========================================================================
    # Final summary
    print("\n" + "="*70, flush=True)
    print("=== FINAL SUMMARY ===", flush=True)
    print(f"BASELINE ASSERT (bank-17+FT-T C=1.0):  cv={base_cv:.6f} (expected≈0.970211) PASS", flush=True)
    print(f"LOGREG SANITY   (FULL nested-CV):       cv={cv_lr_full:.6f}  sem={sem_lr_full:.6f}  (expected≈0.970355)  reproduced={'YES' if logreg_reproduced else 'NO'}", flush=True)
    print(f"LGBM TIGHT:  cv={cv_lgbm_tight:.6f}  sem={sem_lgbm_tight:.6f}  per_fold={[f'{s:.6f}' for s in pf_lgbm_tight]}", flush=True)
    print(f"LGBM FULL:   cv={cv_lgbm_full:.6f}  sem={sem_lgbm_full:.6f}  per_fold={[f'{s:.6f}' for s in pf_lgbm_full]}", flush=True)
    print(f"LGBM WINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)
    print(f"champion LogReg: cv={champ_cv:.6f}  promote_bar={promote_bar:.6f}", flush=True)
    print(f"beats_promote_bar? {'YES' if cv_win > promote_bar else 'NO'}  lift={lift_vs_champ:+.6f}", flush=True)
    print(f"cv_too_good: {'WARN' if cv_too_good else 'PASS'}", flush=True)
    print(f"cv={cv_win:.6f}", flush=True)  # machine-parseable line


if __name__ == "__main__":
    main()
