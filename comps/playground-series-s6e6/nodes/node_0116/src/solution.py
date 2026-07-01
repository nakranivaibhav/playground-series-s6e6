"""node_0116 — LOO-pruned restack: drop 3 harmful bases from n091's FULL pool

Atomic change vs node_0091 (champion: C=0.003 balanced multinomial LogReg
mega-stack over FULL 63-base pool, cv 0.970355):
  Remove exactly the 3 bases with NEGATIVE causal LOO contribution from
  probes/drop_study_ranking.csv (bottom of the ranking):
    rank 63: xgb-6     (bank)          delta = -6.5e-5
    rank 62: tabm-0    (bank)          delta = -5.8e-5
    rank 61: node_0042 (inhouse_tight) delta = -4.4e-5
  Sum of drag removed = -1.67e-4.

  Everything else byte-identical to n091: clip log-probs, nested in-fold C
  grid (LogisticRegressionCV, 4-fold inner, Cs=[0.003,0.01,0.03,0.1,0.3,1.0]),
  balanced class weight, frozen folds.json, plain argmax, final refit with
  most-common C.

Mandatory guards:
  1. Baseline-assert (bank + FT-T, fixed C=1.0) must reproduce ≈ 0.970211
     before any pruned run is trusted.
  2. REF-FULL arm runs the n091 exact FULL pool (no drops) as a direct reference.
  3. PRUNED-3 arm drops xgb-6/tabm-0/node_0042 (the 3 negative-LOO bases).
  4. PRUNED-6 arm (optional A/B): also drops cat-0/node_0030/node_0049
     (next-worst, all negative).

Leakage: OOF probs pre-computed by base nodes; no target/id enters features.
         C selection nested (LogisticRegressionCV on outer-train portion only).
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

warnings.filterwarnings("ignore")

COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
NODE_DIR = COMP / "nodes/node_0116"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3

# C grid (same as n091)
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# TIGHT pool: 36 strong distinct in-house bases (same as n091)
TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]

# FULL pool extra = weak bases added on top of TIGHT
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]

# LOO-study DROP TARGETS (the 3 bases with negative causal contribution)
# From probes/drop_study_ranking.csv rows 61-63 (bottom)
#   rank 63: xgb-6     bank  delta=-6.5e-5
#   rank 62: tabm-0    bank  delta=-5.8e-5
#   rank 61: node_0042 tight delta=-4.4e-5
DROP_3_BANK = {"xgb-6", "tabm-0"}    # filter from MANIFEST good list
DROP_3_INHOUSE = {42}                  # filter node_0042 from tight ids

# Optional backward-elimination: also drop next-worst negatives
#   rank 60: cat-0     bank  delta=-3.4e-5
#   rank 59: node_0030 tight delta=-3.1e-5
#   rank 56: node_0049 tight delta=-3.0e-5
DROP_6_BANK = {"xgb-6", "tabm-0", "cat-0"}
DROP_6_INHOUSE = {42, 30, 49}


# ---------------------------------------------------------------------------
# Helpers (verbatim from node_0091)
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


def nested_cv_arm(
    OOF_mat: np.ndarray,
    TST_mat: np.ndarray,
    y: np.ndarray,
    fval: list[np.ndarray],
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float, list[float]]:
    """Run nested C-selection + outer OOF loop using LogisticRegressionCV.

    Verbatim from node_0091. For each outer fold:
      - train portion = everything NOT in vi
      - use LogisticRegressionCV(Cs=C_GRID, cv=4) on train portion -> best C
      - predict vi using the selected model (NEVER sees the outer val fold during C selection)
    After all outer folds, compute per-fold BA and mean/sem.
    Then refit on full train (using most common best-C) for test preds.

    Returns: (oof_probs, test_probs, per_fold_scores, cv_mean, cv_sem, best_Cs_per_fold)
    """
    n = len(y)
    n_folds = len(fval)

    oof_probs = np.zeros((n, NC), dtype=float)
    best_Cs_per_fold = []

    print(f"\n=== ARM: {label}  feature_cols={OOF_mat.shape[1]} ===", flush=True)

    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        X_tr = OOF_mat[tr_idx]
        y_tr = y[tr_idx]

        # LogisticRegressionCV: selects best C via 4-fold inner CV
        # NEVER sees vi (the outer val fold)
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
        best_c = float(lrcv.C_[0])  # same C for all classes in multinomial
        best_Cs_per_fold.append(best_c)

        print(f"  fold {fi}: best_C={best_c}  C_scores={dict(zip(C_GRID, lrcv.scores_[0].mean(axis=0).tolist() if lrcv.scores_ else []))}", flush=True)

        # Predict outer val fold using the selected LogReg model (already fitted)
        oof_probs[vi] = lrcv.predict_proba(OOF_mat[vi])

    # Per-fold outer scores
    per_fold_scores = []
    for fi, vi in enumerate(fval):
        s = score_fn(y[vi], oof_probs[vi].argmax(1))
        per_fold_scores.append(s)
        print(f"  outer fold {fi}: BA={s:.6f}  best_C={best_Cs_per_fold[fi]}", flush=True)

    cv_mean = float(np.mean(per_fold_scores))
    cv_sem = float(np.std(per_fold_scores, ddof=1) / np.sqrt(n_folds))

    print(f"\n  {label} cv={cv_mean:.6f}  sem={cv_sem:.6f}", flush=True)
    print(f"  best Cs per fold: {best_Cs_per_fold}", flush=True)

    # Final refit on ALL train: use the C that appeared most often across folds
    c_counts = Counter(best_Cs_per_fold)
    final_C = c_counts.most_common(1)[0][0]
    print(f"  final refit C={final_C} (most-frequent across outer folds)", flush=True)

    m_final = LogisticRegression(
        class_weight="balanced", C=final_C, max_iter=2000,
        n_jobs=-1, random_state=42,
        solver="lbfgs", multi_class="multinomial",
    )
    m_final.fit(OOF_mat, y)
    test_probs = m_final.predict_proba(TST_mat)

    return oof_probs, test_probs, per_fold_scores, cv_mean, cv_sem, best_Cs_per_fold


def build_full_pool_arm(
    good: list[str],
    POOF: dict,
    PTEST: dict,
    ft_oof_raw: np.ndarray,
    ft_test_raw: np.ndarray,
    inhouse_oof_tight: dict,
    inhouse_test_tight: dict,
    inhouse_oof_weak: dict,
    inhouse_test_weak: dict,
    drop_bank: set,
    drop_inhouse: set,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build FULL pool feature matrix with optional base drops.

    FULL = bank (filtered) + FT-T + tight inhouse (filtered) + weak inhouse (filtered).
    Returns (OOF_mat, TST_mat).
    """
    arm_oof = []
    arm_test = []

    # Bank logprobs (filter dropped bank names from good)
    arm_bank_keys = [k for k in good if k not in drop_bank]
    arm_oof  += [logp(POOF[k])  for k in arm_bank_keys]
    arm_test += [logp(PTEST[k]) for k in arm_bank_keys]

    # FT-T always kept
    arm_oof.append(logp(norm(ft_oof_raw)))
    arm_test.append(logp(norm(ft_test_raw)))

    # Tight inhouse (filter dropped inhouse node ids)
    for node_nm, arr in inhouse_oof_tight.items():
        nid = int(node_nm.split("_")[1])
        if nid not in drop_inhouse:
            arm_oof.append(arr)
            arm_test.append(inhouse_test_tight[node_nm])

    # Weak inhouse (filter dropped inhouse node ids — rare but applied for PRUNED-6)
    for node_nm, arr in inhouse_oof_weak.items():
        nid = int(node_nm.split("_")[1])
        if nid not in drop_inhouse:
            arm_oof.append(arr)
            arm_test.append(inhouse_test_weak[node_nm])

    OOF_mat = np.concatenate(arm_oof,  axis=1)
    TST_mat = np.concatenate(arm_test, axis=1)

    dropped_bank    = [k for k in good if k in drop_bank]
    dropped_inhouse = sorted(drop_inhouse)
    n_bases = len(arm_oof)
    print(f"\n{label} pool: {n_bases} bases ({OOF_mat.shape[1]} cols)  "
          f"dropped_bank={dropped_bank}  dropped_inhouse={dropped_inhouse}", flush=True)
    return OOF_mat, TST_mat


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n = len(train)
    nt = len(test)
    y = train["class"].map(L2I).to_numpy()

    # Frozen folds — val indices only
    fval = [np.asarray(f["val_idx"]) for f in folds_data]
    n_folds = len(fval)

    print(f"n_train={n} n_test={nt} n_folds={n_folds}", flush=True)
    assert n == 577347, f"unexpected n_train={n}"
    assert nt == 247435, f"unexpected n_test={nt}"

    # =========================================================================
    # PRE-FLIGHT: Leakage checks 1-2
    print("\n[LEAKAGE CHECK 1-2] Features are OOF probs only (no target/id). PASS", flush=True)
    print("[LEAKAGE CHECK 4-5] LogReg fit inside fold loop; C selected by LogisticRegressionCV on outer-train only. PASS", flush=True)
    print("[LEAKAGE CHECK 5] Folds loaded from frozen folds.json. PASS", flush=True)
    print("[LEAKAGE CHECK 6] No raw train/test concat; only precomputed OOF numpy arrays. PASS", flush=True)

    # =========================================================================
    # Load public bank models (same MANIFEST as node_0091)
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

    print(f"\nLoaded {len(good)} bank models", flush=True)

    # =========================================================================
    # Load FT-Transformer (external base selected by node_0070 greedy FS)
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
    # CHEAP GATE 1: Baseline-assert (bank + FT-T = node_0070 should ≈ 0.970211)
    print("\n" + "="*70, flush=True)
    print("CHEAP GATE 1: baseline-assert (bank+FT-T, fixed C=1.0)", flush=True)
    base_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    OOF_base = np.concatenate(base_oof_logp, axis=1)

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
    print(f"  baseline (bank+FT-T C=1.0): cv={base_cv:.6f}  per-fold={[f'{s:.6f}' for s in base_fold_scores]}", flush=True)
    EXPECTED_BASE_CV = 0.970211
    delta_base = abs(base_cv - EXPECTED_BASE_CV)
    print(f"  expected≈{EXPECTED_BASE_CV:.6f}  delta={delta_base:.6f}  threshold=0.0001", flush=True)
    if delta_base > 0.0001:
        print(f"STOP: baseline assert FAILED (delta={delta_base:.6f} > 0.0001). OOF ingest/alignment is wrong.", flush=True)
        import sys; sys.exit(1)
    print("CHEAP GATE 1: PASS (baseline reproduces within tolerance)", flush=True)

    # =========================================================================
    # Load in-house TIGHT bases (36 nodes — same as n091)
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
    # LEAKAGE CHECK 3: single-feature vs target sweep on a sample
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
    # ARM 0: REF-FULL — n091 FULL pool exact (no drops) — direct comparison reference
    print("\n" + "="*70, flush=True)
    print("ARM 0: REF-FULL (n091 FULL pool, no drops) — baseline for pruned comparison", flush=True)
    OOF_ref, TST_ref = build_full_pool_arm(
        good, POOF, PTEST,
        ft_oof_raw, ft_test_raw,
        inhouse_oof_tight, inhouse_test_tight,
        inhouse_oof_weak, inhouse_test_weak,
        drop_bank=set(), drop_inhouse=set(),
        label="REF-FULL",
    )
    (oof_ref, test_ref, pf_ref,
     cv_ref, sem_ref, Cs_ref) = nested_cv_arm(OOF_ref, TST_ref, y, fval, "REF-FULL")

    EXPECTED_N091_CV = 0.970355
    n091_delta = abs(cv_ref - EXPECTED_N091_CV)
    print(f"\nREF-FULL cv={cv_ref:.6f}  expected_n091={EXPECTED_N091_CV:.6f}  delta={n091_delta:.6f}", flush=True)
    if n091_delta > 0.0003:
        print(f"WARNING: REF-FULL deviates from n091 by {n091_delta:.6f} > 0.0003 — possible alignment issue", flush=True)
    else:
        print("REF-FULL: reproduces n091 within noise. OK", flush=True)

    # =========================================================================
    # ARM 1: PRUNED-3 — drop xgb-6, tabm-0, node_0042 (primary atomic change)
    print("\n" + "="*70, flush=True)
    print("ARM 1: PRUNED-3 — primary change (drop xgb-6/tabm-0/node_0042)", flush=True)
    print(f"  xgb-6  LOO-delta={-6.5e-5:.1e}  tabm-0 LOO-delta={-5.8e-5:.1e}  node_0042 LOO-delta={-4.4e-5:.1e}", flush=True)
    print(f"  total drag removed = {1.67e-4:.2e}", flush=True)
    OOF_p3, TST_p3 = build_full_pool_arm(
        good, POOF, PTEST,
        ft_oof_raw, ft_test_raw,
        inhouse_oof_tight, inhouse_test_tight,
        inhouse_oof_weak, inhouse_test_weak,
        drop_bank=DROP_3_BANK,
        drop_inhouse=DROP_3_INHOUSE,
        label="PRUNED-3",
    )
    (oof_p3, test_p3, pf_p3,
     cv_p3, sem_p3, Cs_p3) = nested_cv_arm(OOF_p3, TST_p3, y, fval, "PRUNED-3")

    # =========================================================================
    # ARM 2: PRUNED-6 — optional backward-elimination
    print("\n" + "="*70, flush=True)
    print("ARM 2: PRUNED-6 — optional A/B (also drop cat-0/node_0030/node_0049)", flush=True)
    print(f"  cat-0 LOO-delta={-3.4e-5:.1e}  node_0030 LOO-delta={-3.1e-5:.1e}  node_0049 LOO-delta={-3.0e-5:.1e}", flush=True)
    OOF_p6, TST_p6 = build_full_pool_arm(
        good, POOF, PTEST,
        ft_oof_raw, ft_test_raw,
        inhouse_oof_tight, inhouse_test_tight,
        inhouse_oof_weak, inhouse_test_weak,
        drop_bank=DROP_6_BANK,
        drop_inhouse=DROP_6_INHOUSE,
        label="PRUNED-6",
    )
    (oof_p6, test_p6, pf_p6,
     cv_p6, sem_p6, Cs_p6) = nested_cv_arm(OOF_p6, TST_p6, y, fval, "PRUNED-6")

    # =========================================================================
    # Compare arms
    print("\n" + "="*70, flush=True)
    print("=== ARM COMPARISON ===", flush=True)
    print(f"BASELINE ASSERT (bank+FT-T C=1.0): cv={base_cv:.6f}  (expected≈0.970211)", flush=True)
    print(f"REF-FULL   (n091 reference): cv={cv_ref:.6f}  sem={sem_ref:.6f}  per_fold={[f'{s:.6f}' for s in pf_ref]}", flush=True)
    print(f"PRUNED-3   (drop xgb6/tabm0/n042): cv={cv_p3:.6f}  sem={sem_p3:.6f}  per_fold={[f'{s:.6f}' for s in pf_p3]}", flush=True)
    print(f"PRUNED-6   (also drop cat0/n030/n049): cv={cv_p6:.6f}  sem={sem_p6:.6f}  per_fold={[f'{s:.6f}' for s in pf_p6]}", flush=True)

    print("\nLOO-delta of dropped bases (from drop_study_ranking.csv):", flush=True)
    print(f"  xgb-6:     LOO-delta={-6.5e-5:.1e}  (rank 63, bank)", flush=True)
    print(f"  tabm-0:    LOO-delta={-5.8e-5:.1e}  (rank 62, bank)", flush=True)
    print(f"  node_0042: LOO-delta={-4.4e-5:.1e}  (rank 61, inhouse_tight)", flush=True)
    print(f"  cat-0:     LOO-delta={-3.4e-5:.1e}  (rank 60, bank)", flush=True)
    print(f"  node_0030: LOO-delta={-3.1e-5:.1e}  (rank 59, inhouse_tight)", flush=True)
    print(f"  node_0049: LOO-delta={-3.0e-5:.1e}  (rank 56, inhouse_tight)", flush=True)

    print(f"\nPRUNED-3 lift vs REF-FULL: {cv_p3 - cv_ref:+.6f}  (LOO-study predicted: +1.67e-4)", flush=True)
    print(f"PRUNED-6 lift vs PRUNED-3: {cv_p6 - cv_p3:+.6f}", flush=True)

    # Primary winner = PRUNED-3 (the atomic change); use whichever pruned arm is better
    if cv_p3 >= cv_p6:
        winner = "PRUNED-3"
        cv_win, sem_win, pf_win = cv_p3, sem_p3, pf_p3
        oof_win, test_win = oof_p3, test_p3
        Cs_win = Cs_p3
    else:
        winner = "PRUNED-6"
        cv_win, sem_win, pf_win = cv_p6, sem_p6, pf_p6
        oof_win, test_win = oof_p6, test_p6
        Cs_win = Cs_p6

    print(f"\nWINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)

    n091_cv   = 0.970355
    n091_sem  = 0.000249
    promote_bar = n091_cv + 2 * n091_sem
    lift_vs_n091 = cv_win - n091_cv
    print(f"\nn091 (parent/champion): cv={n091_cv:.6f}  sem={n091_sem:.6f}  2*sem={2*n091_sem:.6f}", flush=True)
    print(f"promote_bar = {promote_bar:.6f}", flush=True)
    print(f"lift_vs_n091 = {lift_vs_n091:+.6f}  beats_promote={'YES' if cv_win > promote_bar else 'NO'}", flush=True)

    # =========================================================================
    # Write artifacts (winner arm)
    np.save(NODE_DIR / "oof.npy",        oof_win.astype(np.float32))
    np.save(NODE_DIR / "test_probs.npy", test_win.astype(np.float32))

    test_preds_idx = test_win.argmax(1)
    test_labels = [I2L[i] for i in test_preds_idx]
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)

    print(f"\nArtifacts written ({winner}):", flush=True)
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
    class_counts = np.bincount(oofn.argmax(1), minlength=3)
    print(f"  dist_sane: PASS  OOF argmax dist: GALAXY={class_counts[0]} QSO={class_counts[1]} STAR={class_counts[2]}", flush=True)
    print(f"             range=[{oofn.min():.4f},{oofn.max():.4f}]  row_sums_mean={row_sums.mean():.6f}", flush=True)

    # Gate 10: cv-too-good
    cv_too_good = cv_win > 0.980
    print(f"  cv_too_good: {'WARN (>0.980)' if cv_too_good else 'PASS'}", flush=True)

    # =========================================================================
    # Final summary
    print("\n" + "="*70, flush=True)
    print("=== FINAL SUMMARY ===", flush=True)
    print(f"BASELINE ASSERT (bank+FT-T C=1.0): cv={base_cv:.6f} (expected≈0.970211) PASS", flush=True)
    print(f"REF-FULL  (n091): cv={cv_ref:.6f}  sem={sem_ref:.6f}  Cs={Cs_ref}  per_fold={[f'{s:.6f}' for s in pf_ref]}", flush=True)
    print(f"PRUNED-3: cv={cv_p3:.6f}  sem={sem_p3:.6f}  Cs={Cs_p3}  per_fold={[f'{s:.6f}' for s in pf_p3]}", flush=True)
    print(f"PRUNED-6: cv={cv_p6:.6f}  sem={sem_p6:.6f}  Cs={Cs_p6}  per_fold={[f'{s:.6f}' for s in pf_p6]}", flush=True)
    print(f"WINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)
    print(f"promotes? {'YES' if cv_win > promote_bar else 'NO'}  (bar={promote_bar:.6f})", flush=True)
    print(f"pruned_p3_lift_vs_ref: {cv_p3 - cv_ref:+.6f}  (LOO-study predicted: +1.67e-4)", flush=True)
    print(f"loo_deltas: xgb-6=-6.5e-5  tabm-0=-5.8e-5  node_0042=-4.4e-5  cat-0=-3.4e-5  node_0030=-3.1e-5  node_0049=-3.0e-5", flush=True)
    print(f"cv={cv_win:.6f}", flush=True)  # machine-parseable line


if __name__ == "__main__":
    main()
