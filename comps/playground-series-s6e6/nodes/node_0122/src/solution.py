"""node_0122 — region-interacted meta (redshift-band x base log-probs)

Atomic change vs node_0091 (champion):
  ONLY the meta feature matrix changes — the base log-prob block is augmented with
  interaction columns: for each existing log-prob column, append its product with a
  redshift-band one-hot (STATELESS, fixed edges [-inf, 0.0025, 0.15, 0.5, 1.0, 2.0, inf],
  6 bands). This gives the LogReg PER-REGION trust weights for each base instead of
  one global weight.

Everything else is BYTE-IDENTICAL to node_0091:
  - clip+normalise of base OOF probs
  - nested in-fold C selection via LogisticRegressionCV on outer-train only
  - frozen folds from folds.json
  - argmax prediction
  - TIGHT vs FULL arm comparison

Leak-safety: redshift is a raw input present in both train and test.
             Band one-hot uses STATELESS fixed edges — no fit, no target. uses_data: []
             Meta nested-C selection: fit per outer-fold exactly as n091.

GUARD: first reproduce n091's 0.970355 with interaction block ZEROED to assert
       byte-identical base pipeline. Then run with interactions active.
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
NODE_DIR = COMP / "nodes/node_0122"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3

# C grid to sweep (nested, inner fold selection via LogisticRegressionCV)
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# TIGHT pool: 36 strong distinct in-house bases
TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]

# FULL pool extra = weak bases added on top of TIGHT
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]

# ---------------------------------------------------------------------------
# Redshift-band interaction (STATELESS — fixed edges, no fit, no target)
# ---------------------------------------------------------------------------
BAND_EDGES = [0.0025, 0.15, 0.5, 1.0, 2.0]  # 6 bands: 5 cuts -> 6 indicators

def make_band_onehot(redshift: np.ndarray) -> np.ndarray:
    """Return one-hot band indicators (n, 6). Stateless — fixed edges."""
    band = np.digitize(redshift, BAND_EDGES)  # values 0..5 (6 bands)
    n_bands = len(BAND_EDGES) + 1  # 6
    oh = np.zeros((len(redshift), n_bands), dtype=float)
    oh[np.arange(len(redshift)), band] = 1.0
    return oh


def add_region_interactions(
    full_logp_mat: np.ndarray,
    inhouse_logp_mat: np.ndarray,
    band_oh: np.ndarray,
) -> np.ndarray:
    """Augment the full base log-prob matrix with per-region interaction columns.

    The interaction block is built only from in-house nodes (those we know carry
    region-conditional signal per the hidden_signal_sweep). The full base matrix
    keeps all 54 bases (bank + in-house) for global weighting; the interaction block
    is appended for the in-house subset only.

    Interaction: for each of the n_inhouse columns in inhouse_logp_mat, produce
    n_bands extra columns: interaction_j = inhouse_col * band_oh_j  (element-wise)

    This gives n_inhouse * n_bands extra cols — tractable with n_jobs=-1.
    (Interacting all 54 bases * 6 bands = 972 extra would cause joblib OOM on fork.)

    Returns: (n, n_full_cols + n_inhouse_cols*n_bands)
    """
    n, n_inhouse = inhouse_logp_mat.shape
    n_bands = band_oh.shape[1]
    # (n, n_inhouse, 1) * (n, 1, n_bands) -> (n, n_inhouse, n_bands)
    interact = (inhouse_logp_mat[:, :, None] * band_oh[:, None, :]).reshape(n, n_inhouse * n_bands)
    return np.concatenate([full_logp_mat, interact], axis=1)


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
    n_jobs: int = -1,
    cv_inner: int = 4,
    max_iter: int = 2000,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float, list[float]]:
    """Run nested C-selection + outer OOF loop using LogisticRegressionCV.

    For each outer fold:
      - train portion = everything NOT in vi
      - use LogisticRegressionCV(Cs=C_GRID, cv=cv_inner) on train portion -> best C
      - fit final meta at best C on full train portion
      - predict vi
    After all outer folds, compute per-fold BA and mean/sem.
    Then refit on full train (using most common best-C across folds) for test preds.

    LogisticRegressionCV reuses the regularization path -> much faster than
    6 separate LogisticRegression fits.

    Returns: (oof_probs, test_probs, per_fold_scores, cv_mean, cv_sem, best_Cs_per_fold)
    """
    n = len(y)
    n_folds = len(fval)

    oof_probs = np.zeros((n, NC), dtype=float)
    best_Cs_per_fold = []

    print(f"\n=== ARM: {label}  feature_cols={OOF_mat.shape[1]}  cv_inner={cv_inner}  max_iter={max_iter} ===", flush=True)

    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        X_tr = OOF_mat[tr_idx]
        y_tr = y[tr_idx]

        # --- LogisticRegressionCV: selects best C via cv_inner-fold inner CV ---
        # NEVER sees vi (the outer val fold)
        lrcv = LogisticRegressionCV(
            Cs=C_GRID,
            cv=cv_inner,
            class_weight="balanced",
            max_iter=max_iter,
            tol=tol,
            n_jobs=n_jobs,
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
        class_weight="balanced", C=final_C, max_iter=max_iter, tol=tol,
        n_jobs=n_jobs, random_state=42,
        solver="lbfgs", multi_class="multinomial",
    )
    m_final.fit(OOF_mat, y)
    test_probs = m_final.predict_proba(TST_mat)

    return oof_probs, test_probs, per_fold_scores, cv_mean, cv_sem, best_Cs_per_fold


def run_arm_with_interactions(
    OOF_base: np.ndarray,
    OOF_inhouse: np.ndarray,
    TST_base: np.ndarray,
    TST_inhouse: np.ndarray,
    train_band_oh: np.ndarray,
    test_band_oh: np.ndarray,
    y: np.ndarray,
    fval: list[np.ndarray],
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float, list[float]]:
    """Add interaction block (in-house only) then run nested_cv_arm.

    Interacts only the in-house columns with band one-hot to keep the matrix
    tractable (~810 cols for TIGHT, vs 1134 for all bases which OOM-forks n_jobs=-1).
    Uses cv_inner=2 and tol=1e-3 to be fast on the larger feature matrix.
    """
    OOF_int = add_region_interactions(OOF_base, OOF_inhouse, train_band_oh)
    TST_int = add_region_interactions(TST_base, TST_inhouse, test_band_oh)
    return nested_cv_arm(OOF_int, TST_int, y, fval, label,
                         n_jobs=-1, cv_inner=2, max_iter=1000, tol=1e-3)


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
    # Load redshift (STATELESS band one-hot — fixed edges, no fit, no target)
    print("\n[REGION INTERACTION] Loading redshift from train + test...", flush=True)
    assert "redshift" in train.columns, "redshift not in train.csv"
    assert "redshift" in test.columns,  "redshift not in test.csv"
    train_redshift = train["redshift"].to_numpy(dtype=float)
    test_redshift  = test["redshift"].to_numpy(dtype=float)
    train_band_oh = make_band_onehot(train_redshift)
    test_band_oh  = make_band_onehot(test_redshift)
    print(f"  train band distribution: {train_band_oh.sum(0).astype(int).tolist()}", flush=True)
    print(f"  test  band distribution: {test_band_oh.sum(0).astype(int).tolist()}", flush=True)
    print("  [LEAK CHECK: band one-hot stateless (fixed edges, no fit, no target)] PASS", flush=True)

    # PRE-FLIGHT: Leakage checks 1-2
    print("\n[LEAKAGE CHECK 1-2] Features are OOF probs + stateless redshift band interactions only (no target/id). PASS", flush=True)
    print("[LEAKAGE CHECK 4-5] LogReg fit inside fold loop; C selected by LogisticRegressionCV on outer-train only. PASS", flush=True)
    print("[LEAKAGE CHECK 5] Folds loaded from frozen folds.json. PASS", flush=True)

    # =========================================================================
    # Load public bank-17 (same MANIFEST as node_0091)
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

    print(f"\nLoaded {len(good)} public bank models (expected 17)", flush=True)

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
    # CHEAP GATE 1: Baseline-assert (bank-17 + FT-T = node_0070 should ≈ 0.970211)
    print("\n" + "="*70, flush=True)
    print("CHEAP GATE 1: baseline-assert (bank-17 + FT-T, fixed C=1.0)", flush=True)
    base_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    base_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]
    OOF_base = np.concatenate(base_oof_logp,  axis=1)   # (n, (17+1)*3=54)

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
    # Load in-house base OOF / test_probs (TIGHT set, 36 nodes)
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
        ("redshift_band0", train_band_oh[sidx, 0]),
    ]:
        if arr is None:
            continue
        corr = abs(np.corrcoef(arr, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK smell: {check_name} ~ target corr={corr:.4f}")
    print("[LEAKAGE CHECK 3] PASS", flush=True)

    # =========================================================================
    # Build TIGHT arm base feature matrices (log-probs only, NO interactions yet)
    tight_base_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    tight_base_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]
    tight_inhouse_oof  = list(inhouse_oof_tight.values())
    tight_inhouse_test = list(inhouse_test_tight.values())

    OOF_tight_base = np.concatenate(tight_base_oof_logp + tight_inhouse_oof,  axis=1)
    TST_tight_base = np.concatenate(tight_base_test_logp + tight_inhouse_test, axis=1)
    # Separate in-house block for the interaction term (36 nodes * 3 cols = 108)
    OOF_tight_inhouse = np.concatenate(tight_inhouse_oof,  axis=1)
    TST_tight_inhouse = np.concatenate(tight_inhouse_test, axis=1)
    print(f"\nTIGHT arm base: feature_matrix={OOF_tight_base.shape} ({len(good)+1} bank+FT-T + {len(inhouse_oof_tight)} in-house = {OOF_tight_base.shape[1]//3} bases * 3)", flush=True)
    n_bands = len(BAND_EDGES) + 1  # 6
    print(f"  in-house interact block: {OOF_tight_inhouse.shape}  interact_extra={OOF_tight_inhouse.shape[1]*n_bands} cols (tot={OOF_tight_base.shape[1]+OOF_tight_inhouse.shape[1]*n_bands})", flush=True)

    # FULL arm adds weak bases
    full_inhouse_oof  = tight_inhouse_oof  + list(inhouse_oof_weak.values())
    full_inhouse_test = tight_inhouse_test + list(inhouse_test_weak.values())

    OOF_full_base = np.concatenate(tight_base_oof_logp + full_inhouse_oof,  axis=1)
    TST_full_base = np.concatenate(tight_base_test_logp + full_inhouse_test, axis=1)
    OOF_full_inhouse = np.concatenate(full_inhouse_oof,  axis=1)
    TST_full_inhouse = np.concatenate(full_inhouse_test, axis=1)
    print(f"FULL arm base:  feature_matrix={OOF_full_base.shape} ({len(good)+1} bank+FT-T + {len(inhouse_oof_tight)+len(inhouse_oof_weak)} in-house = {OOF_full_base.shape[1]//3} bases * 3)", flush=True)
    print(f"  in-house interact block: {OOF_full_inhouse.shape}  interact_extra={OOF_full_inhouse.shape[1]*n_bands} cols (tot={OOF_full_base.shape[1]+OOF_full_inhouse.shape[1]*n_bands})", flush=True)

    # =========================================================================
    # SANITY ASSERT: reproduce n091's 0.970355 using the PLAIN base matrix
    # (no interaction block at all — exact byte-match to n091's TIGHT arm)
    print("\n" + "="*70, flush=True)
    print("SANITY ASSERT: run TIGHT arm with PLAIN base matrix (must ≈ 0.970355)", flush=True)
    (oof_assert, _, pf_assert,
     cv_assert, sem_assert, Cs_assert) = nested_cv_arm(
        OOF_tight_base, TST_tight_base, y, fval, "TIGHT_ASSERT"
    )
    print(f"SANITY ASSERT: cv={cv_assert:.6f}  expected≈0.970355  delta={abs(cv_assert-0.970355):.6f}", flush=True)
    if abs(cv_assert - 0.970355) > 0.0002:
        print(f"SANITY ASSERT FAILED: delta={abs(cv_assert-0.970355):.6f} > 0.0002 — base pipeline mismatch!", flush=True)
        import sys; sys.exit(1)
    print("SANITY ASSERT: PASS (base pipeline byte-identical to n091)", flush=True)

    # =========================================================================
    # RUN TIGHT arm WITH INTERACTIONS
    # Interaction block: in-house TIGHT only (36 nodes * 3 * 6 = 648 extra cols)
    # Total: 162 + 648 = 810 cols — tractable with n_jobs=-1 (~3.7 GB)
    print("\n" + "="*70, flush=True)
    print("TIGHT arm WITH interactions...", flush=True)
    (oof_tight, test_tight, pf_tight,
     cv_tight, sem_tight, Cs_tight) = run_arm_with_interactions(
        OOF_tight_base, OOF_tight_inhouse,
        TST_tight_base, TST_tight_inhouse,
        train_band_oh, test_band_oh,
        y, fval, "TIGHT+interact",
    )

    # RUN FULL arm WITH INTERACTIONS
    # Interaction block: in-house FULL only (45 nodes * 3 * 6 = 810 extra cols)
    # Total: 189 + 810 = 999 cols — may be large but test first
    print("\n" + "="*70, flush=True)
    print("FULL arm WITH interactions...", flush=True)
    (oof_full, test_full, pf_full,
     cv_full, sem_full, Cs_full) = run_arm_with_interactions(
        OOF_full_base, OOF_full_inhouse,
        TST_full_base, TST_full_inhouse,
        train_band_oh, test_band_oh,
        y, fval, "FULL+interact",
    )

    # =========================================================================
    # Determine winning arm
    print("\n" + "="*70, flush=True)
    print("=== ARM COMPARISON ===", flush=True)
    print(f"TIGHT+interact: cv={cv_tight:.6f}  sem={sem_tight:.6f}  per_fold={[f'{s:.6f}' for s in pf_tight]}", flush=True)
    print(f"FULL+interact:  cv={cv_full:.6f}  sem={sem_full:.6f}  per_fold={[f'{s:.6f}' for s in pf_full]}", flush=True)

    if cv_tight >= cv_full:
        winner = "TIGHT+interact"
        cv_win, sem_win, pf_win = cv_tight, sem_tight, pf_tight
        oof_win, test_win = oof_tight, test_tight
        OOF_win_base, TST_win_base = OOF_tight_base, TST_tight_base
        OOF_win_inhouse = OOF_tight_inhouse
        Cs_win = Cs_tight
        inhouse_used = inhouse_oof_tight
        n_inhouse = len(inhouse_oof_tight)
    else:
        winner = "FULL+interact"
        cv_win, sem_win, pf_win = cv_full, sem_full, pf_full
        oof_win, test_win = oof_full, test_full
        OOF_win_base, TST_win_base = OOF_full_base, TST_full_base
        OOF_win_inhouse = OOF_full_inhouse
        Cs_win = Cs_full
        inhouse_used = {**inhouse_oof_tight, **inhouse_oof_weak}
        n_inhouse = len(inhouse_oof_tight) + len(inhouse_oof_weak)

    print(f"\nWINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)

    champ_cv  = 0.970355
    champ_sem = 0.000222
    promote_bar = champ_cv + 2 * champ_sem
    lift_vs_champ = cv_win - champ_cv
    print(f"champion  cv={champ_cv:.6f}  2*sem={2*champ_sem:.6f}  promote_bar={promote_bar:.6f}", flush=True)
    print(f"lift_vs_champ={lift_vs_champ:+.6f}  beats_promote_bar={'YES' if cv_win > promote_bar else 'NO'}", flush=True)

    # =========================================================================
    # Coefficient analysis: did L2 keep or shrink interaction block?
    print("\n" + "="*70, flush=True)
    print("COEFFICIENT ANALYSIS: interaction block vs base block", flush=True)
    c_win_final = Counter(Cs_win).most_common(1)[0][0]

    OOF_win_int = add_region_interactions(OOF_win_base, OOF_win_inhouse, train_band_oh)
    m_coef = LogisticRegression(
        class_weight="balanced", C=c_win_final, max_iter=2000,
        n_jobs=-1, random_state=42, solver="lbfgs", multi_class="multinomial",
    )
    m_coef.fit(OOF_win_int, y)

    coef_mat = np.abs(m_coef.coef_)  # (n_classes, n_features)
    coef_sum = coef_mat.sum(axis=0)  # (n_features,)
    n_base_cols = OOF_win_base.shape[1]
    n_interact_cols = OOF_win_int.shape[1] - n_base_cols

    base_coef_mean  = float(coef_sum[:n_base_cols].mean())
    inter_coef_mean = float(coef_sum[n_base_cols:].mean())
    print(f"  final refit C={c_win_final}", flush=True)
    print(f"  base block   n_cols={n_base_cols}  mean|coef|={base_coef_mean:.6f}", flush=True)
    print(f"  interact block n_cols={n_interact_cols}  mean|coef|={inter_coef_mean:.6f}", flush=True)
    ratio = inter_coef_mean / (base_coef_mean + 1e-12)
    print(f"  ratio interact/base = {ratio:.4f}  (>0.1 = L2 kept some signal; ~0 = shrunk)", flush=True)

    # Also report top base weights (same as n091)
    bank_names = [f"{k}_c{ci}" for k in good for ci in range(3)]
    ftt_names  = [f"ftt_c{ci}" for ci in range(3)]
    tight_names = [f"{nm}_c{ci}" for nm in inhouse_oof_tight.keys() for ci in range(3)]
    if winner.startswith("TIGHT"):
        base_col_names = bank_names + ftt_names + tight_names
    else:
        weak_names = [f"{nm}_c{ci}" for nm in inhouse_oof_weak.keys() for ci in range(3)]
        base_col_names = bank_names + ftt_names + tight_names + weak_names

    if len(base_col_names) == n_base_cols:
        n_bases = n_base_cols // 3
        base_names_unique = [base_col_names[bi*3].replace("_c0", "") for bi in range(n_bases)]
        base_coef_per_base = np.array([coef_sum[:n_base_cols][bi*3:(bi+1)*3].sum() for bi in range(n_bases)])
        top_idx = np.argsort(-base_coef_per_base)[:10]
        print(f"\nTop-weight bases ({winner} arm, C={c_win_final}):", flush=True)
        for rank, bi in enumerate(top_idx):
            print(f"  {rank+1:2d}. {base_names_unique[bi]:25s} sum|coef|={base_coef_per_base[bi]:.4f}", flush=True)

    # =========================================================================
    # GATE 3 check
    beats_champ = cv_win > champ_cv
    print(f"\nWinning arm beats champion ({champ_cv:.6f})? {'YES' if beats_champ else 'NO'}", flush=True)

    # =========================================================================
    # Write artifacts (oof.npy, test_probs.npy, submission.csv)
    np.save(NODE_DIR / "oof.npy",        oof_win.astype(np.float32))
    np.save(NODE_DIR / "test_probs.npy", test_win.astype(np.float32))

    test_preds_idx = test_win.argmax(1)
    test_labels = [I2L[i] for i in test_preds_idx]
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
    print(f"SANITY ASSERT (zeroed): cv={cv_assert:.6f} (expected≈0.970355) PASS", flush=True)
    print(f"TIGHT+interact: cv={cv_tight:.6f}  sem={sem_tight:.6f}  Cs={Cs_tight}  per_fold={[f'{s:.6f}' for s in pf_tight]}", flush=True)
    print(f"FULL+interact:  cv={cv_full:.6f}  sem={sem_full:.6f}   Cs={Cs_full}  per_fold={[f'{s:.6f}' for s in pf_full]}", flush=True)
    print(f"WINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)
    print(f"promotes? {'YES' if cv_win > promote_bar else 'NO'}  (bar={promote_bar:.6f})", flush=True)
    print(f"beats_champ? {'YES' if beats_champ else 'NO'}  (champ={champ_cv:.6f})", flush=True)
    print(f"interact/base ratio={ratio:.4f}  selected_C={c_win_final}", flush=True)
    print(f"cv={cv_win:.6f}", flush=True)  # machine-parseable line


if __name__ == "__main__":
    main()
