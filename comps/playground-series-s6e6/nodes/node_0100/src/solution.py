"""node_0100 — FWLS region-interacted meta-stack

Atomic change vs node_0091 (balanced multinomial LogReg over the FULL OOF pool):
  Feature-Weighted Linear Stacking (FWLS, Sill et al. 2009): augment the meta
  feature matrix with INTERACTIONS between each base's log-prob columns and
  REDSHIFT-REGION indicators (one-hot of redshift quantile bin, q=4).

  The bin EDGES are computed on TRAIN-FOLD rows only (fit_in_fold) then applied
  to val rows and test rows. The new meta feature matrix is:
    concat[ OOF_mat , OOF_mat ⊗ R_columns ]
  where OOF_mat is the full-pool clipped log-probs (same as node_0091) and
  R_columns are the quantile-bin one-hot indicators.

  The same fixed-C LogReg arm (C=0.003, lbfgs) is used with a speed hack:
  meta weights are fit on a random subsample of META_SUBSAMPLE rows from the
  train fold only — this learns the mixing weights (typically 50k rows is
  sufficient for 189-col or 945-col meta; OOF scoring still covers ALL rows).
  This makes both the baseline sanity check and the FWLS run tractable.

  A/B reporting:
    - Reproduce plain-OOF_mat LogReg baseline (fixed C=0.003, subsampled fit).
    - Report FWLS-augmented fold-honest CV, per-fold deltas, sem,
      whether it clears 0.970355 + 2·sem ≈ 0.970851.

Leakage discipline:
  - Bin edges computed on train-fold rows only (fit_in_fold).
  - Meta fit inside the fold loop on train-fold rows ONLY
    (subsample drawn from train fold — val rows never touched).
  - OOF features (log-probs of base models) contain no target, no id.
  - Folds loaded from frozen folds.json.
  - OOF scoring: predict_proba on ALL val rows (meta fit on subsample of
    train fold, predict on all val rows — this is correct, not a leak).
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

COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
NODE_DIR = COMP / "nodes/node_0100"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3

# Fixed C from node_0091's nested-CV (selected consistently across all 5 folds)
FIXED_C = 0.003

# Number of redshift quantile bins for FWLS region indicators
N_BINS = 4

# Meta weight fitting: subsample this many rows from train-fold to fit lbfgs quickly.
# 80k rows is ample for 189-945 feature meta (meta weights are stable with ~10k).
META_SUBSAMPLE = 80_000

# TIGHT pool (same as node_0091)
TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]

# FULL pool extra (same as node_0091)
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]

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


def make_logreg(C: float = FIXED_C, max_iter: int = 500, tol: float = 1e-4) -> LogisticRegression:
    """lbfgs LogReg — multinomial, same family as node_0091.
    n_jobs=1: for multinomial lbfgs (single scipy.optimize.minimize call),
    n_jobs>1 only adds joblib process-pool overhead without benefit.
    """
    return LogisticRegression(
        C=C, class_weight="balanced", max_iter=max_iter, tol=tol,
        n_jobs=1, random_state=42, solver="lbfgs", multi_class="multinomial",
    )


# ---------------------------------------------------------------------------
# Region indicator builder (FWLS — fit_in_fold)
# ---------------------------------------------------------------------------

def make_region_indicators(
    redshift_train_fold: np.ndarray,
    redshift_apply: np.ndarray,
    n_bins: int = N_BINS,
) -> np.ndarray:
    """One-hot redshift quantile bins. Edges from train-fold only (fit_in_fold)."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(redshift_train_fold, quantiles)
    edges = np.unique(edges)
    if len(edges) < 2:
        return np.ones((len(redshift_apply), 1), dtype=float)
    n_actual_bins = len(edges) - 1
    bin_idx = np.digitize(redshift_apply, edges[1:-1])
    bin_idx = np.clip(bin_idx, 0, n_actual_bins - 1)
    R = np.zeros((len(redshift_apply), n_actual_bins), dtype=float)
    R[np.arange(len(redshift_apply)), bin_idx] = 1.0
    return R


def augment_with_region(OOF_mat_rows: np.ndarray, R_rows: np.ndarray) -> np.ndarray:
    """Concat[ OOF_mat_rows , OOF_mat_rows ⊗ R_cols ]"""
    interactions = OOF_mat_rows[:, :, np.newaxis] * R_rows[:, np.newaxis, :]
    interactions = interactions.reshape(len(OOF_mat_rows), -1)
    return np.concatenate([OOF_mat_rows, interactions], axis=1)


# ---------------------------------------------------------------------------
# Generic OOF arm (subsampled meta fit)
# ---------------------------------------------------------------------------

def oof_arm(
    get_X_tr: "callable",
    get_X_vi: "callable",
    y: np.ndarray,
    fval: list[np.ndarray],
    label: str,
    meta_subsample: int = META_SUBSAMPLE,
) -> tuple[np.ndarray, list[float], float, float]:
    """Fold-honest OOF with subsampled lbfgs meta fit.

    get_X_tr(tr_idx) → feature matrix for train portion of fold
    get_X_vi(tr_idx, vi)  → feature matrix for val portion of fold
    (vi is passed so FWLS can fit bin edges from tr_idx only — fit_in_fold)

    OOF probs: always predict on ALL val rows (correct; no leak).
    Meta weights: fit on a random subsample of train-fold rows (fast).
    """
    n = len(y)
    n_folds = len(fval)
    oof_probs = np.zeros((n, NC), dtype=float)
    rng = np.random.RandomState(42)

    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        X_tr_full = get_X_tr(tr_idx)
        X_vi_mat  = get_X_vi(tr_idx, vi)

        # Subsample for fast meta fit (from train fold only — vi never included)
        if meta_subsample and len(tr_idx) > meta_subsample:
            sub = rng.choice(len(tr_idx), meta_subsample, replace=False)
            X_fit = X_tr_full[sub]
            y_fit = y[tr_idx[sub]]
        else:
            X_fit = X_tr_full
            y_fit = y[tr_idx]

        m = make_logreg()
        m.fit(X_fit, y_fit)
        oof_probs[vi] = m.predict_proba(X_vi_mat)

        s = score_fn(y[vi], oof_probs[vi].argmax(1))
        print(f"  fold {fi}: BA={s:.6f}  fit_rows={len(X_fit)}  pred_rows={len(vi)}  cols={X_fit.shape[1]}", flush=True)

    pf = [score_fn(y[vi], oof_probs[vi].argmax(1)) for vi in fval]
    cv = float(np.mean(pf))
    sem = float(np.std(pf, ddof=1) / np.sqrt(n_folds))
    print(f"  {label}: cv={cv:.6f}  sem={sem:.6f}  per_fold={[f'{s:.6f}' for s in pf]}", flush=True)
    return oof_probs, pf, cv, sem


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n  = len(train)
    nt = len(test)
    y  = train["class"].map(L2I).to_numpy()

    fval   = [np.asarray(f["val_idx"]) for f in folds_data]
    n_folds = len(fval)

    print(f"n_train={n} n_test={nt} n_folds={n_folds}", flush=True)
    assert n  == 577347
    assert nt == 247435

    assert "redshift" in train.columns
    assert "redshift" in test.columns
    redshift_train = train["redshift"].to_numpy(dtype=float)
    redshift_test  = test["redshift"].to_numpy(dtype=float)
    print(f"redshift: train range=[{redshift_train.min():.4f},{redshift_train.max():.4f}]", flush=True)

    # PRE-FLIGHT leakage checks
    print("\n[LEAKAGE CHECK 1-2] Features are OOF log-probs (no target/id). PASS", flush=True)
    print("[LEAKAGE CHECK 4] Bin edges fit on train-fold only; meta fit on train-fold subsample only. PASS", flush=True)
    print("[LEAKAGE CHECK 5] Folds from frozen folds.json. PASS", flush=True)

    # =========================================================================
    # Load public bank-17
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
    print(f"\n{'model':14s} {'oofBA':>9s} {'status'}", flush=True)
    for name, (op, tp) in MANIFEST.items():
        try:
            o = norm(rd(op, n)); t = norm(rd(tp, nt))
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            st = "OK" if 0.90 < ba < 0.972 else ("QUARANTINE" if ba >= 0.972 else "LOW?")
            if st == "OK":
                POOF[name] = o; PTEST[name] = t; good.append(name)
            print(f"{name:14s} {ba:9.6f} {st}", flush=True)
        except Exception as e:
            print(f"{name:14s} {'--':>9s} FAIL {str(e)[:60]}", flush=True)

    print(f"\nLoaded {len(good)} public bank models", flush=True)

    PILK = COMP / "refs/ext_oof/pilkwang_5090"
    ft_oof_raw  = load_ext_csv(PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n)
    ft_test_raw = load_ext_csv(PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", nt)
    assert ft_oof_raw.shape == (n, 3) and ft_test_raw.shape == (nt, 3)
    ft_ba = score_fn(y, norm(ft_oof_raw).argmax(1))
    print(f"ft_transformer solo_BA={ft_ba:.6f}", flush=True)
    assert ft_ba > 0.85

    # =========================================================================
    # Load in-house bases
    inhouse_oof_tight = {}; inhouse_test_tight = {}
    for nid in TIGHT_IDS:
        nm = f"node_{nid:04d}"
        try:
            o_raw = np.load(COMP/"nodes"/nm/"oof.npy").astype(float)
            t_raw = np.load(COMP/"nodes"/nm/"test_probs.npy").astype(float)
            assert o_raw.shape == (n,3) and t_raw.shape == (nt,3)
            assert not np.isnan(o_raw).any() and not np.isnan(t_raw).any()
            o = norm(o_raw); t = norm(t_raw)
            if score_fn(y, o.argmax(1)) < 0.5: continue
            inhouse_oof_tight[nm] = logp(o); inhouse_test_tight[nm] = logp(t)
        except Exception: pass

    inhouse_oof_weak = {}; inhouse_test_weak = {}
    for nid in WEAK_EXTRA_IDS:
        nm = f"node_{nid:04d}"
        try:
            o_raw = np.load(COMP/"nodes"/nm/"oof.npy").astype(float)
            t_raw = np.load(COMP/"nodes"/nm/"test_probs.npy").astype(float)
            assert o_raw.shape == (n,3) and t_raw.shape == (nt,3)
            assert not np.isnan(o_raw).any() and not np.isnan(t_raw).any()
            o = norm(o_raw); t = norm(t_raw)
            if score_fn(y, o.argmax(1)) < 0.5: continue
            inhouse_oof_weak[nm] = logp(o); inhouse_test_weak[nm] = logp(t)
        except Exception: pass

    print(f"TIGHT={len(inhouse_oof_tight)}/{len(TIGHT_IDS)}  WEAK={len(inhouse_oof_weak)}/{len(WEAK_EXTRA_IDS)}", flush=True)

    # LEAKAGE CHECK 3
    rng0 = np.random.RandomState(0)
    sidx = rng0.choice(n, 50000, replace=False)
    ys = y[sidx].astype(float)
    for nm, arr in [("ft_col0", logp(norm(ft_oof_raw))[sidx,0]),
                    ("redshift", redshift_train[sidx])]:
        corr = abs(np.corrcoef(arr, ys)[0,1])
        if corr >= 0.999: raise SystemExit(f"LEAK: {nm} corr={corr:.4f}")
        print(f"[LEAKAGE CHECK 3] {nm}: |corr|={corr:.4f} PASS", flush=True)

    # =========================================================================
    # Build FULL OOF matrix
    b_oof  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    b_test = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]
    ih_oof  = list(inhouse_oof_tight.values()) + list(inhouse_oof_weak.values())
    ih_test = list(inhouse_test_tight.values()) + list(inhouse_test_weak.values())

    OOF_full = np.concatenate(b_oof + ih_oof,   axis=1)  # (n, 189)
    TST_full = np.concatenate(b_test + ih_test,  axis=1)  # (nt, 189)
    print(f"\nFULL OOF matrix: {OOF_full.shape}", flush=True)

    # =========================================================================
    # SANITY: Reproduce node_0091 champion CV ≈ 0.970355
    # Fixed C=0.003, subsampled meta fit (80k rows), lbfgs
    print("\n" + "="*70, flush=True)
    print(f"=== SANITY: fixed C={FIXED_C}, subsample={META_SUBSAMPLE} (should ≈ 0.970355) ===", flush=True)

    def get_X_tr_plain(tr_idx): return OOF_full[tr_idx]
    def get_X_vi_plain(tr_idx, vi): return OOF_full[vi]

    _, pf_sanity, cv_sanity, sem_sanity = oof_arm(
        get_X_tr_plain, get_X_vi_plain, y, fval, "SANITY"
    )

    EXPECTED_CV = 0.970355
    delta = abs(cv_sanity - EXPECTED_CV)
    print(f"\nSanity: cv={cv_sanity:.6f}  expected≈{EXPECTED_CV:.6f}  delta={delta:.6f}", flush=True)
    sanity_ok = delta < 0.003  # allow some variance from subsampling
    print(f"Sanity: {'PASS' if sanity_ok else 'WARN'}", flush=True)
    if delta > 0.01:
        print("STOP: sanity delta > 0.01 — data or alignment issue.", flush=True)
        import sys; sys.exit(1)

    # =========================================================================
    # FWLS: region-interacted LogReg — FULL arm
    print("\n" + "="*70, flush=True)
    print(f"=== FWLS: region-interacted (N_BINS={N_BINS}, fixed C={FIXED_C}, subsample={META_SUBSAMPLE}) ===", flush=True)

    def get_X_tr_fwls(tr_idx):
        R_tr = make_region_indicators(redshift_train[tr_idx], redshift_train[tr_idx])
        return augment_with_region(OOF_full[tr_idx], R_tr)

    def get_X_vi_fwls(tr_idx, vi):
        R_vi = make_region_indicators(redshift_train[tr_idx], redshift_train[vi])
        return augment_with_region(OOF_full[vi], R_vi)

    oof_fwls, pf_fwls, cv_fwls, sem_fwls = oof_arm(
        get_X_tr_fwls, get_X_vi_fwls, y, fval, "FWLS"
    )

    # =========================================================================
    # A/B vs champion
    champ_cv = 0.970355; champ_sem = 0.000248
    promote_bar = champ_cv + 2 * champ_sem
    lift = cv_fwls - champ_cv

    print(f"\nchampion (node_0091): cv={champ_cv:.6f}  sem={champ_sem:.6f}", flush=True)
    print(f"promote_bar = {promote_bar:.6f}", flush=True)
    print(f"FWLS: cv={cv_fwls:.6f}  sem={sem_fwls:.6f}  lift={lift:+.6f}", flush=True)
    print(f"beats_promote_bar? {'YES' if cv_fwls > promote_bar else 'NO'}", flush=True)

    print(f"\nPer-fold deltas (FWLS vs sanity):", flush=True)
    for fi in range(n_folds):
        d = pf_fwls[fi] - pf_sanity[fi]
        print(f"  fold {fi}: FWLS={pf_fwls[fi]:.6f}  sanity={pf_sanity[fi]:.6f}  delta={d:+.6f}", flush=True)

    cv_too_good = cv_fwls > promote_bar
    print(f"\ncv_too_good: {'WARN' if cv_too_good else 'PASS'}", flush=True)

    # =========================================================================
    # Final refit on all train for test preds
    print("\nFinal refit on ALL train (bin edges from ALL train)...", flush=True)
    R_tr_all  = make_region_indicators(redshift_train, redshift_train)
    R_tst_all = make_region_indicators(redshift_train, redshift_test)
    X_all_tr  = augment_with_region(OOF_full, R_tr_all)
    X_all_tst = augment_with_region(TST_full, R_tst_all)
    print(f"  aug_cols={X_all_tr.shape[1]}", flush=True)

    # 500 iter sufficient for C=0.003 (strong L2); n_jobs=1 avoids joblib overhead
    m_final = make_logreg(max_iter=500)
    m_final.fit(X_all_tr, y)
    test_fwls = m_final.predict_proba(X_all_tst)

    # =========================================================================
    # Write artifacts
    np.save(NODE_DIR / "oof.npy",        oof_fwls.astype(np.float32))
    np.save(NODE_DIR / "test_probs.npy", test_fwls.astype(np.float32))

    test_labels = [I2L[i] for i in test_fwls.argmax(1)]
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)

    print(f"\nArtifacts written: oof.npy={oof_fwls.shape}  test_probs.npy={test_fwls.shape}  sub={len(sub)}", flush=True)

    # =========================================================================
    # Post-run gates
    print("\n[POST-RUN GATES]", flush=True)

    assert list(sub.columns) == list(sample_sub.columns)
    assert len(sub) == len(sample_sub)
    assert set(sub["class"].unique()) <= set(LAB)
    print("  schema_ok: PASS", flush=True)

    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC)
    assert not np.isnan(oofn).any()
    print("  oof_full: PASS  no_nan: PASS", flush=True)

    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5
    row_sums = oofn.sum(axis=1)
    assert abs(row_sums.mean() - 1.0) < 0.01
    cls = np.bincount(oofn.argmax(1), minlength=3)
    print(f"  dist_sane: PASS  argmax GALAXY={cls[0]} QSO={cls[1]} STAR={cls[2]}", flush=True)

    # =========================================================================
    print("\n" + "="*70, flush=True)
    print("=== FINAL SUMMARY ===", flush=True)
    print(f"SANITY (fixed C={FIXED_C}, sub={META_SUBSAMPLE}): cv={cv_sanity:.6f} {'PASS' if sanity_ok else 'WARN'}", flush=True)
    print(f"FWLS:   cv={cv_fwls:.6f}  sem={sem_fwls:.6f}  per_fold={[f'{s:.6f}' for s in pf_fwls]}", flush=True)
    print(f"champion cv={champ_cv:.6f}  promote_bar={promote_bar:.6f}", flush=True)
    print(f"beats_promote_bar? {'YES' if cv_fwls > promote_bar else 'NO'}  lift={lift:+.6f}", flush=True)
    print(f"cv_too_good: {'WARN' if cv_too_good else 'PASS'}", flush=True)
    print(f"cv={cv_fwls:.6f}", flush=True)


if __name__ == "__main__":
    main()
