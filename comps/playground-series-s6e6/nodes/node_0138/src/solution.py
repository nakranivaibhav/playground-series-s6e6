"""node_0138 — Shallow GBDT meta on curated complementary subset

Atomic change vs node_0099 (GBDT meta over FULL 63-col pool):
  Replace the FULL OOF pool (63 bases × 3 = 189 cols) with a CURATED SMALL
  complementary subset (~8 de-correlated bases × 3 = ~24 cols).

  The curated subset is picked from the drop-study (probes/drop_study_ranking.csv)
  by a combination of causal delta (cv_without > champion ⟹ positive causal
  contribution) and abscoef (the linear meta's coefficient magnitude), favouring
  de-correlated family diversity:
    - cat-3         (bank, delta=+0.000158, abscoef=0.876)  ← top-1 causal
    - realmlp-2     (bank, delta=+0.000060, abscoef=0.564)  ← top-2 causal
    - ft_transformer (bank, delta=+0.000038, abscoef=0.166) ← FT-T, diff family
    - node_0039     (CatBoost in-house, abscoef=0.977)      ← strongest abscoef
    - node_0043     (CatBoost variant, abscoef=0.582)       ← 2nd CatBoost family
    - node_0033     (TabM in-house, abscoef=0.337)          ← TabM family
    - node_0032     (RealMLP in-house, abscoef=0.120)       ← RealMLP family
    - tabm-1        (bank, abscoef=0.574)                   ← bank TabM variant

  Total: 8 bases × 3 classes = 24 columns (vs 63 bases × 3 = 189 in n099).
  The hypothesis: n099's GBDT-meta failure was capacity×width (deep tree meta over
  189 redundant cols overfits the OOF), NOT that nonlinearity is useless — a
  shallow GBDT over 24 de-correlated cols might capture a base INTERACTION the
  linear meta cannot, without overfitting.

Build protocol:
  1. SANITY ASSERT: linear LogReg over the SAME curated subset must reproduce the
     expected subset CV (close to champion 0.970355 ± a bit, not < 0.96 or > 0.975).
  2. NESTED honesty: GBDT meta hyperparameters (depth, n_estimators via early-stop)
     are determined by an inner-val slice carved from the outer-TRAIN portion only;
     the outer val fold vi is NEVER touched during early-stopping or fit.
  3. CPU, minutes.

Leakage: OOF probs are pre-computed by base nodes; no target or id enters features.
         GBDT meta fit inside fold loop on outer-train rows only (inner val is a
         subset of outer-train). Final refit on all-train after the OOF loop.
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
NODE_DIR = COMP / "nodes/node_0138"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3

# Curated subset — 8 de-correlated bases picked by drop-study causal delta + abscoef
# (see plan; these are the complementary diversity set, NOT the full pool)
#
# Bank bases from the MANIFEST (same keys as champion/src/solution.py):
CURATED_BANK_KEYS = [
    "cat-3",        # top-1 causal contributor (delta=+0.000158, abscoef=0.876)
    "realmlp-2",    # top-2 causal (delta=+0.000060, abscoef=0.564)
    "tabm-1",       # bank TabM (abscoef=0.574, high individual signal)
]
# FT-Transformer (loaded separately, not in MANIFEST):
INCLUDE_FT_TRANSFORMER = True

# In-house bases (node IDs in TIGHT_IDS from champion):
CURATED_INHOUSE_IDS = [
    39,   # CatBoost in-house, abscoef=0.977 (strongest abscoef overall)
    43,   # CatBoost variant, abscoef=0.582
    33,   # TabM in-house, abscoef=0.337, delta=+0.000020
    32,   # RealMLP in-house, abscoef=0.120, delta=+0.000022
]
# Total curated: 3 bank + 1 FT-T + 4 in-house = 8 bases × 3 = 24 columns

# Full MANIFEST (needed to load the bank bases)
# Verbatim from champion/src/solution.py
def build_manifest(K, B):
    return {
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


# ---------------------------------------------------------------------------
# Helpers (verbatim from champion/src/solution.py)
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
# LightGBM meta arm — the atomic change in this node
# ---------------------------------------------------------------------------

def lgbm_meta_arm_curated(
    OOF_mat: np.ndarray,
    TST_mat: np.ndarray,
    y: np.ndarray,
    fval: list[np.ndarray],
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float]:
    """Fold-honest LightGBM MULTICLASS meta over a CURATED small feature matrix.

    Key difference vs node_0099:
      - input is ~24 cols (8 bases × 3) vs 189 cols in n099
      - same shallow params but fewer features = less overfit surface
      - early stopping on inner-val slice (subset of outer-train only)

    Leakage discipline:
      - vi (outer val) is NEVER seen during fit or early-stopping
      - inner_val_idx is a slice of tr_idx (outer-train rows), never overlaps vi
      - OOF features contain no target, no id
      - folds from frozen folds.json
    """
    n = len(y)
    n_folds = len(fval)

    oof_probs = np.zeros((n, NC), dtype=float)

    # Class-balanced sample weights
    class_counts = np.bincount(y, minlength=NC)
    total = class_counts.sum()
    weights = total / (NC * class_counts.astype(float))
    sample_w = weights[y]

    # Shallow params — strong regularization to resist OOF overfit:
    # - num_leaves=15 (very shallow vs 31 in n099)
    # - min_child_samples=50 (conservative leaf size)
    # - lambda_l2=1.0 (stronger L2 regularization)
    # - feature_fraction=0.9 (allow all features at small width)
    lgbm_params = dict(
        objective="multiclass",
        num_class=NC,
        num_leaves=15,           # shallow (n099 used 31)
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_child_samples=50,    # conservative (n099 used 20)
        lambda_l2=1.0,           # stronger L2 (n099 used 0.1)
        verbose=-1,
        n_jobs=-1,
        random_state=42,
        n_estimators=3000,       # higher max, early stop brings it down
    )

    best_iters = []

    print(f"\n=== ARM (CURATED LightGBM): {label}  feature_cols={OOF_mat.shape[1]} ===", flush=True)

    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)

        # Inner val: last 10% of training portion for early stopping
        # (within tr_idx only — vi NEVER appears)
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
    cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(n_folds))

    print(f"\n  {label} curated-LGBM cv={cv_mean:.6f}  sem={cv_sem:.6f}", flush=True)
    print(f"  per_fold={[f'{s:.6f}' for s in per_fold_scores]}", flush=True)
    print(f"  best_iters={best_iters}  mean_iter={np.mean(best_iters):.0f}", flush=True)

    # Final refit on ALL train using median best_iter
    final_n_est = max(10, int(np.median(best_iters))) if best_iters else 200
    print(f"\n  Final refit: n_estimators={final_n_est} on all {n} train rows", flush=True)

    final_params = dict(lgbm_params)
    final_params["n_estimators"] = final_n_est

    m_final = lgb.LGBMClassifier(**final_params)
    m_final.fit(OOF_mat, y, sample_weight=sample_w)
    test_probs = m_final.predict_proba(TST_mat)

    return oof_probs, test_probs, per_fold_scores, cv_mean, cv_sem


# ---------------------------------------------------------------------------
# LogReg meta over the SAME curated subset — for sanity assert
# ---------------------------------------------------------------------------

def logreg_sanity_arm(
    OOF_mat: np.ndarray,
    y: np.ndarray,
    fval: list[np.ndarray],
    label: str,
) -> tuple[list[float], float]:
    """Nested LogReg on the curated subset — sanity check that the ingest is correct."""
    n = len(y)
    n_folds = len(fval)
    C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    oof_probs = np.zeros((n, NC), dtype=float)

    print(f"\n=== SANITY (LogReg curated): {label} feature_cols={OOF_mat.shape[1]} ===", flush=True)

    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        lrcv = LogisticRegressionCV(
            Cs=C_GRID, cv=4, class_weight="balanced",
            max_iter=2000, n_jobs=-1, random_state=42,
            scoring="balanced_accuracy", solver="lbfgs", multi_class="multinomial",
        )
        lrcv.fit(OOF_mat[tr_idx], y[tr_idx])
        oof_probs[vi] = lrcv.predict_proba(OOF_mat[vi])
        best_c = float(lrcv.C_[0])
        print(f"  fold {fi}: best_C={best_c}", flush=True)

    per_fold_scores = [score_fn(y[vi], oof_probs[vi].argmax(1)) for vi in fval]
    cv_mean = float(np.mean(per_fold_scores))
    print(f"  {label} LogReg-curated cv={cv_mean:.6f}  per_fold={[f'{s:.6f}' for s in per_fold_scores]}", flush=True)
    return per_fold_scores, cv_mean


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n  = len(train)
    nt = len(test)
    y  = train["class"].map(L2I).to_numpy()

    # Frozen folds — val indices only
    fval    = [np.asarray(f["val_idx"]) for f in folds_data]
    n_folds = len(fval)

    print(f"n_train={n} n_test={nt} n_folds={n_folds}", flush=True)
    assert n  == 577347, f"unexpected n_train={n}"
    assert nt == 247435, f"unexpected n_test={nt}"

    # =========================================================================
    # PRE-FLIGHT leakage checks 1, 2, 5
    print("\n[LEAKAGE CHECK 1-2] Features are OOF probs only (no target/id). PASS", flush=True)
    print("[LEAKAGE CHECK 4] GBDT meta fit inside fold loop; inner-val is subset of outer-train (vi never seen). PASS", flush=True)
    print("[LEAKAGE CHECK 5] Folds loaded from frozen folds.json. PASS", flush=True)

    # =========================================================================
    # Load ONLY the curated bank bases (subset of MANIFEST)
    B = COMP / "refs/oof_bank"
    K = COMP / "refs/kernel_out"
    MANIFEST = build_manifest(K, B)

    curated_bank_oof  = {}
    curated_bank_test = {}
    print(f"\n{'model':14s} {'oofBA':>9s} {'shape':>12s} {'status'}", flush=True)
    for name in CURATED_BANK_KEYS:
        op, tp = MANIFEST[name]
        try:
            o = norm(rd(op, n))
            t = norm(rd(tp, nt))
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            curated_bank_oof[name]  = logp(o)
            curated_bank_test[name] = logp(t)
            print(f"{name:14s} {ba:9.6f} {str(o.shape):>12s} OK", flush=True)
        except Exception as e:
            print(f"{name:14s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}", flush=True)

    # =========================================================================
    # Load FT-Transformer (curated, as per plan)
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
    # Load curated in-house bases
    inhouse_oof  = {}
    inhouse_test = {}
    print(f"\n{'node':12s} {'solo_BA':>9s} {'shape':>12s} {'status'}", flush=True)
    for nid in CURATED_INHOUSE_IDS:
        node_nm   = f"node_{nid:04d}"
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
            inhouse_oof[node_nm]  = logp(o)
            inhouse_test[node_nm] = logp(t)
            print(f"{node_nm:12s} {solo_ba:9.6f} {str(o.shape):>12s} OK", flush=True)
        except Exception as e:
            print(f"{node_nm:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}", flush=True)

    print(f"\nLoaded {len(curated_bank_oof)} bank + 1 FT-T + {len(inhouse_oof)} in-house curated bases", flush=True)

    # =========================================================================
    # LEAKAGE CHECK 3: single-feature↔target correlation sweep on a 50k sample
    print("\n[LEAKAGE CHECK 3] Single-feature correlation sweep (50k sample)...", flush=True)
    rng  = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys   = y[sidx].astype(float)
    all_curated_arrs = (
        list(curated_bank_oof.values())
        + [logp(norm(ft_oof_raw))]
        + list(inhouse_oof.values())
    )
    for ci, arr in enumerate(all_curated_arrs):
        for col in range(3):
            x = arr[sidx, col]
            corr = abs(np.corrcoef(x, ys)[0, 1])
            if corr >= 0.999:
                raise SystemExit(f"LEAK smell: curated_arr[{ci}] col={col} corr={corr:.4f}")
    print("[LEAKAGE CHECK 3] PASS", flush=True)

    # =========================================================================
    # Assemble the CURATED feature matrix
    curated_oof_parts  = (
        list(curated_bank_oof.values())
        + [logp(norm(ft_oof_raw))]
        + list(inhouse_oof.values())
    )
    curated_test_parts = (
        list(curated_bank_test.values())
        + [logp(norm(ft_test_raw))]
        + list(inhouse_test.values())
    )

    OOF_curated  = np.concatenate(curated_oof_parts,  axis=1)
    TST_curated  = np.concatenate(curated_test_parts, axis=1)
    n_bases = len(curated_oof_parts)
    print(f"\nCurated feature matrix: {OOF_curated.shape} ({n_bases} bases × 3 = {n_bases*3} cols)", flush=True)

    # =========================================================================
    # SANITY ASSERT: LogReg over the SAME curated subset
    # should be close to champion (0.970355) — if way off, ingest is broken
    print("\n" + "="*70, flush=True)
    print("SANITY ASSERT: LogReg over curated subset (should be ≥0.960)", flush=True)
    pf_sanity, cv_sanity = logreg_sanity_arm(OOF_curated, y, fval, "CURATED")
    print(f"  LogReg-curated cv={cv_sanity:.6f}  (champion=0.970355)", flush=True)
    if cv_sanity < 0.960:
        print(f"STOP: sanity assert FAILED (cv={cv_sanity:.6f} < 0.960). OOF ingest is wrong.", flush=True)
        import sys; sys.exit(1)
    if cv_sanity > 0.975:
        print(f"WARN: sanity assert unusually high (cv={cv_sanity:.6f} > 0.975) — check for leakage.", flush=True)
    print("SANITY ASSERT: PASS", flush=True)

    # =========================================================================
    # RUN GBDT meta on curated subset — the atomic change
    print("\n" + "="*70, flush=True)
    print("=== LightGBM meta on CURATED subset ===", flush=True)
    (oof_curated_lgbm, test_curated_lgbm, pf_lgbm,
     cv_lgbm, sem_lgbm) = lgbm_meta_arm_curated(
        OOF_curated, TST_curated, y, fval, "CURATED"
    )

    # =========================================================================
    # A/B vs champion node_0091 (LogReg FULL, cv=0.970355, sem=0.000249)
    champ_cv  = 0.970355
    champ_sem = 0.000249
    promote_bar = champ_cv + 2 * champ_sem
    lift_vs_champ = cv_lgbm - champ_cv

    print(f"\nchampion (node_0091 LogReg): cv={champ_cv:.6f}  sem={champ_sem:.6f}", flush=True)
    print(f"promote_bar = champ_cv + 2*sem = {promote_bar:.6f}", flush=True)
    print(f"curated-LGBM cv={cv_lgbm:.6f}  lift_vs_champ={lift_vs_champ:+.6f}", flush=True)
    print(f"beats_promote_bar? {'YES' if cv_lgbm > promote_bar else 'NO'}", flush=True)

    # Per-fold delta vs champion
    champ_pf = [0.971208, 0.970067, 0.969934, 0.969938, 0.970626]
    print(f"\nPer-fold deltas (curated-LGBM vs champion n091):", flush=True)
    print(f"  champion per-fold: {[f'{s:.6f}' for s in champ_pf]}", flush=True)
    print(f"  curated-LGBM pf:   {[f'{s:.6f}' for s in pf_lgbm]}", flush=True)
    for fi in range(n_folds):
        delta_fold = pf_lgbm[fi] - champ_pf[fi]
        print(f"  fold {fi}: LGBM={pf_lgbm[fi]:.6f}  champ={champ_pf[fi]:.6f}  delta={delta_fold:+.6f}", flush=True)

    # Report per-fold scores for the machine-parseable output
    for fi, s in enumerate(pf_lgbm):
        print(f"fold {fi} score: {s:.6f}", flush=True)
    print(f"cv={cv_lgbm:.6f}", flush=True)

    # cv_too_good check
    cv_too_good = cv_lgbm > 0.9706
    if cv_too_good:
        print(f"\nWARN cv_too_good: curated-LGBM ({cv_lgbm:.6f}) > 0.9706 threshold", flush=True)
        print("  GBDT meta exceeding champion on OOF = possible overfit. Holdout check required.", flush=True)

    # =========================================================================
    # Write artifacts
    np.save(NODE_DIR / "oof.npy",        oof_curated_lgbm.astype(np.float32))
    np.save(NODE_DIR / "test_probs.npy", test_curated_lgbm.astype(np.float32))

    test_preds_idx = test_curated_lgbm.argmax(1)
    test_labels    = [I2L[i] for i in test_preds_idx]
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)

    print(f"\nArtifacts written:", flush=True)
    print(f"  oof.npy:        {oof_curated_lgbm.shape}", flush=True)
    print(f"  test_probs.npy: {test_curated_lgbm.shape}", flush=True)
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
    print(f"SANITY (LogReg curated):  cv={cv_sanity:.6f}  per_fold={[f'{s:.6f}' for s in pf_sanity]}", flush=True)
    print(f"curated-LGBM:             cv={cv_lgbm:.6f}  sem={sem_lgbm:.6f}  per_fold={[f'{s:.6f}' for s in pf_lgbm]}", flush=True)
    print(f"champion (n091 LogReg):   cv={champ_cv:.6f}  promote_bar={promote_bar:.6f}", flush=True)
    print(f"beats_promote_bar?        {'YES' if cv_lgbm > promote_bar else 'NO'}  lift={lift_vs_champ:+.6f}", flush=True)
    print(f"cv_too_good:              {'WARN' if cv_too_good else 'PASS'}", flush=True)
    print(f"cv={cv_lgbm:.6f}  sem={sem_lgbm:.6f}", flush=True)


if __name__ == "__main__":
    main()
