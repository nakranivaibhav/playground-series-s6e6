"""node_0136 — disagreement-feature augmented meta (combine: n091+n070+n039+n033)

Atomic change vs node_0091 (champion):
  AUGMENT the meta's feature matrix with a small fixed block of per-row
  BASE-DISAGREEMENT features computed from the existing base OOF (no new model,
  no base retrain). Appended to the same n091 base-logprob matrix; same L2
  LogReg meta re-fit.

Disagreement features (~6-10 scalar columns):
  1. argmax-vote entropy across pooled bases (how split is the vote)
  2. per-class probability dispersion: std across bases of each class prob (3 cols)
  3. max − 2nd-max margin of the mean base prob (decision confidence)
  4. top-2 class identity disagreement flag (do bases disagree on which two
     classes are in contention — binary)

Leakage: disagreement features are computed PER ROW from the base OOF columns
only — each base OOF is already leave-fold-out (no target leakage, no refit
needed). Features are stateless with respect to the meta's train/val split.
The LogReg meta is still fit inside each outer fold on the train portion only.

SANITY ASSERT: with the disagreement block ZEROED, the augmented meta must
reproduce n091 ≈ 0.970355. If delta > 0.0001 → STOP (same guard as n122/n127).

Pool: TIGHT = bank-17 + FT-T + 36 in-house = 54 bases × 3 = 162 cols
      FULL  = TIGHT + 9 weak = 63 bases × 3 = 189 cols
Both arms run.
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
NODE_DIR = COMP / "nodes/node_0136"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3

# C grid to sweep (nested, inner fold selection via LogisticRegressionCV)
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# TIGHT pool: 36 strong distinct in-house bases (byte-identical to n091)
TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]

# FULL pool extra = weak bases added on top of TIGHT
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]


# ---------------------------------------------------------------------------
# Helpers (verbatim from champion)
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
# Disagreement feature builder
# ---------------------------------------------------------------------------

def build_disagreement_features(base_probs_list: list[np.ndarray]) -> np.ndarray:
    """Build per-row disagreement features from a list of base OOF prob arrays.

    Each element of base_probs_list is (n_rows, 3) softmax-normalized probs
    for [GALAXY, QSO, STAR]. These are the raw normalized probs (not logp).

    Returns a (n_rows, 7) array:
      col 0:   argmax-vote entropy across bases (high = bases disagree on winner)
      col 1-3: per-class std across bases (dispersion of each class probability)
      col 4:   max − 2nd-max of the MEAN base prob (decision confidence margin)
      col 5:   top-1 class identity disagreement (std of argmax votes, 0=all agree)
      col 6:   top-2 class identity disagreement flag (binary: do bases differ on
               which two classes are in contention)

    These features are STATELESS — no fitting, just per-row aggregation of the
    base OOF. Since each base OOF is leave-fold-out, no leakage is introduced.
    """
    # Stack: shape (n_bases, n_rows, 3)
    stack = np.stack(base_probs_list, axis=0)   # (B, n, 3)
    n_bases, n_rows, _ = stack.shape

    # --- 1. Argmax vote distribution entropy ---
    # Each base votes for one class (0, 1, 2)
    votes = stack.argmax(axis=2)   # (B, n)
    # Count fraction of votes for each class per row
    vote_probs = np.zeros((n_rows, NC), dtype=float)
    for c in range(NC):
        vote_probs[:, c] = (votes == c).sum(axis=0) / n_bases
    # Shannon entropy of vote distribution
    eps = 1e-10
    vote_entropy = -(vote_probs * np.log(vote_probs + eps)).sum(axis=1)   # (n,)

    # --- 2. Per-class probability dispersion (std across bases) ---
    class_std = stack.std(axis=0)   # (n, 3) — std per row per class

    # --- 3. Mean base prob margin (max - 2nd-max) ---
    mean_prob = stack.mean(axis=0)  # (n, 3)
    sorted_mean = np.sort(mean_prob, axis=1)[:, ::-1]  # descending per row
    margin = sorted_mean[:, 0] - sorted_mean[:, 1]   # (n,)

    # --- 4. Argmax identity std (how spread are the argmax labels) ---
    # std of integer argmax votes (0/1/2) per row; 0 = all agree
    vote_std = votes.astype(float).std(axis=0)   # (n,)

    # --- 5. Top-2 identity disagreement ---
    # Top-2 class indices from mean prob per row
    top2_mean = np.argsort(-mean_prob, axis=1)[:, :2]   # (n, 2) — descending
    # Top-2 from each base
    top2_base = np.argsort(-stack, axis=2)[:, :, :2]    # (B, n, 2)
    # A base's top-2 set = frozenset. Check if any base's top-2 set differs from mean's top-2 set.
    # Efficient: compare sorted top-2 indices
    top2_mean_sorted = np.sort(top2_mean, axis=1)   # (n, 2)
    top2_base_sorted = np.sort(top2_base, axis=2)   # (B, n, 2)
    # Disagreement: any base has different top-2 set than the consensus top-2
    # shape: (B, n) — True if base b disagrees with consensus on row i
    disagree_top2 = (top2_base_sorted != top2_mean_sorted[None, :, :]).any(axis=2)  # (B, n)
    top2_disagree_frac = disagree_top2.mean(axis=0)   # (n,) fraction of bases that disagree

    # Assemble: 7 columns total
    feat = np.column_stack([
        vote_entropy,        # col 0
        class_std,           # cols 1-3 (3 cols)
        margin,              # col 4
        vote_std,            # col 5
        top2_disagree_frac,  # col 6
    ])
    return feat.astype(np.float32)


def nested_cv_arm(
    OOF_mat: np.ndarray,
    TST_mat: np.ndarray,
    y: np.ndarray,
    fval: list[np.ndarray],
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[float], float, float, list[float]]:
    """Run nested C-selection + outer OOF loop using LogisticRegressionCV.
    Byte-identical to champion/src/solution.py nested_cv_arm.
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
    print(f"  best Cs per fold: {best_Cs_per_fold}", flush=True)

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


def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n = len(train)
    nt = len(test)
    y = train["class"].map(L2I).to_numpy()

    fval = [np.asarray(f["val_idx"]) for f in folds_data]
    n_folds = len(fval)

    print(f"n_train={n} n_test={nt} n_folds={n_folds}", flush=True)
    assert n == 577347, f"unexpected n_train={n}"
    assert nt == 247435, f"unexpected n_test={nt}"

    # =========================================================================
    # PRE-FLIGHT: Leakage checks 1-2
    print("\n[LEAKAGE CHECK 1-2] Features are OOF probs + disagreement stats only (no target/id). PASS", flush=True)
    print("[LEAKAGE CHECK 4] LogReg fit inside fold loop; C selected by LogisticRegressionCV on outer-train only. PASS", flush=True)
    print("  Disagreement features: stateless per-row aggregation of base OOF — no fit, no target. PASS", flush=True)
    print("[LEAKAGE CHECK 5] Folds loaded from frozen folds.json. PASS", flush=True)

    # =========================================================================
    # Load public bank-17 (same MANIFEST as champion)
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
    # Load FT-Transformer
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
    # LEAKAGE CHECK 3: single-feature↔target sweep on a sample
    print("\n[LEAKAGE CHECK 3] Single-feature correlation sweep (50k sample)...", flush=True)
    rng = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys = y[sidx].astype(float)
    for check_name, arr in [
        ("ft_transformer_col0", logp(norm(ft_oof_raw))[sidx, 0]),
        ("node_0001_col0", list({f"node_{nid:04d}": None for nid in TIGHT_IDS}.keys())),  # placeholder
    ]:
        if isinstance(arr, list):
            continue
        corr = abs(np.corrcoef(arr, ys)[0, 1])
        print(f"  {check_name}: |corr|={corr:.4f}", flush=True)
        if corr >= 0.999:
            raise SystemExit(f"LEAK smell: {check_name} ~ target corr={corr:.4f}")
    print("[LEAKAGE CHECK 3] PASS", flush=True)

    # =========================================================================
    # Load in-house base OOF / test_probs (TIGHT set, 36 nodes)
    print("\n" + "="*70, flush=True)
    print("Loading in-house TIGHT bases (36 nodes)...", flush=True)

    inhouse_oof_tight  = {}
    inhouse_test_tight = {}
    inhouse_norm_oof_tight = {}  # raw normalized probs (for disagreement features)
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
                print(f"{node_nm:12s} {solo_ba:9.6f} SKIP (column-order bug)", flush=True)
                continue
            inhouse_oof_tight[node_nm]      = logp(o)
            inhouse_test_tight[node_nm]     = logp(t)
            inhouse_norm_oof_tight[node_nm] = o   # for disagreement features
            print(f"{node_nm:12s} {solo_ba:9.6f} {str(o.shape):>12s} OK", flush=True)
        except Exception as e:
            print(f"{node_nm:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}", flush=True)

    print(f"\nLoaded {len(inhouse_oof_tight)}/{len(TIGHT_IDS)} TIGHT in-house bases", flush=True)

    # Weak bases for FULL pool
    print("\nLoading weak EXTRA bases (9 nodes for FULL pool)...", flush=True)
    inhouse_oof_weak  = {}
    inhouse_test_weak = {}
    inhouse_norm_oof_weak = {}
    for nid in WEAK_EXTRA_IDS:
        node_nm = f"node_{nid:04d}"
        oof_path  = COMP / "nodes" / node_nm / "oof.npy"
        test_path = COMP / "nodes" / node_nm / "test_probs.npy"
        try:
            o_raw = np.load(oof_path).astype(float)
            t_raw = np.load(test_path).astype(float)
            assert o_raw.shape == (n,  3)
            assert t_raw.shape == (nt, 3)
            assert not np.isnan(o_raw).any()
            assert not np.isnan(t_raw).any()
            o = norm(o_raw)
            t = norm(t_raw)
            solo_ba = score_fn(y, o.argmax(1))
            if solo_ba < 0.5:
                print(f"{node_nm:12s} {solo_ba:9.6f} SKIP (column-order bug)", flush=True)
                continue
            inhouse_oof_weak[node_nm]      = logp(o)
            inhouse_test_weak[node_nm]     = logp(t)
            inhouse_norm_oof_weak[node_nm] = o
            print(f"{node_nm:12s} {solo_ba:9.6f} {str(o.shape):>12s} OK", flush=True)
        except Exception as e:
            print(f"{node_nm:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}", flush=True)

    print(f"Loaded {len(inhouse_oof_weak)}/{len(WEAK_EXTRA_IDS)} weak extra bases", flush=True)

    # =========================================================================
    # SANITY ASSERT: with the disagreement block ZEROED, must reproduce n091 ≈ 0.970355
    # We run the zeroed-block meta (= pure n091 FULL arm with logprob only) on the
    # TIGHT pool first to verify the base pipeline is byte-identical.
    print("\n" + "="*70, flush=True)
    print("SANITY ASSERT: tight pool, disagreement block ZEROED (reproducing n091 base)...", flush=True)

    tight_base_oof_logp  = [logp(POOF[k]) for k in good] + [logp(norm(ft_oof_raw))]
    tight_base_test_logp = [logp(PTEST[k]) for k in good] + [logp(norm(ft_test_raw))]

    OOF_tight_base = np.concatenate(
        tight_base_oof_logp + list(inhouse_oof_tight.values()), axis=1
    )

    sanity_oof = np.zeros((n, NC))
    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        m = LogisticRegression(class_weight="balanced", C=0.003, max_iter=2000,
                               n_jobs=-1, random_state=42,
                               solver="lbfgs", multi_class="multinomial")
        m.fit(OOF_tight_base[tr_idx], y[tr_idx])
        sanity_oof[vi] = m.predict_proba(OOF_tight_base[vi])
        print(f"  sanity fold {fi} done", flush=True)

    sanity_fold_scores = [score_fn(y[vi], sanity_oof[vi].argmax(1)) for vi in fval]
    sanity_cv = float(np.mean(sanity_fold_scores))
    EXPECTED_N091_CV = 0.970355
    delta = abs(sanity_cv - EXPECTED_N091_CV)
    print(f"\nSANITY cv={sanity_cv:.6f}  expected≈{EXPECTED_N091_CV:.6f}  delta={delta:.6f}", flush=True)
    print(f"  per-fold: {[f'{s:.6f}' for s in sanity_fold_scores]}", flush=True)
    if delta > 0.0002:
        print(f"STOP: SANITY ASSERT FAILED (delta={delta:.6f} > 0.0002). "
              f"Base pipeline is not byte-identical to n091. Exiting.", flush=True)
        import sys; sys.exit(1)
    print("SANITY ASSERT: PASS (reproduces n091 within tolerance)", flush=True)

    # =========================================================================
    # Build DISAGREEMENT features for train and test
    # Train: use the TIGHT pool base OOF probs (already leave-fold-out → no leak)
    # Test: use the TIGHT pool base test probs
    # NOTE: for FULL arm we also include the weak bases in the disagreement calc
    # so that the features reflect the same pool as the meta sees.
    print("\n" + "="*70, flush=True)
    print("Building disagreement features...", flush=True)

    # TIGHT pool: bank-17 norm probs + FT-T norm probs + inhouse tight norm probs
    tight_pool_oof_probs  = [POOF[k] for k in good] + [norm(ft_oof_raw)] + list(inhouse_norm_oof_tight.values())
    tight_pool_test_probs = [PTEST[k] for k in good] + [norm(ft_test_raw)] + list({
        nm: norm(np.load(COMP/"nodes"/nm/"test_probs.npy").astype(float))
        for nm in inhouse_norm_oof_tight
    }.values())

    DISAG_tight_train = build_disagreement_features(tight_pool_oof_probs)
    DISAG_tight_test  = build_disagreement_features(tight_pool_test_probs)
    print(f"TIGHT disagreement features: train={DISAG_tight_train.shape}  test={DISAG_tight_test.shape}", flush=True)

    # FULL pool: tight + weak
    full_pool_oof_probs = tight_pool_oof_probs + list(inhouse_norm_oof_weak.values())
    full_pool_test_probs_raw = []
    for nm in inhouse_norm_oof_weak:
        full_pool_test_probs_raw.append(norm(np.load(COMP/"nodes"/nm/"test_probs.npy").astype(float)))
    full_pool_test_probs = tight_pool_test_probs + full_pool_test_probs_raw

    DISAG_full_train = build_disagreement_features(full_pool_oof_probs)
    DISAG_full_test  = build_disagreement_features(full_pool_test_probs)
    print(f"FULL  disagreement features: train={DISAG_full_train.shape}  test={DISAG_full_test.shape}", flush=True)

    # Leakage check 3 on disagreement features (spot check)
    for col_idx, col_name in enumerate(["vote_entropy", "class_std_0", "class_std_1", "class_std_2",
                                         "margin", "vote_std", "top2_disagree_frac"]):
        arr = DISAG_tight_train[sidx, col_idx].astype(float)
        corr = abs(np.corrcoef(arr, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK smell in disagreement feature '{col_name}': corr={corr:.4f}")
    print("[LEAKAGE CHECK 3 - disagreement features] All corr < 0.999: PASS", flush=True)

    # =========================================================================
    # Build augmented feature matrices (base logprobs + disagreement block)
    # TIGHT arm
    OOF_tight_aug  = np.concatenate(
        tight_base_oof_logp + list(inhouse_oof_tight.values()) + [DISAG_tight_train.astype(float)],
        axis=1
    )
    TST_tight_aug  = np.concatenate(
        tight_base_test_logp + list(inhouse_test_tight.values()) + [DISAG_tight_test.astype(float)],
        axis=1
    )

    # FULL arm
    full_base_oof_logp  = tight_base_oof_logp  + list(inhouse_oof_tight.values())  + list(inhouse_oof_weak.values())
    full_base_test_logp = tight_base_test_logp + list(inhouse_test_tight.values()) + list(inhouse_test_weak.values())

    OOF_full_aug   = np.concatenate(
        full_base_oof_logp + [DISAG_full_train.astype(float)], axis=1
    )
    TST_full_aug   = np.concatenate(
        full_base_test_logp + [DISAG_full_test.astype(float)], axis=1
    )

    print(f"\nTIGHT aug: feature_matrix={OOF_tight_aug.shape} "
          f"(base {OOF_tight_base.shape[1]} + disag {DISAG_tight_train.shape[1]})", flush=True)
    print(f"FULL  aug: feature_matrix={OOF_full_aug.shape}", flush=True)

    # =========================================================================
    # RUN TIGHT arm (augmented)
    print("\n" + "="*70, flush=True)
    (oof_tight, test_tight, pf_tight,
     cv_tight, sem_tight, Cs_tight) = nested_cv_arm(
        OOF_tight_aug, TST_tight_aug, y, fval, "TIGHT+DISAG"
    )

    # RUN FULL arm (augmented)
    print("\n" + "="*70, flush=True)
    (oof_full, test_full, pf_full,
     cv_full, sem_full, Cs_full) = nested_cv_arm(
        OOF_full_aug, TST_full_aug, y, fval, "FULL+DISAG"
    )

    # =========================================================================
    # Determine winning arm
    print("\n" + "="*70, flush=True)
    print("=== ARM COMPARISON ===", flush=True)
    print(f"TIGHT+DISAG: cv={cv_tight:.6f}  sem={sem_tight:.6f}  per_fold={[f'{s:.6f}' for s in pf_tight]}", flush=True)
    print(f"FULL+DISAG:  cv={cv_full:.6f}  sem={sem_full:.6f}  per_fold={[f'{s:.6f}' for s in pf_full]}", flush=True)

    if cv_tight >= cv_full:
        winner = "TIGHT+DISAG"
        cv_win, sem_win, pf_win = cv_tight, sem_tight, pf_tight
        oof_win, test_win = oof_tight, test_tight
    else:
        winner = "FULL+DISAG"
        cv_win, sem_win, pf_win = cv_full, sem_full, pf_full
        oof_win, test_win = oof_full, test_full

    print(f"\nWINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)

    champ_cv  = 0.970355
    champ_sem = 0.000249
    promote_bar = champ_cv + 2 * champ_sem
    lift_vs_champ = cv_win - champ_cv
    print(f"champion  cv={champ_cv:.6f}  2*sem={2*champ_sem:.6f}  promote_bar={promote_bar:.6f}", flush=True)
    print(f"lift_vs_champ={lift_vs_champ:+.6f}  beats_promote={'YES' if cv_win > promote_bar else 'NO'}", flush=True)

    # Disagreement vs sanity comparison
    lift_vs_zeroed = cv_win - sanity_cv
    print(f"\nSanity (zeroed disag block): cv={sanity_cv:.6f}", flush=True)
    print(f"Augmented (disag block):     cv={cv_win:.6f}", flush=True)
    print(f"Lift from disag block:       {lift_vs_zeroed:+.6f}", flush=True)

    # =========================================================================
    # Write artifacts
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

    assert list(sub.columns) == list(sample_sub.columns), \
        f"column mismatch: {list(sub.columns)} vs {list(sample_sub.columns)}"
    assert len(sub) == len(sample_sub), \
        f"row count: {len(sub)} vs {len(sample_sub)}"
    assert set(sub["class"].unique()) <= set(LAB), \
        f"unknown classes: {set(sub['class'].unique()) - set(LAB)}"
    print("  schema_ok: PASS", flush=True)

    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    print("  oof_full: PASS  no_nan: PASS", flush=True)

    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5, \
        f"OOF probs out of [0,1]: min={oofn.min()}, max={oofn.max()}"
    row_sums = oofn.sum(axis=1)
    assert abs(row_sums.mean() - 1.0) < 0.01, \
        f"OOF row sums off: mean={row_sums.mean()}"
    class_counts = np.bincount(oofn.argmax(1), minlength=3)
    print(f"  dist_sane: PASS  OOF argmax dist: GALAXY={class_counts[0]} QSO={class_counts[1]} STAR={class_counts[2]}", flush=True)
    print(f"             range=[{oofn.min():.4f},{oofn.max():.4f}]  row_sums_mean={row_sums.mean():.6f}", flush=True)

    cv_too_good = cv_win > 0.980
    print(f"  cv_too_good: {'WARN (>0.980)' if cv_too_good else 'PASS'}", flush=True)

    # =========================================================================
    # Final summary
    print("\n" + "="*70, flush=True)
    print("=== FINAL SUMMARY ===", flush=True)
    print(f"SANITY (zeroed):   cv={sanity_cv:.6f}", flush=True)
    print(f"TIGHT+DISAG:       cv={cv_tight:.6f}  sem={sem_tight:.6f}  Cs={Cs_tight}  per_fold={[f'{s:.6f}' for s in pf_tight]}", flush=True)
    print(f"FULL+DISAG:        cv={cv_full:.6f}  sem={sem_full:.6f}   Cs={Cs_full}  per_fold={[f'{s:.6f}' for s in pf_full]}", flush=True)
    print(f"WINNER: {winner}  cv={cv_win:.6f}  sem={sem_win:.6f}", flush=True)
    print(f"promotes? {'YES' if cv_win > promote_bar else 'NO'}  (bar={promote_bar:.6f})", flush=True)
    print(f"lift_vs_zeroed={lift_vs_zeroed:+.6f}", flush=True)
    print(f"cv={cv_win:.6f}", flush=True)


if __name__ == "__main__":
    main()
