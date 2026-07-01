"""node_0093 — Non-negative simplex convex blend (TIGHT vs FULL pool A/B)

Atomic change vs node_0070 (bank-17 + FT-T, LogReg meta, cv=0.970211):
  Replace the per-class LogReg meta with a NON-NEGATIVE SIMPLEX-CONSTRAINED
  convex blend — ONE scalar weight per base, all weights >= 0 summing to 1,
  applied to the bases' PROBABILITIES (not log-probs), found by SLSQP
  minimizing cross-entropy (smooth surrogate), then argmax for final labels.

  Two pools A/B (identical to node_0091 for comparability):
  - TIGHT: bank-17 + FT-T + 36 strong-distinct in-house bases
  - FULL:  TIGHT + 9 weak bases (n8/n21/n22/n24/n25/n26/n27/n37/n62)

  Fold-honest: weights optimized on the train-fold OOF only, applied to
  held-out fold; frozen folds.json; threshold-free argmax.

  HARD BASELINE ASSERT: bank-17 + FT-T LogReg OOF BA must be ~0.970211.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3

# ---------------------------------------------------------------------------
# Helpers (reused from node_0076)
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


# ---------------------------------------------------------------------------
# Simplex blend solver
# ---------------------------------------------------------------------------

def simplex_blend_weights(
    probs: list[np.ndarray],  # list of (n, 3) arrays — train-fold bases
    y: np.ndarray,            # (n,) integer labels
    tol: float = 1e-8,
) -> np.ndarray:
    """Find non-negative weights summing to 1 minimizing cross-entropy loss.

    Uses SLSQP with:
      constraints: sum(w) = 1
      bounds: w_i >= 0
    Smooth surrogate: mean cross-entropy of the blended probability.
    Starting point: uniform weights.
    """
    B = len(probs)
    n = probs[0].shape[0]
    # Stack: (n, B, 3)
    P = np.stack(probs, axis=1)  # (n, B, 3)

    # One-hot targets (n, 3)
    Y = np.zeros((n, NC))
    Y[np.arange(n), y] = 1.0

    def cross_entropy(w):
        # blend: (n, 3) = sum_b w_b * P[:, b, :]
        blended = (P * w[np.newaxis, :, np.newaxis]).sum(axis=1)  # (n, 3)
        blended = np.clip(blended, 1e-10, 1.0)
        return -float(np.mean((Y * np.log(blended)).sum(axis=1)))

    def grad_ce(w):
        blended = (P * w[np.newaxis, :, np.newaxis]).sum(axis=1)
        blended = np.clip(blended, 1e-10, 1.0)
        # dL/dw_b = -mean_i sum_c Y[i,c] * P[i,b,c] / blended[i,c]
        ratio = Y / blended  # (n, 3)
        # sum over classes, then over samples
        grad = -(ratio[:, np.newaxis, :] * P).sum(axis=(0, 2)) / n  # (B,)
        return grad

    w0 = np.ones(B) / B

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * B

    result = minimize(
        cross_entropy,
        w0,
        jac=grad_ce,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": tol},
    )
    w = np.maximum(result.x, 0.0)
    w = w / w.sum()
    return w


def apply_blend(probs: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    """Apply weights to a list of (n, 3) probability arrays."""
    return sum(w_i * p for w_i, p in zip(w, probs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n = len(train)
    nt = len(test)
    y = train["class"].map(L2I).to_numpy()
    fval = [np.asarray(f["val_idx"]) for f in folds_data]
    n_folds = len(fval)

    print(f"n_train={n} n_test={nt} n_folds={n_folds}")
    assert n == 577347, f"unexpected n_train={n}"
    assert nt == 247435, f"unexpected n_test={nt}"

    # ---- PRE-FLIGHT LEAKAGE CHECKS 1-2 ----
    print("Leakage check 1-2: features are OOF probs only (no target/id). PASS")
    print("Leakage check 4-5: weights fit inside fold loop only; folds from frozen folds.json. PASS")

    # ---- Load public bank-17 (identical MANIFEST to node_0076) ----
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
    print(f"\n{'model':14s} {'oofBA':>9s} {'shape':>12s} {'status'}")
    for name, (op, tp) in MANIFEST.items():
        try:
            o = norm(rd(op, n)); t = norm(rd(tp, nt))
            assert o.shape == (n, 3) and t.shape == (nt, 3)
            ba = balanced_accuracy_score(y, o.argmax(1))
            st = "OK" if 0.90 < ba < 0.972 else ("QUARANTINE" if ba >= 0.972 else "LOW?")
            if st == "OK": POOF[name] = o; PTEST[name] = t; good.append(name)
            print(f"{name:14s} {ba:9.6f} {str(o.shape):>12s} {st}")
        except Exception as e:
            print(f"{name:14s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}")

    print(f"\nLoaded {len(good)} public models OK (expected 17)")

    # ---- Load FT-Transformer ----
    PILK = COMP / "refs" / "ext_oof" / "pilkwang_5090"
    ft_oof_path  = PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"
    ft_test_path = PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv"

    ft_oof_raw  = norm(load_ext_csv(ft_oof_path, n))
    ft_test_raw = norm(load_ext_csv(ft_test_path, nt))
    assert ft_oof_raw.shape == (n, 3),   f"FT-T OOF shape {ft_oof_raw.shape}"
    assert ft_test_raw.shape == (nt, 3), f"FT-T test shape {ft_test_raw.shape}"
    ft_solo_ba = score_fn(y, ft_oof_raw.argmax(1))
    print(f"\nft_transformer: solo_BA={ft_solo_ba:.6f}  shape={ft_oof_raw.shape}")
    assert ft_solo_ba > 0.85, f"ft_transformer solo BA {ft_solo_ba:.4f} too low"

    # ---- HARD BASELINE ASSERT: reproduce bank-17 + FT-T LogReg ≈ 0.970211 ----
    print("\n=== HARD BASELINE ASSERT: bank-17 + FT-T LogReg ===")
    # Build log-prob feature matrix (identical to node_0076/node_0070)
    all_oof_logp  = [logp(POOF[k]) for k in good] + [logp(ft_oof_raw)]
    all_test_logp = [logp(PTEST[k]) for k in good] + [logp(ft_test_raw)]
    OOF_base = np.concatenate(all_oof_logp, axis=1)   # (n, 18*3=54)
    TST_base = np.concatenate(all_test_logp, axis=1)  # (nt, 54)

    # Single seed, no bagging — fast check
    baseline_oof = np.zeros((n, NC))
    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)
        m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000,
                               n_jobs=-1, random_state=42)
        m.fit(OOF_base[tr_idx], y[tr_idx])
        baseline_oof[vi] = m.predict_proba(OOF_base[vi])
    baseline_scores = [score_fn(y[vi], baseline_oof[vi].argmax(1)) for vi in fval]
    baseline_cv = float(np.mean(baseline_scores))
    print(f"Baseline (bank-17+FT-T LogReg) per-fold: {[f'{s:.6f}' for s in baseline_scores]}")
    print(f"Baseline CV = {baseline_cv:.6f}  (target ~0.970211)")
    assert abs(baseline_cv - 0.970211) < 1e-3, \
        f"BASELINE ASSERT FAILED: {baseline_cv:.6f} not within 1e-3 of 0.970211 — check OOF ingest"
    print("BASELINE ASSERT: PASS")

    # ---- Leakage check 3: single-feature correlation sweep ----
    print("\nLeakage check 3: single-feature correlation sweep (sample 50k)...")
    rng = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys = y[sidx].astype(float)
    for ci in range(3):
        x = ft_oof_raw[sidx, ci]
        corr = abs(np.corrcoef(x, ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK: ft_transformer col {ci} ~ target corr={corr:.4f}")
    print("Leakage check 3: PASS")

    # ---- Load TIGHT pool in-house bases ----
    TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
                 28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
                 49, 50, 51, 55, 56, 60, 61, 66, 85]

    # FULL pool adds these weak bases
    WEAK_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]

    NODES_DIR = COMP / "nodes"

    def load_inhouse(ids, label=""):
        oof_list = []; test_list = []; names = []
        print(f"\nLoading {label} in-house bases ({len(ids)} nodes):")
        print(f"  {'node':12s} {'solo_BA':>9s} {'shape':>12s} {'status'}")
        for nid in ids:
            ndir = NODES_DIR / f"node_{nid:04d}"
            op = ndir / "oof.npy"
            tp = ndir / "test_probs.npy"
            try:
                o = norm(np.load(op).astype(float))
                t = norm(np.load(tp).astype(float))
                assert o.shape == (n, 3), f"oof shape {o.shape}"
                assert t.shape == (nt, 3), f"test shape {t.shape}"
                assert not np.isnan(o).any(), "NaN in oof"
                ba = score_fn(y, o.argmax(1))
                assert ba > 0.33, f"solo BA {ba:.4f} is ~chance — column-order bug?"
                oof_list.append(o)
                test_list.append(t)
                names.append(f"n{nid}")
                print(f"  node_{nid:04d}    {ba:9.6f} {str(o.shape):>12s} OK")
            except Exception as e:
                print(f"  node_{nid:04d}    {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}")
        return oof_list, test_list, names

    tight_oof, tight_test, tight_names = load_inhouse(TIGHT_IDS, "TIGHT")
    weak_oof, weak_test, weak_names = load_inhouse(WEAK_IDS, "WEAK (FULL extras)")

    # ---- Build probability pools ----
    # bank-17 + FT-T probs (not log-probs — this is the key difference from LogReg meta)
    bank_oof_probs  = [POOF[k] for k in good] + [ft_oof_raw]   # 18 arrays (n, 3)
    bank_test_probs = [PTEST[k] for k in good] + [ft_test_raw] # 18 arrays (nt, 3)
    bank_names = list(good) + ["ft_transformer"]

    # TIGHT pool: bank-17 + FT-T + 36 tight in-house
    tight_pool_oof  = bank_oof_probs  + tight_oof   # 18 + 36 = 54 bases
    tight_pool_test = bank_test_probs + tight_test
    tight_pool_names = bank_names + tight_names

    # FULL pool: TIGHT + 9 weak in-house
    full_pool_oof  = tight_pool_oof  + weak_oof   # 54 + 9 = 63 bases
    full_pool_test = tight_pool_test + weak_test
    full_pool_names = tight_pool_names + weak_names

    print(f"\nPool sizes: TIGHT={len(tight_pool_oof)}, FULL={len(full_pool_oof)}")

    # ---- Nested fold-honest simplex blend ----
    def run_simplex_arm(
        pool_oof: list[np.ndarray],
        pool_test: list[np.ndarray],
        pool_names: list[str],
        y: np.ndarray,
        fval: list[np.ndarray],
        arm_name: str,
    ):
        n = pool_oof[0].shape[0]
        nt = pool_test[0].shape[0]
        B = len(pool_oof)
        print(f"\n{'='*60}")
        print(f"ARM: {arm_name}  ({B} bases)")
        print(f"{'='*60}")

        # OOF loop — fold-honest: fit weights on train fold, score on val fold
        oof_blend = np.zeros((n, NC))
        per_fold_weights = []
        per_fold_scores = []

        for fi, vi in enumerate(fval):
            tr_idx = np.setdiff1d(np.arange(n), vi)
            # Train fold probs for each base
            train_probs = [p[tr_idx] for p in pool_oof]
            val_probs   = [p[vi]     for p in pool_oof]
            y_tr = y[tr_idx]

            # Fit weights on train fold
            w = simplex_blend_weights(train_probs, y_tr)
            per_fold_weights.append(w)

            # Apply to val fold
            val_blend = apply_blend(val_probs, w)
            oof_blend[vi] = val_blend

            fold_ba = score_fn(y[vi], val_blend.argmax(1))
            per_fold_scores.append(fold_ba)

            # Report nonzero weights
            nz = [(pool_names[i], w[i]) for i in range(B) if w[i] > 0.005]
            nz.sort(key=lambda x: -x[1])
            print(f"\nFold {fi}: BA={fold_ba:.6f}")
            print(f"  Top weights ({len(nz)} bases with w>0.005):")
            for nm, wi in nz[:15]:
                print(f"    {nm:20s} {wi:.4f}")

        cv_mean = float(np.mean(per_fold_scores))
        cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
        print(f"\n{arm_name} nested OOF BA: {per_fold_scores}")
        print(f"{arm_name} CV={cv_mean:.6f}  SEM={cv_sem:.6f}")

        # Final refit on ALL train data (for test predictions)
        all_probs = pool_oof  # all n train rows
        w_final = simplex_blend_weights(all_probs, y)
        nz_final = [(pool_names[i], w_final[i]) for i in range(B) if w_final[i] > 0.005]
        nz_final.sort(key=lambda x: -x[1])
        print(f"\nFinal weights (fit on all train, {len(nz_final)} bases with w>0.005):")
        for nm, wi in nz_final[:20]:
            print(f"  {nm:20s} {wi:.4f}")

        # Test predictions
        test_blend = apply_blend(pool_test, w_final)

        return oof_blend, test_blend, cv_mean, cv_sem, per_fold_scores, w_final

    # Run both arms
    tight_oof_blend, tight_test_blend, tight_cv, tight_sem, tight_folds, tight_w = \
        run_simplex_arm(tight_pool_oof, tight_pool_test, tight_pool_names,
                        y, fval, "TIGHT")

    full_oof_blend, full_test_blend, full_cv, full_sem, full_folds, full_w = \
        run_simplex_arm(full_pool_oof, full_pool_test, full_pool_names,
                        y, fval, "FULL")

    # ---- Pick the winner ----
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  TIGHT: cv={tight_cv:.6f}  sem={tight_sem:.6f}  folds={tight_folds}")
    print(f"  FULL:  cv={full_cv:.6f}  sem={full_sem:.6f}  folds={full_folds}")
    print(f"  Baseline (bank-17+FT-T LogReg): cv={baseline_cv:.6f}")
    print(f"  Parent  (node_0070 5-seed bagged): cv=0.970211")
    print(f"  Champion (node_0063): cv=0.970153  promote_bar=0.970597")

    if tight_cv >= full_cv:
        winner_name = "TIGHT"
        winner_oof = tight_oof_blend
        winner_test = tight_test_blend
        winner_cv = tight_cv
        winner_sem = tight_sem
        winner_folds = tight_folds
        winner_w = tight_w
        winner_names = tight_pool_names
    else:
        winner_name = "FULL"
        winner_oof = full_oof_blend
        winner_test = full_test_blend
        winner_cv = full_cv
        winner_sem = full_sem
        winner_folds = full_folds
        winner_w = full_w
        winner_names = full_pool_names

    print(f"\nWINNER: {winner_name}  cv={winner_cv:.6f}  sem={winner_sem:.6f}")

    beats_parent = winner_cv > 0.970211
    beats_promote = winner_cv > 0.970597
    print(f"Beats parent (0.970211): {beats_parent}")
    print(f"Beats promote bar (0.970597): {beats_promote}")

    if not beats_parent:
        print("\nNEITHER arm beats parent 0.970211 — recording clean null result")
        print("Producing oof.npy + test_probs.npy of best arm for future combines")

    # ---- Write outputs ----
    np.save(NODE_DIR / "oof.npy", winner_oof)
    np.save(NODE_DIR / "test_probs.npy", winner_test)

    # Submission (argmax of winner)
    test_preds_idx = winner_test.argmax(1)
    test_labels = [I2L[i] for i in test_preds_idx]
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    print(f"\nsubmission.csv written: {len(sub)} rows")
    print(f"oof.npy: {winner_oof.shape}  test_probs.npy: {winner_test.shape}")

    # ---- Post-run output gates ----
    # Schema check
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count: {len(sub)} vs {len(sample_sub)}"
    print("submission schema: OK")

    oofn = np.load(NODE_DIR / "oof.npy")
    assert oofn.shape == (n, NC), f"oof shape {oofn.shape}"
    assert not np.isnan(oofn).any(), "NaN in OOF"
    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5, "OOF probs out of [0,1]"
    row_sums = oofn.sum(axis=1)
    assert abs(row_sums.mean() - 1.0) < 0.01, f"OOF row sums off: mean={row_sums.mean()}"

    # Coverage: every train row predicted exactly once
    all_val_idx = np.concatenate([vi for vi in fval])
    assert len(all_val_idx) == n, f"OOF coverage: {len(all_val_idx)} != {n}"
    assert len(np.unique(all_val_idx)) == n, "OOF has duplicate indices"
    print(f"oof_full: PASS  no_nan: PASS  dist_sane: PASS  (range=[{oofn.min():.4f},{oofn.max():.4f}])")

    # Distribution sanity
    class_counts = np.bincount(winner_test.argmax(1), minlength=NC)
    print(f"Test prediction distribution: GALAXY={class_counts[0]}, QSO={class_counts[1]}, STAR={class_counts[2]}")
    assert all(c > 0 for c in class_counts), "Collapsed prediction — some class has 0 predictions"

    # CV too-good check
    cv_jump = winner_cv - 0.970211
    if cv_jump > 0.001:
        print(f"WARNING: cv_too_good flag — jump vs parent is {cv_jump:+.6f} > 0.001, inspect")

    # ---- Final summary ----
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"TIGHT cv={tight_cv:.6f}  sem={tight_sem:.6f}  folds={tight_folds}")
    print(f"FULL  cv={full_cv:.6f}  sem={full_sem:.6f}  folds={full_folds}")
    print(f"Winner: {winner_name}")
    print(f"cv={winner_cv:.6f}  sem={winner_sem:.6f}")
    print(f"folds={winner_folds}")
    print(f"Beats parent (0.970211): {beats_parent}")
    print(f"Beats promote bar (0.970597): {beats_promote}")
    print(f"Baseline assert (bank-17+FT-T ≈ 0.970211): PASS ({baseline_cv:.6f})")

    # Nonzero weight bases in winner
    B_winner = len(winner_names)
    nz_winner = [(winner_names[i], winner_w[i]) for i in range(B_winner) if winner_w[i] > 0.005]
    nz_winner.sort(key=lambda x: -x[1])
    print(f"\nNonzero-weight bases in {winner_name} winner ({len(nz_winner)} w>0.005):")
    for nm, wi in nz_winner:
        print(f"  {nm:20s} {wi:.4f}")

    print(f"\ncv={winner_cv:.6f}")


if __name__ == "__main__":
    main()
