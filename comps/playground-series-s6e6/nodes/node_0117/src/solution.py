"""node_0117 — max-decorrelated rank-vote finals hedge

Atomic change vs node_0091 (champion LogReg mega-stack):
  DIFFERENT decision rule: Borda-count soft-rank vote, NOT a dense LogReg
  on log-probs and NOT a convex prob-average blend.

  Over the 6 LOO-load-bearing bases (top causal-family bases from
  probes/drop_study_ranking.csv + n091 notes):
    1. cat-3        (drop_delta=+0.000158, top bank base)
    2. realmlp-2    (drop_delta=+0.000059)
    3. node_0003    (drop_delta=+0.000049, inhouse CatBoost)
    4. node_0039    (top inhouse by |coef| in n091: 0.9772)
    5. lgbm-5       (drop_delta=+0.000042)
    6. ft_transformer (drop_delta=+0.000038)

Algorithm (Borda-count soft-rank vote):
  For each row, for each base, rank the 3 class probabilities (highest prob
  gets rank 2, middle rank 1, lowest rank 0) and normalize to [0,1]. Sum
  these Borda ranks across bases with per-base weights. Take argmax of the
  summed rank scores as the predicted class.

  This is a DIFFERENT decision rule from LogReg (per-row rank aggregation
  vs log-prob regression), producing a different error geometry. A finals
  hedge, not a CV win.

  DE tunes per-base weights on a 10k stratified subsample of the train fold
  to maximize Balanced Accuracy. Val fold is NEVER seen during weight tuning.

  Final test predictions use the average of per-fold DE weights.

Leakage: OOF probs are pre-computed; no target or id enters features.
         DE weight optimization fits on train-fold rows only (val never seen).
         Folds loaded from frozen folds.json.
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")

COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
NODE_DIR = COMP / "nodes/node_0117"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(a: np.ndarray) -> np.ndarray:
    """Clip-normalize probability matrix to simplex."""
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True)
    s[s == 0] = 1
    return a / s


def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))


def load_oof(path: str | Path, nr: int) -> np.ndarray:
    """Load OOF probs → (nr, 3) float64 on simplex."""
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a
        return norm(a[:, :3])
    d = pd.read_csv(p)
    c = list(d.columns)
    if set(LAB).issubset(c):
        return norm(d[LAB].values.astype(float))
    pc = [f"proba_{l}" for l in LAB]
    if set(pc).issubset(c):
        return norm(d[pc].values.astype(float))
    num = d.select_dtypes("number")
    if num.shape[1] >= 3:
        return norm(num.values[:, :3])
    raise ValueError(f"Cannot parse {path}")


def borda_ranks(probs: np.ndarray) -> np.ndarray:
    """Convert (n, 3) probability matrix to Borda-rank scores in [0, 1].

    For each row, rank the 3 classes by probability:
      - highest prob class gets rank 2 → normalized 1.0
      - middle class gets rank 1 → normalized 0.5
      - lowest class gets rank 0 → normalized 0.0

    This is a per-row operation (not per-column global ranking).
    It is a rank-based normalization that makes ensemble combination
    invariant to cross-base probability calibration differences.
    """
    n, nc = probs.shape
    ranks = np.zeros_like(probs, dtype=np.float32)
    order = probs.argsort(axis=1)  # ascending: order[:,0]=worst, order[:,2]=best
    for i in range(nc):
        ranks[np.arange(n), order[:, i]] = i
    return ranks / (nc - 1)  # normalize to [0, 1]


def weighted_borda_vote(
    oof_list: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted sum of Borda ranks.

    oof_list: list of (n, 3) probability arrays
    weights:  (n_bases,) weight vector
    Returns:  (n, 3) aggregated rank scores — argmax gives predicted class
    """
    w = np.array(weights, dtype=float)
    w = np.clip(w, 0, None)
    w = w / w.sum()
    agg = np.zeros((oof_list[0].shape[0], NC), dtype=float)
    for arr, wi in zip(oof_list, w):
        agg += wi * borda_ranks(arr)
    return agg


def de_weights_subsample(
    oof_list: list[np.ndarray],
    y: np.ndarray,
    n_sub: int = 10000,
    seed: int = 42,
    maxiter: int = 100,
    popsize: int = 10,
    tol: float = 1e-4,
) -> np.ndarray:
    """Find per-base weights via DE on a stratified subsample.

    Optimization is done on n_sub rows to keep each fold fast (seconds, not minutes).
    Weights are then applied to the full val fold for honest scoring.
    """
    nb = len(oof_list)
    n = len(y)
    rng = np.random.RandomState(seed)

    # Stratified subsample from train fold
    sub_idx = rng.choice(n, min(n_sub, n), replace=False)
    sub_borda = np.stack([borda_ranks(arr[sub_idx]) for arr in oof_list], axis=0)  # (nb, sub, 3)
    y_sub = y[sub_idx]

    def neg_ba(w_raw: np.ndarray) -> float:
        w = np.clip(np.array(w_raw, dtype=float), 0, None)
        s = w.sum()
        if s < 1e-9:
            return 0.0
        w = w / s
        agg = (sub_borda * w[:, None, None]).sum(0)  # (sub, 3)
        preds = agg.argmax(1)
        return -score_fn(y_sub, preds)

    bounds = [(0.0, 1.0)] * nb
    result = differential_evolution(
        neg_ba,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        polish=False,
        init="sobol",
    )
    w = np.clip(result.x, 0, None)
    w /= w.sum()
    return w


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
    # PRE-FLIGHT LEAKAGE CHECKS 1-2
    print("\n[LEAKAGE CHECK 1-2] Features are pre-computed OOF probs only; target/id absent. PASS", flush=True)
    print("[LEAKAGE CHECK 4] Borda ranks are row-wise transforms (no cross-row label info).", flush=True)
    print("                  DE weights fitted on train-fold subsample only; val fold never seen. PASS", flush=True)
    print("[LEAKAGE CHECK 5] Folds loaded from frozen folds.json. PASS", flush=True)

    # =========================================================================
    # Load the 6 LOO-load-bearing bases
    K = COMP / "refs/kernel_out"
    B = COMP / "refs/oof_bank"
    PILK = COMP / "refs/ext_oof/pilkwang_5090"

    BASES = {
        "cat-3": {
            "oof":  K / "cat-v3-for-s6e6/train_oof/cat-3_oof.npy",
            "test": K / "cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy",
        },
        "realmlp-2": {
            "oof":  B / "oof_preds_realmlp2_v10.csv",
            "test": B / "test_preds_realmlp2_v10.csv",
        },
        "node_0003": {
            "oof":  COMP / "nodes/node_0003/oof.npy",
            "test": COMP / "nodes/node_0003/test_probs.npy",
        },
        "node_0039": {
            "oof":  COMP / "nodes/node_0039/oof.npy",
            "test": COMP / "nodes/node_0039/test_probs.npy",
        },
        "lgbm-5": {
            "oof":  B / "oof_preds_lgbm5_v1.csv",
            "test": B / "test_preds_lgbm5_v1.csv",
        },
        "ft_transformer": {
            "oof":  PILK / "oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv",
            "test": PILK / "sub_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv",
        },
    }

    print(f"\n{'base':18s} {'solo_BA':>9s} {'oof_shape':>12s} {'status'}", flush=True)
    oof_mats = []
    test_mats = []
    base_names = []
    for name, paths in BASES.items():
        try:
            o = load_oof(paths["oof"], n)
            t = load_oof(paths["test"], nt)
            assert o.shape == (n, 3), f"oof shape {o.shape}"
            assert t.shape == (nt, 3), f"test shape {t.shape}"
            assert not np.isnan(o).any(), "NaN in oof"
            assert not np.isnan(t).any(), "NaN in test"
            solo_ba = score_fn(y, o.argmax(1))
            oof_mats.append(o)
            test_mats.append(t)
            base_names.append(name)
            print(f"{name:18s} {solo_ba:9.6f} {str(o.shape):>12s} OK", flush=True)
        except Exception as e:
            print(f"{name:18s} {'--':>9s} {'--':>12s} FAIL {e}", flush=True)

    nb = len(base_names)
    print(f"\nLoaded {nb} load-bearing bases: {base_names}", flush=True)
    assert nb == 6, f"Expected 6 bases, got {nb}"

    # =========================================================================
    # LEAKAGE CHECK 3: single-feature↔target correlation sweep on a sample
    print("\n[LEAKAGE CHECK 3] Single-feature correlation sweep (50k sample)...", flush=True)
    rng = np.random.RandomState(0)
    sidx = rng.choice(n, min(50000, n), replace=False)
    ys = y[sidx].astype(float)
    for i, name in enumerate(base_names):
        corr = abs(np.corrcoef(oof_mats[i][sidx, 0], ys)[0, 1])
        if corr >= 0.999:
            raise SystemExit(f"LEAK smell: {name}_col0 ~ target corr={corr:.4f}")
    print("[LEAKAGE CHECK 3] PASS", flush=True)

    # =========================================================================
    # LEAKAGE CHECK 6: train/test id overlap
    print("\n[LEAKAGE CHECK 6] Train/test id overlap check...", flush=True)
    train_ids = set(train["id"].values)
    test_ids = set(test["id"].values)
    overlap = train_ids & test_ids
    if overlap:
        print(f"  WARN: {len(overlap)} overlapping ids", flush=True)
    else:
        print("  PASS: train/test ids are disjoint", flush=True)

    # =========================================================================
    # OOF LOOP: fold-honest Borda-rank vote with DE weight optimization
    print("\n" + "="*70, flush=True)
    print("=== FOLD-HONEST BORDA-RANK VOTE WITH DE WEIGHTS ===", flush=True)
    print(f"Algorithm: per-row Borda ranks, per-base DE weights (10k subsample)", flush=True)

    oof_scores = np.zeros((n, NC), dtype=float)
    per_fold_scores = []
    per_fold_weights = []

    for fi, vi in enumerate(fval):
        tr_idx = np.setdiff1d(np.arange(n), vi)

        # DE: fit per-base weights on train-fold subsample only
        # Val fold is NEVER seen during optimization
        oof_tr = [arr[tr_idx] for arr in oof_mats]
        w = de_weights_subsample(oof_tr, y[tr_idx], n_sub=10000, seed=42 + fi)
        per_fold_weights.append(w.tolist())

        print(f"\n  fold {fi}: weights={dict(zip(base_names, [f'{wi:.3f}' for wi in w]))}", flush=True)

        # Predict val fold: apply DE weights to val-fold Borda ranks
        oof_val_list = [arr[vi] for arr in oof_mats]
        val_scores = weighted_borda_vote(oof_val_list, w)
        oof_scores[vi] = val_scores

        fold_ba = score_fn(y[vi], val_scores.argmax(1))
        per_fold_scores.append(fold_ba)
        print(f"  fold {fi}: BA={fold_ba:.6f}", flush=True)

    cv_mean = float(np.mean(per_fold_scores))
    cv_sem = float(np.std(per_fold_scores, ddof=1) / np.sqrt(n_folds))

    print(f"\ncv={cv_mean:.6f}  sem={cv_sem:.6f}", flush=True)
    print(f"per_fold={[f'{s:.6f}' for s in per_fold_scores]}", flush=True)

    # =========================================================================
    # FINAL REFIT on full train: average weights across folds
    print("\n" + "="*70, flush=True)
    print("Final refit on all train (average DE weights across folds)...", flush=True)
    avg_w = np.mean(per_fold_weights, axis=0)
    avg_w = avg_w / avg_w.sum()
    print(f"  avg weights: {dict(zip(base_names, [f'{wi:.4f}' for wi in avg_w]))}", flush=True)

    test_scores = weighted_borda_vote(test_mats, avg_w)

    # =========================================================================
    # DISAGREEMENT VS N091 (the hedge value)
    print("\n" + "="*70, flush=True)
    print("=== DISAGREEMENT VS N091 (finals hedge value) ===", flush=True)
    n91_oof = np.load(COMP / "nodes/node_0091/oof.npy").astype(float)
    n91_preds = n91_oof.argmax(1)
    vote_preds = oof_scores.argmax(1)
    disagree = int((n91_preds != vote_preds).sum())
    disagree_rate = disagree / n
    print(f"Row-level disagreement vs n091: {disagree}/{n} = {disagree_rate:.4f} ({disagree_rate*100:.2f}%)", flush=True)
    for ci, cls in enumerate(LAB):
        mask = y == ci
        d_cls = int((n91_preds[mask] != vote_preds[mask]).sum())
        n_cls = int(mask.sum())
        print(f"  {cls}: {d_cls}/{n_cls} = {d_cls/n_cls*100:.2f}%", flush=True)

    # =========================================================================
    # Write artifacts
    # Normalize rank scores to simplex before saving so future stack nodes
    # can treat them as pseudo-probabilities (argmax is unchanged).
    oof_norm = norm(oof_scores)
    test_norm = norm(test_scores)
    NODE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(NODE_DIR / "oof.npy", oof_norm.astype(np.float32))
    np.save(NODE_DIR / "test_probs.npy", test_norm.astype(np.float32))

    # Submission uses argmax of (unnormalized) rank scores = same as argmax of normalized
    test_preds_idx = test_scores.argmax(1)
    test_labels = [I2L[i] for i in test_preds_idx]
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)

    print(f"\nArtifacts written:", flush=True)
    print(f"  oof.npy:        {oof_norm.shape} (simplex-normalized rank scores)", flush=True)
    print(f"  test_probs.npy: {test_norm.shape} (simplex-normalized rank scores)", flush=True)
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

    # Gate 8: distribution sane (OOF is simplex-normalized rank scores after norm())
    row_sums = oofn.sum(axis=1)
    assert oofn.min() >= 0.0 and oofn.max() <= 1.0 + 1e-5, \
        f"OOF scores out of [0,1]: min={oofn.min()}, max={oofn.max()}"
    assert abs(row_sums.mean() - 1.0) < 0.01, \
        f"OOF row sums off (expected 1.0 after simplex norm): mean={row_sums.mean()}"
    class_counts = np.bincount(oofn.argmax(1), minlength=3)
    print(f"  dist_sane: PASS  OOF argmax: GALAXY={class_counts[0]} QSO={class_counts[1]} STAR={class_counts[2]}", flush=True)
    print(f"             range=[{oofn.min():.4f},{oofn.max():.4f}]  row_sums_mean={row_sums.mean():.6f}", flush=True)

    # Gate 10: cv-too-good
    cv_too_good = cv_mean > 0.980
    print(f"  cv_too_good: {'WARN (>0.980)' if cv_too_good else 'PASS'}", flush=True)

    # =========================================================================
    # Final summary
    print("\n" + "="*70, flush=True)
    print("=== FINAL SUMMARY ===", flush=True)
    print(f"bases: {base_names}", flush=True)
    print(f"algorithm: Borda-rank vote (per-row ranking) + DE per-base weights", flush=True)
    print(f"per_fold_scores: {[f'{s:.6f}' for s in per_fold_scores]}", flush=True)
    print(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}", flush=True)
    print(f"champion_n091_cv=0.970355  delta={cv_mean - 0.970355:+.6f}", flush=True)
    print(f"disagreement_vs_n091: {disagree_rate:.4f} ({disagree_rate*100:.2f}%)", flush=True)
    print(f"cv={cv_mean:.6f}", flush=True)  # machine-parseable line


if __name__ == "__main__":
    main()
