"""node_0053 — multi-seed re-partition meta stack (5x10fold, SEEDS=[42,63,55555,37,47]).

Built on:   node_0041 CORE15 stack (15 base nodes, byte-identical bases + logp +
            balanced-multinomial LogReg meta hyperparams + DE per-class threshold).
            No base model is retrained.

Change:     SINGLE ATOMIC CHANGE — re-partition the meta stack over 5 seeds instead
            of 1 seed. For each seed s in SEEDS: StratifiedKFold(n_splits=10,
            shuffle=True, random_state=s) over the 15-base log-prob OOF matrix and y.
            For each fold: fit balanced LogReg on meta-train, predict_proba on
            meta-val -> fill oof_s[val]; accumulate predict_proba on TEST -> test_s
            (avg over 10 folds). After all seeds: oof = mean(oof_s over seeds);
            test_probs = mean(test_s over seeds).
            DE per-class threshold fit fold-honestly on the AVERAGED oof.

Leakage note: each row's base-OOF feature is honest OOF from the seed-42 base CV
            (held out once there). Re-partitioning only the meta over multiple seeds
            introduces no additional leakage -- the meta never sees the label of the
            row it's predicting within each seed/fold partition.

Metric:     Balanced Accuracy Score (maximize). Parent champion CV = 0.969808.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]

# BYTE-IDENTICAL to node_0041 ------------------------------------------------
BASES = [
    "node_0006", "node_0004", "node_0001", "node_0009",
    "node_0011", "node_0003", "node_0019", "node_0016", "node_0014",
    "node_0028", "node_0032", "node_0035",
    "node_0033", "node_0030",
    "node_0039",
]
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
I2L = {i: l for l, i in L2I.items()}
NC = 3
DIRECTION = "maximize"

# THE ONE ATOMIC CHANGE (was: single seed 42, 5 folds from folds.json)
SEEDS = [42, 63, 55555, 37, 47]
N_FOLDS = 10

# ---------------------------------------------------------------------------
# BYTE-IDENTICAL helpers from node_0041
def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy = mean per-class recall."""
    return float(np.mean(
        [(y_pred[y_true == c] == c).mean() for c in range(NC) if (y_true == c).any()]
    ))

def logp(a: np.ndarray) -> np.ndarray:
    return np.log(np.clip(a, 1e-7, 1.0))

def fit_meta(Xtr: np.ndarray, ytr: np.ndarray) -> LogisticRegression:
    m = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1)
    m.fit(Xtr, ytr)
    return m

def best_thr_de(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Find w=(w_GAL, w_QSO, 1.0) via differential_evolution."""
    def neg(w):
        pred = np.argmax(probs * np.array([w[0], w[1], 1.0]), axis=1)
        return -score_fn(labels, pred)
    r = differential_evolution(
        neg, [(0.1, 5.0), (0.1, 5.0)],
        maxiter=40, tol=1e-7, seed=0, polish=False, workers=1
    )
    return np.array([r.x[0], r.x[1], 1.0])

# ---------------------------------------------------------------------------
def ensure_base_artifacts():
    """Generate any missing test_probs.npy for base nodes."""
    import importlib.util
    helper = Path(__file__).parent / "gen_node0003_test_probs.py"
    spec = importlib.util.spec_from_file_location("gen_node0003", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main():
    ensure_base_artifacts()
    train = pd.read_csv(COMP / "data/train.csv")
    test  = pd.read_csv(COMP / "data/test.csv")
    sample_sub = pd.read_csv(COMP / "data/sample_submission.csv")

    # folds.json used for holdout check and fold-honest DE scoring
    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    holdout_idx = np.asarray(folds_data[-1]["val_idx"])  # fold 4 = inviolable holdout

    n = len(train)
    m_test = len(test)
    y = train["class"].map(L2I).to_numpy()

    # Build stacked OOF feature matrix: (N, 45) -- 15 bases x 3 classes
    nodes_dir = COMP / "nodes"
    OOF = np.concatenate(
        [logp(np.load(nodes_dir / b / "oof.npy")) for b in BASES], axis=1
    )
    print(f"stack OOF shape: {OOF.shape}  ({len(BASES)} bases x {NC} classes)")

    # Build stacked test feature matrix: (M, 45)
    TEST = np.concatenate(
        [logp(np.load(nodes_dir / b / "test_probs.npy")) for b in BASES], axis=1
    )
    print(f"stack TEST shape: {TEST.shape}")

    # ----- MULTI-SEED META AVERAGING (THE ONE ATOMIC CHANGE) -----------------
    oof_sum  = np.zeros((n, NC), dtype=np.float64)
    test_sum = np.zeros((m_test, NC), dtype=np.float64)

    for si, seed in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof_s  = np.zeros((n, NC), dtype=np.float64)
        test_s = np.zeros((m_test, NC), dtype=np.float64)

        for fi, (tr_idx, val_idx) in enumerate(skf.split(OOF, y)):
            meta = fit_meta(OOF[tr_idx], y[tr_idx])
            oof_s[val_idx] = meta.predict_proba(OOF[val_idx])
            test_s += meta.predict_proba(TEST) / N_FOLDS

        seed_ba = score_fn(y, np.argmax(oof_s, axis=1))
        print(f"seed {seed}: OOF BA (argmax, no DE) = {seed_ba:.6f}")
        oof_sum  += oof_s
        test_sum += test_s

    # Average over seeds
    stack_oof        = (oof_sum  / len(SEEDS)).astype(np.float64)
    stack_test_probs = (test_sum / len(SEEDS)).astype(np.float64)
    print(f"avg OOF BA (argmax, no DE) = {score_fn(y, np.argmax(stack_oof, axis=1)):.6f}")

    # ----- FOLD-HONEST DE THRESHOLD SCORING on the AVERAGED OOF --------------
    # Use the frozen 5-fold outer CV (folds.json) for per-fold scores
    fval = [np.asarray(f["val_idx"]) for f in folds_data]

    per_fold_scores = []
    for i, vi in enumerate(fval):
        other = np.setdiff1d(np.arange(n), vi)
        w = best_thr_de(stack_oof[other], y[other])
        pred = np.argmax(stack_oof[vi] * w, axis=1)
        s = score_fn(y[vi], pred)
        per_fold_scores.append(s)
        print(f"fold {i}: score={s:.6f}  w=[{w[0]:.4f},{w[1]:.4f},{w[2]:.4f}]")

    cv_mean = float(np.mean(per_fold_scores))
    cv_sem  = float(np.std(per_fold_scores, ddof=1) / np.sqrt(len(per_fold_scores)))
    print(f"cv={cv_mean:.6f}  sem={cv_sem:.6f}")

    # ----- HOLDOUT HONESTY CHECK (fold 4, inviolable) -------------------------
    # Fit DE on training portion (folds 0-3), evaluate on holdout
    train_for_holdout = np.setdiff1d(np.arange(n), holdout_idx)
    w_holdout = best_thr_de(stack_oof[train_for_holdout], y[train_for_holdout])
    pred_holdout = np.argmax(stack_oof[holdout_idx] * w_holdout, axis=1)
    holdout_ba = score_fn(y[holdout_idx], pred_holdout)
    print(f"HOLDOUT BA (node_0053 multi-seed): {holdout_ba:.6f}")

    # Compare: load node_0041 oof and compute its holdout BA
    oof_0041 = np.load(nodes_dir / "node_0041" / "oof.npy")
    w_0041_holdout = best_thr_de(oof_0041[train_for_holdout], y[train_for_holdout])
    pred_0041_holdout = np.argmax(oof_0041[holdout_idx] * w_0041_holdout, axis=1)
    holdout_ba_0041 = score_fn(y[holdout_idx], pred_0041_holdout)
    print(f"HOLDOUT BA (node_0041 single-seed): {holdout_ba_0041:.6f}")

    # ----- FINAL FIT -- w on full averaged OOF for test predictions ----------
    w_full = best_thr_de(stack_oof, y)
    print(f"final w=[{w_full[0]:.4f},{w_full[1]:.4f},{w_full[2]:.4f}]")

    test_preds_idx = np.argmax(stack_test_probs * w_full, axis=1)
    test_labels = [I2L[i] for i in test_preds_idx]

    # submission
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    print(f"submission written: {len(sub)} rows")

    # save artifacts
    np.save(NODE_DIR / "oof.npy", stack_oof)
    np.save(NODE_DIR / "test_probs.npy", stack_test_probs)

    # features.txt (identical to node_0041)
    feat_names = [f"{b}_p{c}" for b in BASES for c in range(NC)]
    (NODE_DIR / "src" / "features.txt").write_text("\n".join(feat_names) + "\n")
    print(f"features.txt written: {len(feat_names)} features")

    # validate submission matches sample
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count mismatch: {len(sub)} vs {len(sample_sub)}"
    print("submission schema OK")


if __name__ == "__main__":
    main()
