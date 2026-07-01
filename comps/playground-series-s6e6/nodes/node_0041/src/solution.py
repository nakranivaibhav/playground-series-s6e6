"""node_0020 — balanced LogReg STACKER + DE threshold (combine, ensemble).

Built on:   OOF probability matrices of 9 base nodes — node_0006, node_0004,
            node_0001, node_0009, node_0011, node_0003, node_0019, node_0016,
            node_0014. No base model is retrained; this node only fits a
            meta-model on top of their saved oof.npy / test_probs.npy.

Change:     STACKER (not a probability-average blend). Meta features =
            log(clip(p, 1e-7, 1)) of each base's OOF probs, concatenated
            → 9 bases × 3 classes = 27 columns. Meta model = sklearn
            LogisticRegression(class_weight='balanced', C=1.0, max_iter=2000,
            n_jobs=-1) — multinomial by default, no multi_class kwarg needed.
            Fold-honest: for each held-out fold, meta is fit on the OTHER 4
            folds' stacked features, then predicts the held fold → honest
            stacked OOF. Per-class threshold calibration via
            scipy.optimize.differential_evolution over w=(w_GAL, w_QSO, 1.0),
            bounds (0.1, 5), maximizing balanced accuracy, fit on the OTHER
            folds' stacked OOF and applied to the held fold via
            argmax(prob * w). For TEST: meta fit on full stacked OOF, w fit
            on full stacked OOF, applied to stacked base test_probs.

Metric:     Balanced Accuracy Score (maximize). Expected honest CV ~0.966627.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
COMP = Path(__file__).resolve().parents[3]   # comps/playground-series-s6e6
NODE_DIR = Path(__file__).resolve().parents[1]
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

# ---------------------------------------------------------------------------
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

    folds_data = json.loads((COMP / "folds.json").read_text())["folds"]
    n = len(train)
    y = train["class"].map(L2I).to_numpy()
    fval = [np.asarray(f["val_idx"]) for f in folds_data]

    # Build stacked OOF feature matrix: (N, 27)
    nodes_dir = COMP / "nodes"
    OOF = np.concatenate(
        [logp(np.load(nodes_dir / b / "oof.npy")) for b in BASES], axis=1
    )
    print(f"stack OOF shape: {OOF.shape}  ({len(BASES)} bases × {NC} classes)")

    # Build stacked test feature matrix: (M, 27)
    TEST = np.concatenate(
        [logp(np.load(nodes_dir / b / "test_probs.npy")) for b in BASES], axis=1
    )
    print(f"stack TEST shape: {TEST.shape}")

    # ----- fold-honest stacked OOF (meta fit on other 4 folds) ---------------
    stack_oof = np.zeros((n, NC))
    for vi in fval:
        tr = np.setdiff1d(np.arange(n), vi)
        m = fit_meta(OOF[tr], y[tr])
        stack_oof[vi] = m.predict_proba(OOF[vi])

    # ----- fold-honest DE threshold scoring ----------------------------------
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

    # ----- fit meta on full OOF, w on full stacked OOF -----------------------
    meta_full = fit_meta(OOF, y)
    stack_oof_full_check = meta_full.predict_proba(OOF)  # sanity only
    w_full = best_thr_de(stack_oof, y)
    print(f"final w=[{w_full[0]:.4f},{w_full[1]:.4f},{w_full[2]:.4f}]")

    # ----- test predictions --------------------------------------------------
    stack_test_probs = meta_full.predict_proba(TEST)
    test_preds_idx = np.argmax(stack_test_probs * w_full, axis=1)
    test_labels = [I2L[i] for i in test_preds_idx]

    # submission
    sub = pd.DataFrame({"id": test["id"], "class": test_labels})
    sub.to_csv(NODE_DIR / "submission.csv", index=False)
    print(f"submission written: {len(sub)} rows")

    # save oof.npy and test_probs.npy (stacked probs before threshold)
    np.save(NODE_DIR / "oof.npy", stack_oof)
    np.save(NODE_DIR / "test_probs.npy", stack_test_probs)

    # features.txt
    feat_names = [f"{b}_p{c}" for b in BASES for c in range(NC)]
    (NODE_DIR / "src" / "features.txt").write_text("\n".join(feat_names) + "\n")
    print(f"features.txt written: {len(feat_names)} features")

    # validate submission matches sample
    assert list(sub.columns) == list(sample_sub.columns), "column mismatch"
    assert len(sub) == len(sample_sub), f"row count mismatch: {len(sub)} vs {len(sample_sub)}"
    print("submission schema OK")


if __name__ == "__main__":
    main()
