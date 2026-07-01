"""node_0102 — Saerens-Latinne-Decaestecker EM test-prior correction (improve n091).

Post-process the champion n091 stacked probabilities with the SLD EM label-shift
algorithm: estimate the (unlabeled) target-set class prior from its own probability
matrix via fixed-point EM, rescale posteriors, argmax.

ONE atomic change vs n091 (plain argmax). NO training, NO labels used to fit anything.

HONEST eval: the stratified OOF folds carry NO prior shift by construction, so the
per-fold OOF delta likely reads ~0. The DECISIVE signal is the EM-estimated TEST prior
vs the train prior — logged and surfaced. Kill if test prior within 1% of train on
every class.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

NODE = Path(__file__).resolve().parent.parent
COMP = NODE.parent.parent
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {c: i for i, c in enumerate(LAB)}


def saerens_em(probs: np.ndarray, train_prior: np.ndarray, max_iter: int = 1000,
               tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Return (corrected_probs, estimated_target_prior).

    probs: (n,k) classifier posteriors estimated under train_prior.
    Fixed-point: p_new(y) propto train_prior * mean_x[ p(y|x)*r(y) ] where
    r(y)=p_new(y)/train_prior(y); the per-row posterior is renormalized each step.
    """
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum(1, keepdims=True)
    cur = train_prior.copy()
    for _ in range(max_iter):
        ratio = cur / train_prior                       # (k,)
        num = probs * ratio[None, :]                    # (n,k) un-normalized adjusted posterior
        adj = num / num.sum(1, keepdims=True)           # renormalized per row
        new = adj.mean(0)                               # new prior = mean adjusted posterior
        new = new / new.sum()
        if np.max(np.abs(new - cur)) < tol:
            cur = new
            break
        cur = new
    ratio = cur / train_prior
    num = probs * ratio[None, :]
    corrected = num / num.sum(1, keepdims=True)
    return corrected, cur


def main():
    train = pd.read_csv(COMP / "data" / "train.csv")
    y = train["class"].map(L2I).to_numpy()
    n = len(y)
    oof = np.load(COMP / "champion" / "oof.npy")          # (n,3) stacked probs
    test = np.load(COMP / "champion" / "test_probs.npy")  # (n_test,3)
    assert oof.shape[0] == n, (oof.shape, n)

    folds = json.loads((COMP / "folds.json").read_text())
    vis = [np.asarray(f["val_idx"], dtype=int) for f in folds["folds"]]

    train_prior = np.bincount(y, minlength=3).astype(float)
    train_prior /= train_prior.sum()
    print(f"train prior        : {train_prior}", flush=True)

    # ---- baseline OOF BA (plain argmax) ----
    base_oof_ba = balanced_accuracy_score(y, oof.argmax(1))
    print(f"\nbaseline OOF BA (plain argmax) : {base_oof_ba:.6f}", flush=True)

    # ---- honest per-fold EM on OOF (treat each val fold as the unlabeled target) ----
    em_oof = oof.copy()
    fold_base, fold_em = [], []
    for k, vi in enumerate(vis):
        p = oof[vi]
        corr, est = saerens_em(p, train_prior)
        em_oof[vi] = corr
        bb = balanced_accuracy_score(y[vi], p.argmax(1))
        be = balanced_accuracy_score(y[vi], corr.argmax(1))
        fold_base.append(bb); fold_em.append(be)
        print(f"  fold {k}: base={bb:.6f}  EM={be:.6f}  d={be-bb:+.6f}  est_prior={np.round(est,4)}", flush=True)

    em_oof_ba = balanced_accuracy_score(y, em_oof.argmax(1))
    folds_em = [float(x) for x in fold_em]
    cv = float(np.mean(folds_em))
    sem = float(np.std(folds_em, ddof=1) / np.sqrt(len(folds_em)))
    print(f"\nEM OOF BA (per-fold corrected) : {em_oof_ba:.6f}", flush=True)
    print(f"CV(mean per-fold EM)={cv:.6f}  sem={sem:.6f}  folds={[f'{x:.6f}' for x in folds_em]}", flush=True)
    print(f"delta vs baseline OOF BA: {em_oof_ba - base_oof_ba:+.6f}", flush=True)

    # ---- the DECISIVE signal: EM-estimated TEST prior vs train prior ----
    test_corr, test_prior = saerens_em(test, train_prior)
    print(f"\n=== TEST-PRIOR SIGNAL ===", flush=True)
    print(f"train prior        : {np.round(train_prior,5)}", flush=True)
    print(f"EM test prior      : {np.round(test_prior,5)}", flush=True)
    shift = test_prior - train_prior
    rel = shift / train_prior
    print(f"abs shift          : {np.round(shift,5)}", flush=True)
    print(f"rel shift          : {np.round(rel,4)}", flush=True)
    max_rel = float(np.max(np.abs(rel)))
    print(f"max |rel shift|    : {max_rel:.4f}  (KILL if < 0.01 on every class)", flush=True)
    print(f"LEVER {'LIVE' if max_rel >= 0.01 else 'MOOT'}", flush=True)

    # argmax-change counts on test from the correction
    chg = int((test_corr.argmax(1) != test.argmax(1)).sum())
    print(f"test argmax changes from EM correction: {chg} / {len(test)}", flush=True)

    # ---- write artifacts (EM-corrected) ----
    np.save(NODE / "oof.npy", em_oof.astype(np.float32))
    np.save(NODE / "test_probs.npy", test_corr.astype(np.float32))
    samp = pd.read_csv(COMP / "data" / "sample_submission.csv")
    samp[samp.columns[1]] = [LAB[i] for i in test_corr.argmax(1)]
    samp.to_csv(NODE / "submission.csv", index=False)
    print(f"\nwrote oof.npy test_probs.npy submission.csv", flush=True)
    print(f"RESULT cv={cv:.6f} sem={sem:.6f} max_rel_shift={max_rel:.4f}", flush=True)


if __name__ == "__main__":
    main()
