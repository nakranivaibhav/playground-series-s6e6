"""Probe: does the n74 A4-pseudo TabM (CLOUT) help the bank-17 stack?
Reuses n70's bank-17 manifest + meta. Reports bank17, +n74, +n67, +n67+n74.
CLOUT node — restack is informational only (slot-2 diversity), never slot-1/champion."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import differential_evolution

COMP = Path(__file__).resolve().parent.parent
L2I = {"GALAXY": 0, "QSO": 1, "STAR": 2}
EPS = 1e-6

def norm(a):
    a = np.clip(a, 0, None); s = a.sum(1, keepdims=True); s[s == 0] = 1; return a / s
def logp(a):
    return np.log(np.clip(a, EPS, 1.0)).astype(np.float64)
def rd(path, nr):
    p = str(path)
    a = np.load(p) if p.endswith(".npy") else pd.read_csv(p).iloc[:, -3:].to_numpy()
    if a.ndim == 3: a = a.mean(0)
    return a[:nr]
def score_fn(y, p): return balanced_accuracy_score(y, p)
def fit_meta(X, y):
    m = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    m.fit(X, y); return m
def best_thr_de(probs, labels):
    def neg(w): return -balanced_accuracy_score(labels, np.argmax(probs * w, 1))
    r = differential_evolution(neg, [(0.5, 1.5)] * 3, seed=0, maxiter=40, tol=1e-6, polish=True)
    return r.x
def eval_cols(oof_cols, y, fval):
    X = np.concatenate(oof_cols, 1); scores = []
    oof_pred = np.zeros((len(y), 3))
    for vi in fval:
        oth = np.setdiff1d(np.arange(len(y)), vi)
        m = fit_meta(X[oth], y[oth])
        pv = m.predict_proba(X[vi]); oof_pred[vi] = pv
        w = best_thr_de(m.predict_proba(X[oth]), y[oth])
        scores.append(balanced_accuracy_score(y[vi], np.argmax(pv * w, 1)))
    return float(np.mean(scores)), float(np.std(scores) / np.sqrt(len(scores))), scores

def main():
    train = pd.read_csv(COMP / "data/train.csv")
    folds = json.loads((COMP / "folds.json").read_text())["folds"]
    n = len(train); y = train["class"].map(L2I).to_numpy()
    fval = [np.asarray(f["val_idx"]) for f in folds]
    B = COMP / "refs/oof_bank"; K = COMP / "refs/kernel_out"
    MANIFEST = {
        'xgb-1': K/"xgb-v1-for-s6e6/oof_preds.npy", 'realmlp-0': B/"oof_preds_realmlp0_v12.csv",
        'realmlp-1': K/"realmlp-v1-for-s6e6/oof_preds.npy", 'tabm-0': B/"oof_preds_tabm0_v2.csv",
        'cat-0': K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", 'realmlp-2': B/"oof_preds_realmlp2_v10.csv",
        'tabicl-2': K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", 'lgbm-3': K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",
        'logreg-1': K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy", 'nn-1': K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",
        'xgb-5': K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy", 'realmlp-5': K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",
        'nn-2': K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy", 'cat-3': K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",
        'lgbm-5': B/"oof_preds_lgbm5_v1.csv", 'xgb-6': B/"oof_final_xgb6_v1.csv", 'tabm-1': B/"oof_final_tabm1_v1.csv",
    }
    bank = []
    for name, op in MANIFEST.items():
        o = norm(rd(op, n)); ba = balanced_accuracy_score(y, o.argmax(1))
        if 0.90 < ba < 0.972: bank.append(logp(o))
    print(f"bank bases loaded: {len(bank)}")
    base_cv, base_sem, _ = eval_cols(bank, y, fval)
    print(f"bank-{len(bank)} baseline cv={base_cv:.6f} sem={base_sem:.6f}  (expect 0.970153)")
    n74 = logp(norm(np.load(COMP / "nodes/node_0074/oof.npy")[:n]))
    n67 = logp(norm(np.load(COMP / "nodes/node_0067/oof.npy")[:n]))
    for tag, cols in [("bank+n74", bank+[n74]), ("bank+n67", bank+[n67]), ("bank+n67+n74", bank+[n67, n74])]:
        cv, sem, _ = eval_cols(cols, y, fval)
        print(f"{tag:14s} cv={cv:.6f} sem={sem:.6f}  delta={cv-base_cv:+.6f}  beats2sem={cv-base_cv > 2*base_sem}")

if __name__ == "__main__":
    main()
