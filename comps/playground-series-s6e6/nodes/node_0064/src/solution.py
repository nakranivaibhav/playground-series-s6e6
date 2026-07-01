"""node_0064: recover xgb-0/xgb-3 into bank-19 stack.
Byte-identical meta recipe to node_0063 (balanced-LogReg + DE per-class threshold).
Base set grows 17 -> 18 (xgb-0) or 19 (xgb-0 + xgb-3).

A/B attribution:
  bank-17  (parent baseline)
  bank-18  (+xgb-0 only)
  bank-19  (+xgb-0 +xgb-3)
  bank-17+xgb3  (+xgb-3 only)
"""
import json, warnings, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import differential_evolution

warnings.filterwarnings("ignore")

# ---- Paths ----------------------------------------------------------------
ROOT = Path("comps/playground-series-s6e6")
NODE_DIR = ROOT / "nodes/node_0064"
C = ROOT
B = C / "refs/oof_bank"
K = C / "refs/kernel_out"
LAB = ["GALAXY", "QSO", "STAR"]
L2I = {l: i for i, l in enumerate(LAB)}
NC = 3

# ---- Data -----------------------------------------------------------------
tr = pd.read_csv(C / "data/train.csv")
te = pd.read_csv(C / "data/test.csv")
n = len(tr); nt = len(te)
y = tr["class"].map(L2I).to_numpy()
folds = json.loads((C / "folds.json").read_text())["folds"]
fval = [np.asarray(f["val_idx"]) for f in folds]
print(f"train={n} test={nt} folds={len(folds)}")

# ---- Helpers ---------------------------------------------------------------
def norm(a):
    a = np.clip(a, 0, None)
    s = a.sum(1, keepdims=True); s[s == 0] = 1
    return a / s

def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def logp(a): return np.log(np.clip(a, 1e-7, 1.0))

def rd(path, nr):
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a
        return a[:, :3]
    d = pd.read_csv(p); c = list(d.columns)
    if set(LAB).issubset(c): return d[LAB].values.astype(float)
    pc = [f"prob_{l}" for l in LAB]
    if set(pc).issubset(c): return d[pc].values.astype(float)
    num = d.select_dtypes("number")
    if num.shape[1] >= 3: return num.values[:, :3]
    v = d.iloc[:, 0].values.astype(float); return v.reshape(nr, 3)

def balacc(yy, pred):
    return float(np.mean([(pred[yy == c] == c).mean() for c in range(NC) if (yy == c).any()]))

def de_thr(P, yy):
    f = lambda w: -balacc(yy, np.argmax(P * np.array([w[0], w[1], 1.0]), 1))
    r = differential_evolution(f, [(0.1, 5.0), (0.1, 5.0)], maxiter=40, tol=1e-7, seed=0, polish=False)
    return np.array([r.x[0], r.x[1], 1.0])

def eval_cv(cols):
    OOF = np.concatenate(cols, 1)
    stack = np.zeros((n, NC))
    for vi in fval:
        trr = np.setdiff1d(np.arange(n), vi)
        stack[vi] = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=2000, n_jobs=-1
        ).fit(OOF[trr], y[trr]).predict_proba(OOF[vi])
    pf = []
    for vi in fval:
        oth = np.setdiff1d(np.arange(n), vi)
        w = de_thr(stack[oth], y[oth])
        pf.append(balacc(y[vi], np.argmax(stack[vi] * w, 1)))
    cv_mean = float(np.mean(pf))
    cv_sem = float(np.std(pf, ddof=1) / np.sqrt(len(pf)))
    return cv_mean, cv_sem, pf, stack

# ---- Load bank-17 (exactly as node_0063) ----------------------------------
M17 = {
    "xgb-1":      (K/"xgb-v1-for-s6e6/oof_preds.npy",         K/"xgb-v1-for-s6e6/test_preds.npy"),
    "realmlp-0":  (B/"oof_preds_realmlp0_v12.csv",              B/"test_preds_realmlp0_v12.csv"),
    "realmlp-1":  (K/"realmlp-v1-for-s6e6/oof_preds.npy",       K/"realmlp-v1-for-s6e6/test_preds.npy"),
    "tabm-0":     (B/"oof_preds_tabm0_v2.csv",                  B/"test_preds_tabm0_v2.csv"),
    "cat-0":      (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv", K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
    "realmlp-2":  (B/"oof_preds_realmlp2_v10.csv",               B/"test_preds_realmlp2_v10.csv"),
    "tabicl-2":   (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy", K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
    "lgbm-3":     (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",    K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
    "logreg-1":   (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy", K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
    "nn-1":       (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",         K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
    "xgb-5":      (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",      K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
    "realmlp-5":  (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy", K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
    "nn-2":       (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",         K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
    "cat-3":      (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",       K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
    "lgbm-5":     (B/"oof_preds_lgbm5_v1.csv",                  B/"test_preds_lgbm5_v1.csv"),
    "xgb-6":      (B/"oof_final_xgb6_v1.csv",                   B/"test_final_xgb6_v1.csv"),
    "tabm-1":     (B/"oof_final_tabm1_v1.csv",                   B/"test_final_tabm1_v1.csv"),
}

print(f"\n{'model':12s} {'oofBA':>9s} {'shape':>12s} {'status'}")
POOF = {}; PTEST = {}
for name, (op, tp) in M17.items():
    try:
        o = norm(rd(op, n)); t = norm(rd(tp, nt))
        assert o.shape == (n, 3) and t.shape == (nt, 3), f"shape mismatch o={o.shape} t={t.shape}"
        ba = balanced_accuracy_score(y, o.argmax(1))
        st = "OK" if 0.90 < ba < 0.972 else ("QUARANTINE" if ba >= 0.972 else "LOW?")
        if st == "OK": POOF[name] = o; PTEST[name] = t
        print(f"{name:12s} {ba:9.6f} {str(o.shape):>12s} {st}")
    except Exception as e:
        print(f"{name:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:60]}")

good17 = list(POOF.keys())
print(f"\nloaded {len(good17)}/17 bank-17 models OK")

# ---- Load xgb-0 -----------------------------------------------------------
# xgb-0: OOF has 677,347 rows (100k original SDSS appended AT END).
# Keep first 577,347 rows which align positionally to our train.csv.
xgb0_oof_raw = pd.read_csv(K/"xgb-v0-for-s6e6/oof_xgb_cv.csv")
assert len(xgb0_oof_raw) == 677347, f"xgb-0 oof row count={len(xgb0_oof_raw)}, expected 677347"
xgb0_oof = xgb0_oof_raw.iloc[:n][["prob_GALAXY", "prob_QSO", "prob_STAR"]].values.astype(float)
xgb0_test = pd.read_csv(K/"xgb-v0-for-s6e6/test_xgb_preds.csv")
# test file: check columns
xgb0_test_arr = xgb0_test[["prob_GALAXY", "prob_QSO", "prob_STAR"]].values.astype(float) if set(["prob_GALAXY","prob_QSO","prob_STAR"]).issubset(xgb0_test.columns) else xgb0_test.select_dtypes("number").values[:, :3]
xgb0_oof = norm(xgb0_oof); xgb0_test_arr = norm(xgb0_test_arr)
assert xgb0_oof.shape == (n, 3), f"xgb-0 oof shape after strip={xgb0_oof.shape}"
assert xgb0_test_arr.shape == (nt, 3), f"xgb-0 test shape={xgb0_test_arr.shape}"
ba_xgb0 = balanced_accuracy_score(y, xgb0_oof.argmax(1))
print(f"\nxgb-0: BA={ba_xgb0:.6f} oof={xgb0_oof.shape} (stripped 100k orig rows from end) test={xgb0_test_arr.shape}")

# ---- Load xgb-3 -----------------------------------------------------------
# xgb-3: raw margin logits, 577347 rows, class order is [GALAXY, STAR, QSO].
# Apply softmax then reorder columns to [GALAXY, QSO, STAR].
xgb3_raw_oof = np.load(K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy").astype(float)
xgb3_raw_test = np.load(K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy").astype(float)
assert xgb3_raw_oof.shape == (n, 3), f"xgb-3 oof shape={xgb3_raw_oof.shape}"
xgb3_oof = softmax(xgb3_raw_oof)[:, [0, 2, 1]]  # reorder: GALAXY,STAR,QSO -> GALAXY,QSO,STAR
xgb3_test = softmax(xgb3_raw_test)[:, [0, 2, 1]]
ba_xgb3 = balanced_accuracy_score(y, xgb3_oof.argmax(1))
print(f"xgb-3: BA={ba_xgb3:.6f} oof={xgb3_oof.shape} (softmax + col perm [0,2,1]) test={xgb3_test.shape}")

# Verify xgb-3 is sane (~0.96, not ~0.33)
assert 0.95 < ba_xgb3 < 0.975, f"xgb-3 BA not sane: {ba_xgb3:.6f} — check column order permutation"

# ---- Pre-flight leakage checks (before any training) ----------------------
print("\n=== Pre-flight leakage checks ===")
# 1. target not in features: we use stacked OOF probs, not raw features
# 2. id not in features: we use positional numpy arrays, no id column present
print("1-2: target/id not in feature arrays: OK (stacked OOF probs only)")
# 3. single-feature corr sweep: logp probs are not going to be near-perfect corr with y
# quick check on xgb0 logp cols vs y
sample_idx = np.random.RandomState(0).choice(n, min(50000, n), replace=False)
for col_i in range(3):
    x = logp(xgb0_oof)[sample_idx, col_i]
    ys = (y[sample_idx] == col_i).astype(float)
    corr = abs(np.corrcoef(x, ys)[0, 1])
    assert corr < 0.999, f"leak smell: xgb0 col {col_i} corr={corr:.4f}"
print("3: single-feature corr sweep: OK")
# 4. fit-inside-fold: by code inspection, LogReg is fit on trr (train-fold) only inside eval_cv loop
print("4: fit-inside-fold: OK (LogReg fit on trr inside loop, DE thr on oth)")
# 5. frozen folds: loaded from folds.json
print("5: frozen folds from folds.json: OK")
# 6. train/test near-dups: not applicable (we're working on OOF probs not raw features)
print("6: train/test near-dups: N/A for OOF probability arrays")
print("=== Pre-flight checks PASSED ===\n")

# ---- A/B variants ---------------------------------------------------------
bank17_oof = [logp(POOF[k]) for k in good17]
bank17_test = [logp(PTEST[k]) for k in good17]

variants = {
    "bank-17": (bank17_oof, bank17_test),
    "bank-18 (+xgb-0)": (bank17_oof + [logp(xgb0_oof)], bank17_test + [logp(xgb0_test_arr)]),
    "bank-17+xgb-3": (bank17_oof + [logp(xgb3_oof)], bank17_test + [logp(xgb3_test)]),
    "bank-19 (+xgb-0+xgb-3)": (bank17_oof + [logp(xgb0_oof), logp(xgb3_oof)], bank17_test + [logp(xgb0_test_arr), logp(xgb3_test)]),
}

print("=== A/B CV evaluation ===")
results = {}
for vname, (co, ct) in variants.items():
    cv_mean, cv_sem, pf, stack = eval_cv(co)
    print(f"{vname:28s} cv={cv_mean:.6f} sem={cv_sem:.6f} folds={[round(x,6) for x in pf]}")
    results[vname] = (cv_mean, cv_sem, pf, co, ct, stack)

# ---- Pick best variant ----------------------------------------------------
best_name = max(results, key=lambda k: results[k][0])
best_cv, best_sem, best_folds, best_co, best_ct, best_stack = results[best_name]
print(f"\nBEST: {best_name} cv={best_cv:.6f} sem={best_sem:.6f}")
print(f"Per-fold: {best_folds}")

# ---- Final OOF (fold-honest) for output -----------------------------------
# Recompute best_stack properly (already computed above)
# For submission: refit on all train
OOF_mat = np.concatenate(best_co, 1)
TST_mat = np.concatenate(best_ct, 1)
meta_final = LogisticRegression(class_weight="balanced", C=1.0, max_iter=3000, n_jobs=-1).fit(OOF_mat, y)
stack_full_oof = meta_final.predict_proba(OOF_mat)
stack_test_final = meta_final.predict_proba(TST_mat)
w_final = de_thr(stack_full_oof, y)
print(f"DE weights (final): {w_final}")

# ---- Save artifacts -------------------------------------------------------
np.save(NODE_DIR / "oof.npy", best_stack)           # fold-honest OOF, shape (n, 3)
np.save(NODE_DIR / "test_probs.npy", stack_test_final)  # test probs, shape (nt, 3)

# Submission
pred = np.argmax(stack_test_final * w_final, 1)
sub = pd.DataFrame({"id": te["id"], "class": [LAB[i] for i in pred]})
sub.to_csv(NODE_DIR / "submission.csv", index=False)
print(f"\nclass dist: {sub['class'].value_counts().to_dict()}")
print(f"submission rows: {len(sub)}")

# ---- Final cv= line -------------------------------------------------------
print(f"\ncv={best_cv:.6f}")
for i, s in enumerate(best_folds):
    print(f"fold_{i}={s:.6f}")
print(f"best_variant={best_name}")
