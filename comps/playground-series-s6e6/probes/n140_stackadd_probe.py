"""PROBE: does n140 (RealMLP + fs_zsoft) add anything to the n091 FULL-pool stack?

Re-fits the EXACT n091 mega-stack (bank-17 + FT-T + 36 TIGHT + 9 WEAK in-house,
balanced multinomial LogReg, nested LogisticRegressionCV C-grid on frozen folds).
ARM A = the champion FULL pool.  ARM B = FULL pool + n140 OOF (the fs_zsoft base).
Then a paired row-bootstrap on the OOF argmax to estimate P(B > A) — the canonical
keep/combine gate (>=0.90). CPU-only, no submission written. Verbatim helpers +
MANIFEST copied from champion/src/solution.py.
"""
import json, warnings
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")
COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
LAB = ["GALAXY", "QSO", "STAR"]; L2I = {l: i for i, l in enumerate(LAB)}; NC = 3
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
TIGHT_IDS = [1, 3, 4, 5, 6, 9, 11, 12, 13, 15, 16, 18, 19, 23,
             28, 30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45,
             49, 50, 51, 55, 56, 60, 61, 66, 85]
WEAK_EXTRA_IDS = [8, 21, 22, 24, 25, 26, 27, 37, 62]

def logp(a): return np.log(np.clip(a, 1e-7, 1.0))
def norm(a):
    a = np.clip(a, 0, None); s = a.sum(1, keepdims=True); s[s == 0] = 1; return a / s
def score_fn(yt, yp):
    return float(np.mean([(yp[yt == c] == c).mean() for c in range(NC) if (yt == c).any()]))
def rd(path, nr):
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p, allow_pickle=True).astype(float)
        a = a.reshape(nr, -1) if a.ndim == 1 else a; return a[:, :3]
    d = pd.read_csv(p); c = list(d.columns)
    if set(LAB).issubset(c): return d[LAB].values.astype(float)
    pc = [f"prob_{l}" for l in LAB]
    if set(pc).issubset(c): return d[pc].values.astype(float)
    num = d.select_dtypes("number")
    if num.shape[1] >= 3: return num.values[:, :3]
    return d.iloc[:, 0].values.astype(float).reshape(nr, 3)
def load_ext_csv(path, nr):
    d = pd.read_csv(path); pcols = ["proba_GALAXY", "proba_QSO", "proba_STAR"]
    if set(pcols).issubset(d.columns): return d[pcols].values.astype(float)
    return rd(path, nr)

def nested_cv_arm(OOF_mat, y, fval, label):
    n = OOF_mat.shape[0]; oof = np.zeros((n, NC)); pf = []; Cs = []
    print(f"\n=== ARM {label}  cols={OOF_mat.shape[1]} ({OOF_mat.shape[1]//3} bases) ===", flush=True)
    for fi, vi in enumerate(fval):
        tr = np.setdiff1d(np.arange(n), vi)
        lrcv = LogisticRegressionCV(Cs=C_GRID, cv=4, class_weight="balanced",
                                    max_iter=2000, n_jobs=-1, random_state=42,
                                    solver="lbfgs", multi_class="multinomial", scoring="balanced_accuracy")
        lrcv.fit(OOF_mat[tr], y[tr])
        Cs.append(float(lrcv.C_[0])); oof[vi] = lrcv.predict_proba(OOF_mat[vi])
        s = score_fn(y[vi], oof[vi].argmax(1)); pf.append(s)
        print(f"  fold {fi}: BA={s:.6f} C={lrcv.C_[0]}", flush=True)
    cv = float(np.mean(pf)); sem = float(np.std(pf, ddof=1) / np.sqrt(len(fval)))
    print(f"  {label}: cv={cv:.6f} sem={sem:.6f} per_fold={[f'{s:.6f}' for s in pf]}", flush=True)
    return oof, pf, cv, sem

# ---- load data + folds
train = pd.read_csv(COMP / "data/train.csv")
y = train["class"].map(L2I).values
n = len(y); nt = len(pd.read_csv(COMP / "data/test.csv"))
fval = [np.asarray(f["val_idx"]) for f in json.loads((COMP / "folds.json").read_text())["folds"]]
print(f"n_train={n} n_test={nt} folds={len(fval)}", flush=True)

# ---- bank-17
B = COMP / "refs/oof_bank"; K = COMP / "refs/kernel_out"
MANIFEST = {
    'xgb-0': (K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",),
    'xgb-1': (K/"xgb-v1-for-s6e6/oof_preds.npy",),
    'realmlp-0': (B/"oof_preds_realmlp0_v12.csv",),
    'realmlp-1': (K/"realmlp-v1-for-s6e6/oof_preds.npy",),
    'tabm-0': (B/"oof_preds_tabm0_v2.csv",),
    'cat-0': (K/"cat-v0-for-s6e6/catboost_oof_predictions.csv",),
    'realmlp-2': (B/"oof_preds_realmlp2_v10.csv",),
    'tabicl-2': (K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy",),
    'lgbm-3': (K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",),
    'logreg-1': (K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",),
    'nn-1': (K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",),
    'xgb-3': (K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy",),
    'xgb-5': (K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",),
    'realmlp-5': (K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",),
    'nn-2': (K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",),
    'cat-3': (K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",),
    'lgbm-5': (B/"oof_preds_lgbm5_v1.csv",),
    'xgb-6': (B/"oof_final_xgb6_v1.csv",),
    'tabm-1': (B/"oof_final_tabm1_v1.csv",),
}
bank = []
for name, (op,) in MANIFEST.items():
    try:
        o = norm(rd(op, n)); assert o.shape == (n, 3)
        ba = balanced_accuracy_score(y, o.argmax(1))
        if 0.90 < ba < 0.972:
            bank.append(logp(o))
    except Exception as e:
        print(f"  bank {name} FAIL {str(e)[:50]}", flush=True)
print(f"bank-17 loaded: {len(bank)}", flush=True)

# ---- FT-T external
PILK = COMP / "refs/ext_oof/pilkwang_5090"
ftt = logp(norm(load_ext_csv(PILK/"oof_ft_transformer_lite_seed42_full_fullrows_fullorig_5fold.csv", n)))
bank.append(ftt); print("FT-T appended", flush=True)

# ---- in-house TIGHT + WEAK
def load_inhouse(ids):
    out = []
    for nid in ids:
        nm = f"node_{nid:04d}"
        try:
            o = norm(np.load(COMP/"nodes"/nm/"oof.npy").astype(float)); assert o.shape == (n, 3)
            if score_fn(y, o.argmax(1)) < 0.5: continue
            out.append(logp(o))
        except Exception as e:
            print(f"  inhouse {nm} FAIL {str(e)[:50]}", flush=True)
    return out
tight = load_inhouse(TIGHT_IDS); weak = load_inhouse(WEAK_EXTRA_IDS)
print(f"in-house TIGHT={len(tight)} WEAK={len(weak)}", flush=True)

# ---- n140 OOF (the fs_zsoft base under test)
n140 = logp(norm(np.load(COMP/"nodes/node_0140/oof.npy").astype(float)))
print(f"n140 solo BA={score_fn(y, np.load(COMP/'nodes/node_0140/oof.npy').astype(float).argmax(1)):.6f}", flush=True)

# ---- ARMS
OOF_A = np.concatenate(bank + tight + weak, axis=1)          # champion FULL pool
OOF_B = np.concatenate(bank + tight + weak + [n140], axis=1) # + n140
oof_A, pf_A, cv_A, sem_A = nested_cv_arm(OOF_A, y, fval, "A_FULL(champ-pool)")
oof_B, pf_B, cv_B, sem_B = nested_cv_arm(OOF_B, y, fval, "B_FULL+n140")

# ---- paired row-bootstrap on OOF argmax: P(B > A)
predA = oof_A.argmax(1); predB = oof_B.argmax(1)
rng = np.random.RandomState(0); Bk = 3000; wins = 0; deltas = []
for _ in range(Bk):
    idx = rng.randint(0, n, n)
    dA = score_fn(y[idx], predA[idx]); dB = score_fn(y[idx], predB[idx])
    deltas.append(dB - dA); wins += (dB > dA)
deltas = np.array(deltas); P = wins / Bk
print("\n" + "="*60, flush=True)
print("STACK-ADD RESULT (n140 / fs_zsoft into champion FULL pool)", flush=True)
print(f"  ARM A FULL champ-pool : cv={cv_A:.6f} sem={sem_A:.6f}", flush=True)
print(f"  ARM B FULL + n140     : cv={cv_B:.6f} sem={sem_B:.6f}", flush=True)
print(f"  delta(B-A)            : {cv_B - cv_A:+.6f}", flush=True)
print(f"  bootstrap P(B>A)      : {P:.3f}  (keep/combine bar >= 0.90)", flush=True)
print(f"  bootstrap delta 95%CI : [{np.percentile(deltas,2.5):+.6f}, {np.percentile(deltas,97.5):+.6f}]", flush=True)
print(f"  VERDICT: {'KEEP n140 (fs_zsoft adds stack value)' if P >= 0.90 else 'WASH — fs_zsoft adds nothing to the stack'}", flush=True)
