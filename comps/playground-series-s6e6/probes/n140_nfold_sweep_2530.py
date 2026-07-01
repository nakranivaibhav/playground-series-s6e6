"""PROBE: does more CV folds help the RealMLP (n140) standalone submission?

Runs the verified n140 recipe at n_folds = 10, 15, 20 (vs the frozen 5-fold).
More folds = more data per model (90/93/95% vs 80%) + more models averaged for the
test preds (less variance). Reports OOF balanced accuracy per setting and saves a
submission + test_probs for each. Reuses the verified kernel's functions verbatim
(no model change) so each fold is byte-faithful to n140.
"""
import importlib.util, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

COMP = Path("/home/vaibhav/projects/personal/grandmaster/comps/playground-series-s6e6")
PROBE_DIR = COMP / "probes"

# import the verified kernel's functions (main() is __main__-guarded, so import is safe)
spec = importlib.util.spec_from_file_location("krn", COMP / "nodes/node_0140/kaggle_kernel_realmlp.py")
k = importlib.util.module_from_spec(spec); spec.loader.exec_module(k)
k.DATA_DIR = COMP / "data"

T0 = time.perf_counter()
def log(m): print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)

# ---- load + features (once) ----
train_raw = pd.read_csv(k.DATA_DIR / "train.csv")
test_raw  = pd.read_csv(k.DATA_DIR / "test.csv")
sample_sub = pd.read_csv(k.DATA_DIR / "sample_submission.csv")
y_all = train_raw[k.TARGET].map(k.LABEL_MAP).astype(int).values
n_train, n_test = len(train_raw), len(test_raw)
X      = k.zsoft_fe(k.stateless_fe(train_raw.drop(columns=[k.IDC, k.TARGET])))
X_test = k.zsoft_fe(k.stateless_fe(test_raw.drop(columns=[k.IDC])))
log(f"loaded train={n_train} test={n_test} feats={X.shape[1]}")

SUMMARY = {}
for N_SPLITS in (25, 30):
    t_start = time.perf_counter()
    log(f"========== {N_SPLITS}-FOLD ==========")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=k.SEED)
    folds = list(skf.split(np.arange(n_train), y_all))
    oof = np.zeros((n_train, k.N_CLASSES), dtype=np.float32)
    test_proba = np.zeros((n_test, k.N_CLASSES), dtype=np.float32)
    fold_scores = []
    for fid, (tr_idx, val_idx) in enumerate(folds):
        fseed = k.SEED + (fid + 1) * 100
        k.seed_everything(fseed)
        Xtr, Xvl, Xtt, cats, combos = k.fit_fold_categoricals(
            X.iloc[tr_idx].reset_index(drop=True), X.iloc[val_idx].reset_index(drop=True), X_test.copy())
        Xtr, Xvl, Xtt = k.add_target_encoding(Xtr, y_all[tr_idx], Xvl, Xtt, combos, fseed)
        Xtr = Xtr.reindex(sorted(Xtr.columns), axis=1); Xvl = Xvl.reindex(sorted(Xvl.columns), axis=1); Xtt = Xtt.reindex(sorted(Xtt.columns), axis=1)
        cats = sorted(cats)
        m = k.RealMLP_TD_Classifier(random_state=fseed, device=str(k.DEVICE))
        m.fit(Xtr, y_all[tr_idx], Xvl, y_all[val_idx], cat_col_names=cats, X_test=Xtt)
        oof[val_idx] = m.best_val_probs_.astype("float32")
        test_proba += m.predict_proba(Xtt).astype("float32") / N_SPLITS
        s = balanced_accuracy_score(y_all[val_idx], oof[val_idx].argmax(1))
        fold_scores.append(s)
        import gc, torch
        del m, Xtr, Xvl, Xtt; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        log(f"  fold {fid}: BA={s:.6f}")
    cv = balanced_accuracy_score(y_all, oof.argmax(1))   # full-OOF BA (single number, all rows)
    cv_mean = float(np.mean(fold_scores))
    sem = float(np.std(fold_scores, ddof=1) / np.sqrt(len(fold_scores)))
    np.save(PROBE_DIR / f"n140_{N_SPLITS}fold_test_probs.npy", test_proba)
    sub = pd.DataFrame({k.IDC: test_raw[k.IDC].values, k.TARGET: [k.CLASSES[i] for i in test_proba.argmax(1)]})
    sub[list(sample_sub.columns)].to_csv(PROBE_DIR / f"n140_{N_SPLITS}fold_submission.csv", index=False)
    elapsed = time.perf_counter() - t_start
    SUMMARY[N_SPLITS] = (cv, cv_mean, sem, elapsed)
    log(f"  {N_SPLITS}-fold: full-OOF BA={cv:.6f}  fold-mean={cv_mean:.6f}  sem={sem:.6f}  ({elapsed/60:.1f}min)")

print("\n" + "="*60, flush=True)
print("N-FOLD SWEEP SUMMARY (RealMLP n140 recipe)", flush=True)
print(f"  5-fold (frozen, reference): full-OOF BA ~0.9693", flush=True)
for N, (cv, cvm, sem, el) in SUMMARY.items():
    print(f"  {N:2d}-fold: full-OOF BA={cv:.6f}  fold-mean={cvm:.6f}±{sem:.6f}  ({el/60:.1f}min)", flush=True)
print("Done.", flush=True)
