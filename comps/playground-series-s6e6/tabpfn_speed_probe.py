"""Confirm the batching fix: encode the 50k context ONCE vs re-encoding per 4k chunk.

Times a single subsample (fit on 50k context, predict a block of val rows) under two
regimes: BIG single predict call (context encoded once) vs the old 4k-chunk loop
(context re-encoded ~N/4000 times). On a 5090 the big call should be seconds.
"""
import time, numpy as np, pandas as pd, json, torch
from pathlib import Path

COMP = Path(__file__).resolve().parent
CKPT = Path.home()/".cache"/"tabpfn"/"tabpfn-v2-classifier.ckpt"
LAB = ["GALAXY","QSO","STAR"]; L2I={l:i for i,l in enumerate(LAB)}

train = pd.read_csv(COMP/"data/train.csv")
folds = json.loads((COMP/"folds.json").read_text())["folds"]
y = train["class"].map(L2I).to_numpy()
num = train.select_dtypes("number").fillna(0).to_numpy(np.float32)
X = num  # raw numerics is enough to MEASURE SPEED (not accuracy)

vi = np.asarray(folds[0]["val_idx"]); tr = np.setdiff1d(np.arange(len(train)), vi)
rng = np.random.default_rng(0)
ctx = rng.choice(tr, size=50_000, replace=False)
QN = 20_000                      # measure on 20k query rows
q = X[vi[:QN]]

from tabpfn import TabPFNClassifier
def mk():
    return TabPFNClassifier(device="cuda", n_estimators=1, model_path=str(CKPT),
                            ignore_pretraining_limits=True, show_progress_bar=False)

clf = mk(); clf.fit(X[ctx], y[ctx])
torch.cuda.synchronize()

# warmup small
_ = clf.predict_proba(q[:512]); torch.cuda.synchronize()

t0=time.time(); _ = clf.predict_proba(q); torch.cuda.synchronize(); t_big=time.time()-t0
print(f"BIG single call : {QN} query rows in {t_big:6.2f}s  ({1000*t_big/QN:.2f} ms/row)")

t0=time.time()
for s in range(0, QN, 4000):
    _ = clf.predict_proba(q[s:s+4000])
torch.cuda.synchronize(); t_chunk=time.time()-t0
print(f"4k-chunk loop   : {QN} query rows in {t_chunk:6.2f}s  ({1000*t_chunk/QN:.2f} ms/row)")
print(f"\nspeedup from single-call batching: {t_chunk/max(t_big,1e-6):.1f}x")
print(f"extrapolated per-subsample (115k val) BIG  ≈ {t_big*115470/QN:6.1f}s")
print(f"extrapolated per-subsample (115k val) 4k   ≈ {t_chunk*115470/QN:6.1f}s")
