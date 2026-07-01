"""A1: merge public 18-model OOF bank (the 6 we have) into our 15-base champion stack.
Fold scheme verified IDENTICAL (SKF(5,shuffle,42), 100% row match), so public OOF aligns row-wise.
Same champion meta recipe (balanced multinomial LogReg on clipped log-probs + honest DE threshold)."""
import json, warnings, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
COMP=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; L2I={l:i for i,l in enumerate(LAB)}; NC=3
CHAMP9=["node_0006","node_0004","node_0001","node_0009","node_0011","node_0003","node_0019","node_0016","node_0014"]
CHAMP15=CHAMP9+["node_0028","node_0032","node_0035","node_0033","node_0030","node_0039"]
train=pd.read_csv(COMP/"data/train.csv"); folds=json.loads((COMP/"folds.json").read_text())["folds"]
n=len(train); y=train["class"].map(L2I).to_numpy(); fval=[np.asarray(f["val_idx"]) for f in folds]
def logp(a): return np.log(np.clip(a,1e-7,1.0))
def load_ours(b): return logp(np.load(COMP/"nodes"/b/"oof.npy"))

# load the 6 public OOFs (robust to format), aligned to train order
BANK=COMP/"refs/oof_bank"; ytr_id=train['id'].values
def load_pub(f):
    d=pd.read_csv(f); c=list(d.columns)
    if set(LAB).issubset(c): return d[LAB].values
    v=d.iloc[:,0].values; return v.reshape(n,3)
PUB={os.path.basename(f).replace("oof_preds_","").replace("oof_final_","").replace(".csv",""):logp(load_pub(f))
     for f in sorted(glob.glob(str(BANK/"oof_*.csv")))}
def balacc(yy,pred): return float(np.mean([(pred[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de_thr(P,yy):
    f=lambda w:-balacc(yy,np.argmax(P*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=40,tol=1e-7,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
def eval_cols(cols):
    OOF=np.concatenate(cols,axis=1); stack=np.zeros((n,NC))
    for vi in fval:
        tr=np.setdiff1d(np.arange(n),vi)
        m=LogisticRegression(class_weight="balanced",C=1.0,max_iter=2000,n_jobs=-1).fit(OOF[tr],y[tr])
        stack[vi]=m.predict_proba(OOF[vi])
    pf=[]
    for vi in fval:
        oth=np.setdiff1d(np.arange(n),vi); w=de_thr(stack[oth],y[oth])
        pf.append(balacc(y[vi],np.argmax(stack[vi]*w,1)))
    return float(np.mean(pf)), float(np.std(pf,ddof=1)/np.sqrt(len(pf)))

ours=[load_ours(b) for b in CHAMP15]
pub_all=[PUB[k] for k in PUB]
print(f"public bases: {list(PUB.keys())}")
VAR={
 "champion 15 (ours)": ours,
 "15 + all6 public":   ours+pub_all,
 "bank-only 6 public": pub_all,
}
# also each public base individually added
for k in PUB: VAR[f"15 + {k}"]=ours+[PUB[k]]
print(f"\n{'set':26s} {'cv':>11s} {'sem':>9s} {'Δ vs champ':>11s}")
base=None
for name,cols in VAR.items():
    cv,sem=eval_cols(cols)
    if base is None and name.startswith("champion"): base=cv
    d=cv-(base if base else cv)
    print(f"{name:26s} {cv:11.6f} {sem:9.6f} {d:+11.6f}")
