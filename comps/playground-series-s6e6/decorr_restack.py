"""Test whether the DE-CORRELATED weak models (DAE, binchain, OvR) lift the bank-17 stack,
even though weak solo. LogReg stack (meta downweights) + a fold-honest prob-blend variant."""
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; NC=3
tr=pd.read_csv(C/"data/train.csv"); n=len(tr); y=tr["class"].map({l:i for i,l in enumerate(LAB)}).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
def lp(a): return np.log(np.clip(a,1e-7,1.0))
good=json.loads(open("/tmp/pub_good.json").read()); POOF=np.load("/tmp/pub_oof.npy",allow_pickle=True).item()
bank=[lp(POOF[k]) for k in good]
def oo(p): return lp(np.load(C/"nodes"/p))
DAE=oo("node_0062/oof.npy"); BIN=oo("node_0049/oof.npy"); OVR=oo("node_0050/oof.npy")
CLEAN=oo("node_0059/oof_all.npy")
def balacc(yy,p): return float(np.mean([(p[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de(P,yy):
    f=lambda w:-balacc(yy,np.argmax(P*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=40,tol=1e-7,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
def ev(cols):
    X=np.concatenate(cols,1); st=np.zeros((n,NC))
    for vi in fval:
        trr=np.setdiff1d(np.arange(n),vi)
        st[vi]=LogisticRegression(class_weight="balanced",C=1.0,max_iter=2000,n_jobs=-1).fit(X[trr],y[trr]).predict_proba(X[vi])
    pf=[balacc(y[vi],np.argmax(st[vi]*de(st[np.setdiff1d(np.arange(n),vi)],y[np.setdiff1d(np.arange(n),vi)]),1)) for vi in fval]
    return float(np.mean(pf)),float(np.std(pf,ddof=1)/np.sqrt(5))
V={"bank17 (base)":bank,
   "bank17 +DAE":bank+[DAE],
   "bank17 +binchain":bank+[BIN],
   "bank17 +OvR":bank+[OVR],
   "bank17 +DAE+bin+OvR":bank+[DAE,BIN,OVR],
   "bank17 +DAE+cleanlab":bank+[DAE,CLEAN]}
base=None
for nm,c in V.items():
    cv,s=ev(c)
    if base is None: base=cv
    print(f"{nm:24s} cv={cv:.6f} sem={s:.6f}  Δ={cv-base:+.6f}")
