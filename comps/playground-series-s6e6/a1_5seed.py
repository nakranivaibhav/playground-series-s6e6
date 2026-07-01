"""Test Deotte's 5-seed x 5-fold re-partition meta on bank-17 vs our single-partition (0.970153)."""
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; NC=3
tr=pd.read_csv(C/"data/train.csv"); n=len(tr); y=tr["class"].map({l:i for i,l in enumerate(LAB)}).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
good=json.loads(open("/tmp/pub_good.json").read()); POOF=np.load("/tmp/pub_oof.npy",allow_pickle=True).item()
def lp(a): return np.log(np.clip(a,1e-7,1.0))
X=np.concatenate([lp(POOF[k]) for k in good],1)
def balacc(yy,p): return float(np.mean([(p[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de(P,yy):
    f=lambda w:-balacc(yy,np.argmax(P*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=40,tol=1e-7,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
def stack_oof(seeds):
    acc=np.zeros((n,NC))
    for sd in seeds:
        skf=StratifiedKFold(5,shuffle=True,random_state=sd)
        st=np.zeros((n,NC))
        for trr,vi in skf.split(X,y):
            st[vi]=LogisticRegression(class_weight="balanced",C=1.0,max_iter=2000,n_jobs=-1).fit(X[trr],y[trr]).predict_proba(X[vi])
        acc+=st
    return acc/len(seeds)
# honest CV on our frozen folds (single seed) — reference
def cv_frozen():
    st=np.zeros((n,NC))
    for vi in fval:
        trr=np.setdiff1d(np.arange(n),vi)
        st[vi]=LogisticRegression(class_weight="balanced",C=1.0,max_iter=2000,n_jobs=-1).fit(X[trr],y[trr]).predict_proba(X[vi])
    pf=[balacc(y[vi],np.argmax(st[vi]*de(st[np.setdiff1d(np.arange(n),vi)],y[np.setdiff1d(np.arange(n),vi)]),1)) for vi in fval]
    return np.mean(pf),np.std(pf,ddof=1)/np.sqrt(5)
cv0,s0=cv_frozen(); print(f"frozen single-partition CV = {cv0:.6f} ± {s0:.6f}")
# 5-seed averaged OOF, scored on frozen folds with honest DE
for nseed in [1,5]:
    so=stack_oof(list(range(42,42+nseed)))
    pf=[balacc(y[vi],np.argmax(so[vi]*de(so[np.setdiff1d(np.arange(n),vi)],y[np.setdiff1d(np.arange(n),vi)]),1)) for vi in fval]
    print(f"{nseed}-seed re-partition meta, scored on frozen folds: {np.mean(pf):.6f} ± {np.std(pf,ddof=1)/np.sqrt(5):.6f}")
