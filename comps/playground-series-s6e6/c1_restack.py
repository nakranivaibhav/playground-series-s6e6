"""C1 restack: does the cleanlab-pruned RealMLP base (node_0059) lift the stack?
Test: champ15 baseline; champ15 swap n28->cleaned; champ15 + cleaned(16th); bank17; bank17+cleaned."""
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; NC=3
tr=pd.read_csv(C/"data/train.csv"); n=len(tr); y=tr["class"].map({l:i for i,l in enumerate(LAB)}).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
def logp(a): a=np.clip(a,1e-7,1.0); return np.log(a)
CH=["node_0006","node_0004","node_0001","node_0009","node_0011","node_0003","node_0019","node_0016","node_0014"]
RMLP=["node_0028","node_0032","node_0035"]; REST=["node_0033","node_0030","node_0039"]
def oo(b): return logp(np.load(C/"nodes"/b/"oof.npy"))
champ15=[oo(b) for b in CH+RMLP+REST]
clean_all=logp(np.load(C/"nodes/node_0059/oof_all.npy"))
good=json.loads(open("/tmp/pub_good.json").read()); POOF=np.load("/tmp/pub_oof.npy",allow_pickle=True).item()
bank17=[logp(POOF[k]) for k in good]
def balacc(yy,pred): return float(np.mean([(pred[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de(P,yy):
    f=lambda w:-balacc(yy,np.argmax(P*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=40,tol=1e-7,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
def ev(cols):
    X=np.concatenate(cols,1); st=np.zeros((n,NC))
    for vi in fval:
        trr=np.setdiff1d(np.arange(n),vi)
        st[vi]=LogisticRegression(class_weight="balanced",C=1.0,max_iter=2000,n_jobs=-1).fit(X[trr],y[trr]).predict_proba(X[vi])
    pf=[]
    for vi in fval:
        oth=np.setdiff1d(np.arange(n),vi); w=de(st[oth],y[oth]); pf.append(balacc(y[vi],np.argmax(st[vi]*w,1)))
    return float(np.mean(pf)),float(np.std(pf,ddof=1)/np.sqrt(len(pf)))
# champ15 swap n28(index 9)->cleaned
champ15_swap=champ15.copy(); champ15_swap[9]=clean_all
V={"champ15 (base)":champ15,
   "champ15 swap n28->clean":champ15_swap,
   "champ15 + clean (16th)":champ15+[clean_all],
   "bank17 (base)":bank17,
   "bank17 + clean":bank17+[clean_all]}
for nm,c in V.items():
    cv,s=ev(c); print(f"{nm:28s} cv={cv:.6f} sem={s:.6f}")
