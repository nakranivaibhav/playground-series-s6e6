import json,warnings,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; NC=3
tr=pd.read_csv(C/"data/train.csv"); n=len(tr); y=tr["class"].map({l:i for i,l in enumerate(LAB)}).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
def lp(a): return np.log(np.clip(a,1e-7,1.0))
CH=["node_0006","node_0004","node_0001","node_0009","node_0011","node_0003","node_0019","node_0016","node_0014","node_0028","node_0032","node_0035","node_0033","node_0030","node_0039"]
champ15=[lp(np.load(C/"nodes"/b/"oof.npy")) for b in CH]
good=json.loads(open("/tmp/pub_good.json").read()); POOF=np.load("/tmp/pub_oof.npy",allow_pickle=True).item()
bank17=[lp(POOF[k]) for k in good]
NID=sys.argv[1]; new=lp(np.load(C/"nodes"/NID/"oof.npy"))
# n30 is index 13 in champ15 (node_0030)
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
sw=champ15.copy(); sw[13]=new  # swap node_0030 -> new
solo=balacc(y,np.load(C/"nodes"/NID/"oof.npy").argmax(1))
print(f"{NID} solo argmax BA={solo:.6f}")
for nm,c in {"champ15 base":champ15,"champ15 swap n30->new":sw,"champ15 +new(16th)":champ15+[new],
             "bank17 base":bank17,"bank17 +new":bank17+[new]}.items():
    cv,s=ev(c); print(f"{nm:24s} cv={cv:.6f} sem={s:.6f}")
