"""B1: feature-weighted linear stacking (Sill et al.) — meta columns = base log-probs + gates + products.
Test on both our champion-15 set and the bank-17 set. Honest nested (meta fit on other folds)."""
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; L2I={l:i for i,l in enumerate(LAB)}; NC=3
tr=pd.read_csv(C/"data/train.csv"); n=len(tr); y=tr["class"].map(L2I).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
def logp(a): return np.log(np.clip(a,1e-7,1.0))
CHAMP15=["node_0006","node_0004","node_0001","node_0009","node_0011","node_0003","node_0019","node_0016","node_0014","node_0028","node_0032","node_0035","node_0033","node_0030","node_0039"]
ours=[logp(np.load(C/"nodes"/b/"oof.npy")) for b in CHAMP15]
good=json.loads(open("/tmp/pub_good.json").read()); POOF=np.load("/tmp/pub_oof.npy",allow_pickle=True).item()
pub=[logp(POOF[k]) for k in good]
# gates (standardized)
def z(v): return (v-v.mean())/v.std()
g=np.stack([z(tr['redshift'].values), z(np.log1p(np.clip(tr['redshift'].values,0,None))), z((tr['u']-tr['r']).values)],1)  # (n,3)
def balacc(yy,pred): return float(np.mean([(pred[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de_thr(P,yy):
    f=lambda w:-balacc(yy,np.argmax(P*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=40,tol=1e-7,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
def build(cols, fwls):
    L=np.concatenate(cols,1)  # (n, k)
    if not fwls: return L
    # products: each col * each gate
    prods=[L*g[:,j:j+1] for j in range(g.shape[1])]
    return np.concatenate([L,g]+prods,1)
def eval_cv(cols,fwls):
    X=build(cols,fwls); stack=np.zeros((n,NC))
    for vi in fval:
        trr=np.setdiff1d(np.arange(n),vi)
        stack[vi]=LogisticRegression(class_weight="balanced",C=1.0,max_iter=3000,n_jobs=-1).fit(X[trr],y[trr]).predict_proba(X[vi])
    pf=[]
    for vi in fval:
        oth=np.setdiff1d(np.arange(n),vi); w=de_thr(stack[oth],y[oth]); pf.append(balacc(y[vi],np.argmax(stack[vi]*w,1)))
    return float(np.mean(pf)),float(np.std(pf,ddof=1)/np.sqrt(len(pf)))
for label,cols in [("champ15",ours),("bank17",pub)]:
    cv0,s0=eval_cv(cols,False); cv1,s1=eval_cv(cols,True)
    print(f"{label}: plain={cv0:.6f}±{s0:.6f}  FWLS={cv1:.6f}±{s1:.6f}  Δ={cv1-cv0:+.6f}")
