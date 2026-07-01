"""A2: bagged Caruana greedy ensemble selection (with replacement) directly on balanced accuracy,
over our base OOFs + the 17 public bank models. Honest: select on 4 folds, score held fold, rotate.
DE per-class threshold applied per held fold (fit on the selection folds' ensemble probs)."""
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; L2I={l:i for i,l in enumerate(LAB)}; NC=3
tr=pd.read_csv(C/"data/train.csv"); n=len(tr); y=tr["class"].map(L2I).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
# our base pool (exclude ensemble/stack nodes)
EXCL={"node_0007","node_0010","node_0017","node_0020","node_0029","node_0040","node_0041",
      "node_0052","node_0053","node_0002","node_0047"}  # ensembles/threshold-only/mirage
import glob,os
ours={os.path.basename(os.path.dirname(f)):np.load(f) for f in glob.glob(str(C/"nodes/*/oof.npy"))}
ours={k:v for k,v in ours.items() if k not in EXCL and v.shape==(n,3)}
POOF=np.load("/tmp/pub_oof.npy",allow_pickle=True).item()
pool={**{f"ours:{k}":v for k,v in ours.items()},**{f"pub:{k}":v for k,v in POOF.items()}}
names=list(pool); P=np.stack([pool[k] for k in names])  # (M,n,3)
M=len(names); print(f"pool size {M}")
def balacc(yy,pred): return float(np.mean([(pred[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de_w(prob,yy):
    f=lambda w:-balacc(yy,np.argmax(prob*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=30,tol=1e-7,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
def caruana(idx_sel, rng, init=5, steps=60):
    # idx_sel: row indices to select on; returns weight vector over M (counts)
    Psel=P[:,idx_sel,:]; ysel=y[idx_sel]
    solo=np.array([balacc(ysel,Psel[m].argmax(1)) for m in range(M)])
    order=np.argsort(-solo); counts=np.zeros(M)
    ens=np.zeros((len(idx_sel),3))
    for m in order[:init]: ens+=Psel[m]; counts[m]+=1
    bag=order[:init].tolist()
    for _ in range(steps):
        best,bm=-1,-1
        for m in range(M):
            cand=ens+Psel[m]
            b=balacc(ysel,cand.argmax(1))
            if b>best: best,bm=b,m
        ens+=Psel[bm]; counts[bm]+=1
    return counts/counts.sum()
# honest CV: select on other folds, score held fold (bagged over 5 subsamples)
rng=np.random.RandomState(0); pf=[]
ensemble_full=np.zeros((n,3))
for k,vi in enumerate(fval):
    oth=np.setdiff1d(np.arange(n),vi)
    w=np.zeros(M)
    for b in range(5):
        sub=rng.choice(oth,size=len(oth)//2,replace=False)
        w+=caruana(sub,rng)
    w/=5
    ens_oth=np.tensordot(w,P[:,oth,:],axes=(0,0)); ens_vi=np.tensordot(w,P[:,vi,:],axes=(0,0))
    th=de_w(ens_oth,y[oth])
    pf.append(balacc(y[vi],np.argmax(ens_vi*th,1)))
    ensemble_full[vi]=ens_vi
    print(f"fold{k} held-BA={pf[-1]:.6f}")
print(f"\nA2 Caruana honest CV = {np.mean(pf):.6f} ± {np.std(pf,ddof=1)/np.sqrt(5):.6f}")
print("champion 0.969808 | bank-only-17 0.970153")
# top selected members on full data
wf=np.zeros(M)
for b in range(5):
    sub=rng.choice(np.arange(n),size=n//2,replace=False); wf+=caruana(sub,rng)
wf/=5
top=np.argsort(-wf)[:12]
print("top members:",[(names[i],round(float(wf[i]),3)) for i in top if wf[i]>0])
