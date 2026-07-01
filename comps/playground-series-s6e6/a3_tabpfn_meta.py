"""A3: TabPFN as a NON-LINEAR meta on bank-17 logits + raw features. Compare vs LogReg meta (0.970153).
fold-0 timing probe first (TabPFN on ~460k rows can be slow)."""
import json,warnings,time,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; NC=3
tr=pd.read_csv(C/"data/train.csv"); te=pd.read_csv(C/"data/test.csv")
n=len(tr); y=tr["class"].map({l:i for i,l in enumerate(LAB)}).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
good=json.loads(open("/tmp/pub_good.json").read()); POOF=np.load("/tmp/pub_oof.npy",allow_pickle=True).item()
def lp(a): return np.log(np.clip(a,1e-7,1.0))
Xlog=np.concatenate([lp(POOF[k]) for k in good],1)  # (n,51)
def raws(df):
    z=df['redshift'].values
    return np.stack([z,np.log1p(np.clip(z,0,None)),(df['u']-df['r']).values,(df['r']-df['g']).values,
                     (df['g']-df['r']).values,df[['u','g','r','i','z']].mean(1).values,df['alpha'].values,df['delta'].values],1)
Xraw=raws(tr)
X=np.concatenate([Xlog,Xraw],1).astype(np.float32)  # (n,59)
def balacc(yy,p): return float(np.mean([(p[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de(P,yy):
    f=lambda w:-balacc(yy,np.argmax(P*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=40,tol=1e-7,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
from tabpfn import TabPFNClassifier
_CK=__import__('os').path.expanduser('~/.cache/tabpfn/tabpfn-v3-classifier-v3_20260417_multiclass.ckpt')
def fit_pred(Xtr,ytr,Xpred):
    rng=np.random.RandomState(0)
    # stratified subsample context to ~30k (TabPFN-v2 cannot fit 460k)
    idx=[]
    for c in range(NC):
        ci=np.where(ytr==c)[0]; k=min(len(ci),10000); idx+=list(rng.choice(ci,k,replace=False))
    idx=np.array(idx)
    m=TabPFNClassifier(n_estimators=2,device='cuda',ignore_pretraining_limits=True,model_path=_CK)
    m.fit(Xtr[idx],ytr[idx])
    # predict in chunks to bound memory
    out=np.zeros((len(Xpred),NC))
    for s0 in range(0,len(Xpred),20000):
        out[s0:s0+20000]=m.predict_proba(Xpred[s0:s0+20000])
    return out
# fold-0 timing probe
vi=fval[0]; trr=np.setdiff1d(np.arange(n),vi)
t0=time.time(); p0=fit_pred(X[trr],y[trr],X[vi]); dt=time.time()-t0
ba0=balacc(y[vi],p0.argmax(1)); print(f"fold-0 TabPFN meta: BA={ba0:.6f} time={dt:.0f}s",flush=True)
print(f"projected 5-fold ~{5*dt/60:.0f}min",flush=True)
if dt>1800: print("TOO SLOW (>30min/fold) — abort full run"); sys.exit(0)
# full nested OOF
stack=np.zeros((n,NC)); stack[vi]=p0
for vi in fval[1:]:
    trr=np.setdiff1d(np.arange(n),vi); stack[vi]=fit_pred(X[trr],y[trr],X[vi]); print(f"fold done",flush=True)
pf=[balacc(y[v],np.argmax(stack[v]*de(stack[np.setdiff1d(np.arange(n),v)],y[np.setdiff1d(np.arange(n),v)]),1)) for v in fval]
print(f"A3 TabPFN-meta CV = {np.mean(pf):.6f} ± {np.std(pf,ddof=1)/np.sqrt(5):.6f}  (LogReg meta bank17=0.970153)")
np.save("/tmp/a3_stack_oof.npy",stack)
