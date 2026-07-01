"""A1 full: 18-model public bank (Deotte) ingest+validate, reproduce bank-only stack,
then merge with our 15 CORE bases. Fold scheme verified identical (SKF5,shuffle,42)."""
import json,warnings,os
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; L2I={l:i for i,l in enumerate(LAB)}; NC=3
tr=pd.read_csv(C/"data/train.csv"); te=pd.read_csv(C/"data/test.csv")
n=len(tr); nt=len(te); y=tr["class"].map(L2I).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
B=C/"refs/oof_bank"; K=C/"refs/kernel_out"
def rd(path,nr):
    p=str(path)
    if p.endswith(".npy"):
        a=np.load(p,allow_pickle=True).astype(float)
        a=a.reshape(nr,-1) if a.ndim==1 else a
        return a[:,:3]
    d=pd.read_csv(p); c=list(d.columns)
    if set(LAB).issubset(c): return d[LAB].values.astype(float)
    pc=[f"prob_{l}" for l in LAB]
    if set(pc).issubset(c): return d[pc].values.astype(float)
    # numeric 3-col without proper header, or flattened single col
    num=d.select_dtypes('number')
    if num.shape[1]>=3: return num.values[:,:3]
    v=d.iloc[:,0].values.astype(float); return v.reshape(nr,3)
def norm(a):
    a=np.clip(a,0,None); s=a.sum(1,keepdims=True); s[s==0]=1; return a/s
# manifest: name -> (oof_path, test_path)
M={
 'xgb-0':(K/"xgb-v0-for-s6e6/oof_xgb_cv.csv",K/"xgb-v0-for-s6e6/test_xgb_preds.csv"),
 'xgb-1':(K/"xgb-v1-for-s6e6/oof_preds.npy",K/"xgb-v1-for-s6e6/test_preds.npy"),
 'realmlp-0':(B/"oof_preds_realmlp0_v12.csv",B/"test_preds_realmlp0_v12.csv"),
 'realmlp-1':(K/"realmlp-v1-for-s6e6/oof_preds.npy",K/"realmlp-v1-for-s6e6/test_preds.npy"),
 'tabm-0':(B/"oof_preds_tabm0_v2.csv",B/"test_preds_tabm0_v2.csv"),
 'cat-0':(K/"cat-v0-for-s6e6/catboost_oof_predictions.csv",K/"cat-v0-for-s6e6/catboost_test_predictions.csv"),
 'realmlp-2':(B/"oof_preds_realmlp2_v10.csv",B/"test_preds_realmlp2_v10.csv"),
 'tabicl-2':(K/"tabicl-v2-for-s6e6/train_oof/tabicl-2_oof.npy",K/"tabicl-v2-for-s6e6/test_preds/tabicl-2_test_preds.npy"),
 'lgbm-3':(K/"lgbm-v3-for-s6e6/train_oof/lgbm-3_oof.npy",K/"lgbm-v3-for-s6e6/test_preds/lgbm-3_test_preds.npy"),
 'logreg-1':(K/"logreg-v1-for-s6e6/train_oof/logreg-1_oof.npy",K/"logreg-v1-for-s6e6/test_preds/logreg-1_test_preds.npy"),
 'nn-1':(K/"nn-v1-for-s6e6/train_oof/nn-1_oof.npy",K/"nn-v1-for-s6e6/test_preds/nn-1_test_preds.npy"),
 'xgb-3':(K/"xgb-v3-for-s6e6/stellar_class_xgb_oof_preds_raw.npy",K/"xgb-v3-for-s6e6/stellar_class_xgb_test_preds_raw.npy"),
 'xgb-5':(K/"xgb-v5-for-s6e6/train_oof/xgb-5_oof.npy",K/"xgb-v5-for-s6e6/test_preds/xgb-5_test_preds.npy"),
 'realmlp-5':(K/"realmlp-v5-for-s6e6/train_oof/realmlp-5_oof.npy",K/"realmlp-v5-for-s6e6/test_preds/realmlp-5_test_preds.npy"),
 'nn-2':(K/"nn-v2-for-s6e6/train_oof/nn-2_oof.npy",K/"nn-v2-for-s6e6/test_preds/nn-2_test_preds.npy"),
 'cat-3':(K/"cat-v3-for-s6e6/train_oof/cat-3_oof.npy",K/"cat-v3-for-s6e6/test_preds/cat-3_test_preds.npy"),
 'lgbm-5':(B/"oof_preds_lgbm5_v1.csv",B/"test_preds_lgbm5_v1.csv"),
 'xgb-6':(B/"oof_final_xgb6_v1.csv",B/"test_final_xgb6_v1.csv"),
 'tabm-1':(B/"oof_final_tabm1_v1.csv",B/"test_final_tabm1_v1.csv"),
}
POOF={}; PTEST={}; good=[]
print(f"{'model':12s} {'oofBA':>9s} {'shape':>12s} {'status'}")
for name,(op,tp) in M.items():
    try:
        o=norm(rd(op,n)); t=norm(rd(tp,nt))
        assert o.shape==(n,3) and t.shape==(nt,3)
        ba=balanced_accuracy_score(y,o.argmax(1))
        st="OK" if 0.90<ba<0.972 else ("QUARANTINE" if ba>=0.972 else "LOW?")
        if st=="OK": POOF[name]=o; PTEST[name]=t; good.append(name)
        print(f"{name:12s} {ba:9.6f} {str(o.shape):>12s} {st}")
    except Exception as e:
        print(f"{name:12s} {'--':>9s} {'--':>12s} FAIL {str(e)[:50]}")
print(f"\nloaded {len(good)}/19 public models OK")
np.save("/tmp/pub_oof.npy",{k:POOF[k] for k in good},allow_pickle=True)
np.save("/tmp/pub_test.npy",{k:PTEST[k] for k in good},allow_pickle=True)
open("/tmp/pub_good.json","w").write(json.dumps(good))

# ===== MERGE EXPERIMENT =====
import sys
CHAMP9=["node_0006","node_0004","node_0001","node_0009","node_0011","node_0003","node_0019","node_0016","node_0014"]
CHAMP15=CHAMP9+["node_0028","node_0032","node_0035","node_0033","node_0030","node_0039"]
def logp(a): return np.log(np.clip(a,1e-7,1.0))
def load_ours_oof(b): return logp(np.load(C/"nodes"/b/"oof.npy"))
def load_ours_test(b):
    f=C/"nodes"/b/"test_probs.npy"
    return logp(np.load(f)) if f.exists() else None
def balacc(yy,pred): return float(np.mean([(pred[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de_thr(P,yy):
    f=lambda w:-balacc(yy,np.argmax(P*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=40,tol=1e-7,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
def eval_cols(cols):
    OOF=np.concatenate(cols,1); stack=np.zeros((n,NC))
    for vi in fval:
        trr=np.setdiff1d(np.arange(n),vi)
        stack[vi]=LogisticRegression(class_weight="balanced",C=1.0,max_iter=2000,n_jobs=-1).fit(OOF[trr],y[trr]).predict_proba(OOF[vi])
    pf=[]
    for vi in fval:
        oth=np.setdiff1d(np.arange(n),vi); w=de_thr(stack[oth],y[oth]); pf.append(balacc(y[vi],np.argmax(stack[vi]*w,1)))
    return float(np.mean(pf)),float(np.std(pf,ddof=1)/np.sqrt(len(pf)))
pub_oof=[logp(POOF[k]) for k in good]
ours_oof=[load_ours_oof(b) for b in CHAMP15]
print("\n=== MERGE (CV) ===")
for name,cols in {"champion 15 (ours)":ours_oof,
                  f"bank-only {len(good)} public":pub_oof,
                  f"15 + {len(good)} public (ALL)":ours_oof+pub_oof}.items():
    cv,sem=eval_cols(cols); print(f"{name:30s} cv={cv:.6f} sem={sem:.6f}")
