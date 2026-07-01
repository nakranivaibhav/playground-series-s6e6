"""A1: subset search around bank-only-17, then build submission + save stacked oof/test for best."""
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; L2I={l:i for i,l in enumerate(LAB)}; NC=3
tr=pd.read_csv(C/"data/train.csv"); te=pd.read_csv(C/"data/test.csv")
n=len(tr); nt=len(te); y=tr["class"].map(L2I).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
good=json.loads(open("/tmp/pub_good.json").read())
POOF=np.load("/tmp/pub_oof.npy",allow_pickle=True).item(); PTEST=np.load("/tmp/pub_test.npy",allow_pickle=True).item()
def logp(a): return np.log(np.clip(a,1e-7,1.0))
CHAMP15=["node_0006","node_0004","node_0001","node_0009","node_0011","node_0003","node_0019","node_0016","node_0014","node_0028","node_0032","node_0035","node_0033","node_0030","node_0039"]
def oo(b): return logp(np.load(C/"nodes"/b/"oof.npy"))
def tt(b):
    f=C/"nodes"/b/"test_probs.npy"; return logp(np.load(f)) if f.exists() else None
def balacc(yy,pred): return float(np.mean([(pred[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de_thr(P,yy):
    f=lambda w:-balacc(yy,np.argmax(P*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=40,tol=1e-7,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
def eval_cv(cols):
    OOF=np.concatenate(cols,1); stack=np.zeros((n,NC))
    for vi in fval:
        trr=np.setdiff1d(np.arange(n),vi)
        stack[vi]=LogisticRegression(class_weight="balanced",C=1.0,max_iter=2000,n_jobs=-1).fit(OOF[trr],y[trr]).predict_proba(OOF[vi])
    pf=[]
    for vi in fval:
        oth=np.setdiff1d(np.arange(n),vi); w=de_thr(stack[oth],y[oth]); pf.append(balacc(y[vi],np.argmax(stack[vi]*w,1)))
    return float(np.mean(pf)),float(np.std(pf,ddof=1)/np.sqrt(len(pf))),stack

pub_oof=[logp(POOF[k]) for k in good]; pub_test=[logp(PTEST[k]) for k in good]
# our unique strong bases that might add (have test_probs)
UNIQUE=["node_0028","node_0032","node_0035","node_0033"]  # 3 RealMLP-ref seeds + TabM-richFE
uoof={b:oo(b) for b in UNIQUE}; utest={b:tt(b) for b in UNIQUE}
print("our-unique test_probs present:",{b:(utest[b] is not None) for b in UNIQUE})

variants={"bank17":(pub_oof,pub_test)}
# + all 4 unique
av=[b for b in UNIQUE if utest[b] is not None]
variants["bank17+4uniq"]=(pub_oof+[uoof[b] for b in av], pub_test+[utest[b] for b in av])
# + just TabM (most de-correlated)
if utest["node_0033"] is not None:
    variants["bank17+tabm"]=(pub_oof+[uoof["node_0033"]],pub_test+[utest["node_0033"]])

best=None
for name,(co,ct) in variants.items():
    cv,sem,_=eval_cv(co); print(f"{name:16s} cv={cv:.6f} sem={sem:.6f}")
    if best is None or cv>best[1]: best=(name,cv,sem,co,ct)

name,cv,sem,co,ct=best
print(f"\nBEST: {name} cv={cv:.6f}")
# final fit: meta on ALL oof rows, DE on ALL oof, predict test
OOF=np.concatenate(co,1); TST=np.concatenate(ct,1)
meta=LogisticRegression(class_weight="balanced",C=1.0,max_iter=3000,n_jobs=-1).fit(OOF,y)
stack_oof=meta.predict_proba(OOF); stack_test=meta.predict_proba(TST)
w=de_thr(stack_oof,y)
print("DE weights:",w)
pred=np.argmax(stack_test*w,1)
sub=pd.DataFrame({"id":te["id"],"class":[LAB[i] for i in pred]})
outdir=C/"refs/a1_bank_stack"; outdir.mkdir(exist_ok=True)
sub.to_csv(outdir/"submission_bank17.csv",index=False)
np.save(outdir/"stack_oof.npy",stack_oof); np.save(outdir/"stack_test.npy",stack_test)
print("class dist:",sub['class'].value_counts().to_dict())
print("wrote",outdir/"submission_bank17.csv")
