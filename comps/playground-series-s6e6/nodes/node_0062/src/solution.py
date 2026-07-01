"""node_0062 — C9: swap-noise Denoising AutoEncoder (DAE) representation -> MLP head.
The one untried REPRESENTATION family (Jahrer/ryancheunggit lineage), built FOLD-HONESTLY:
per train fold, fit a swap-noise DAE on the train-fold rows ONLY (no labels, no test rows),
freeze it, extract hidden activations for train/val/test, train a balanced MLP head -> OOF+test.
Explicitly NOT the transductive train+test fit (user constraint: no test fitting).

Outputs: oof.npy (577347,3), test_probs.npy (247435,3).
"""
from __future__ import annotations
import json, time, math
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score

T0=time.perf_counter()
def log(m): print(f"[{time.perf_counter()-T0:7.1f}s] {m}", flush=True)
NODE_DIR=Path(__file__).resolve().parent.parent; COMP=NODE_DIR.parent.parent
DEV="cuda" if torch.cuda.is_available() else "cpu"
SEED=42; N_CLASSES=3; CLASSES=["GALAXY","QSO","STAR"]
BASE_CAT=["spectral_type","galaxy_population"]; BASE_NUM=["alpha","delta","u","g","r","i","z","redshift"]
COLOR_PAIRS=[("u","g"),("g","r"),("r","i"),("i","z"),("u","r"),("g","i"),("r","z")]

def stateless_fe(df):
    df=df.copy()
    df["_g_div_rs"]=(df["g"]/(df["redshift"]+1e-6)).replace([np.inf,-np.inf],np.nan).fillna(0)
    df["_i_div_rs"]=(df["i"]/(df["redshift"]+1e-6)).replace([np.inf,-np.inf],np.nan).fillna(0)
    for a,b in COLOR_PAIRS: df[f"_{a}-{b}"]=df[a]-df[b]
    mags=df[["u","g","r","i","z"]]
    df["_mag_mean"]=mags.mean(1); df["_mag_range"]=mags.max(1)-mags.min(1)
    sh=df["redshift"]-min(0.0,float(df["redshift"].min()))+1e-4
    df["_log1p_rs"]=np.log1p(sh)
    return df

def seed_all(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class DAE(nn.Module):
    def __init__(self,d,h=(512,256,128)):
        super().__init__()
        enc=[]; p=d
        for w in h: enc+= [nn.Linear(p,w),nn.BatchNorm1d(w),nn.PReLU()]; p=w
        self.enc=nn.Sequential(*enc); self.rep_dim=sum(h)  # concat all hidden activations
        dec=[]; ph=list(h)[::-1]
        q=h[-1]
        for w in ph[1:]+[d]: dec+=[nn.Linear(q,w),nn.PReLU()]; q=w
        self.dec=nn.Sequential(*dec[:-1])  # last PReLU dropped, linear output
    def encode_reps(self,x):
        reps=[]; h=x
        for layer in self.enc:
            h=layer(h)
            if isinstance(layer,nn.PReLU): reps.append(h)
        return torch.cat(reps,1)
    def forward(self,x):
        h=x
        for layer in self.enc: h=layer(h)
        return self.dec(h)

class Head(nn.Module):
    def __init__(self,d,h=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,h),nn.BatchNorm1d(h),nn.PReLU(),nn.Dropout(0.2),
                               nn.Linear(h,h),nn.BatchNorm1d(h),nn.PReLU(),nn.Dropout(0.2),
                               nn.Linear(h,N_CLASSES))
    def forward(self,x): return self.net(x)

def rankgauss_fit(Xtr):
    # per-column rank->gaussian mapping fit on train fold; returns transform fn
    sorted_cols=[np.sort(Xtr[:,j]) for j in range(Xtr.shape[1])]
    def tf(X):
        out=np.zeros_like(X,dtype=np.float32)
        for j in range(X.shape[1]):
            r=np.searchsorted(sorted_cols[j],X[:,j],side="right")/(len(sorted_cols[j])+1)
            r=np.clip(r,1e-6,1-1e-6)
            out[:,j]=np.sqrt(2)*torch.erfinv(torch.tensor(2*r-1)).numpy()
        return out
    return tf

def main():
    train=pd.read_csv(COMP/"data/train.csv"); test=pd.read_csv(COMP/"data/test.csv")
    folds=json.loads((COMP/"folds.json").read_text())["folds"]
    y=train["class"].map({c:i for i,c in enumerate(CLASSES)}).to_numpy()
    n=len(train); nt=len(test)
    Xtr_df=stateless_fe(train.drop(columns=["id","class"]))
    Xte_df=stateless_fe(test.drop(columns=["id"]))
    # one-hot the 2 base cats, keep all numerics
    allcols=[c for c in Xtr_df.columns if c not in BASE_CAT]
    cat=pd.get_dummies(pd.concat([Xtr_df[BASE_CAT],Xte_df[BASE_CAT]],axis=0).astype(str))
    Xnum=np.vstack([Xtr_df[allcols].values,Xte_df[allcols].values]).astype(np.float32)
    Xall=np.hstack([Xnum,cat.values.astype(np.float32)])
    Xtr_full=Xall[:n]; Xte_full=Xall[n:]
    log(f"feature matrix {Xall.shape}")
    oof=np.zeros((n,N_CLASSES),dtype=np.float32); test_acc=np.zeros((nt,N_CLASSES),dtype=np.float32)
    scores=[]
    for fi in folds:
        fid=fi["fold"]; vi=np.asarray(fi["val_idx"]); ti=np.setdiff1d(np.arange(n),vi)
        seed_all(SEED+fid)
        tf=rankgauss_fit(Xtr_full[ti])
        Ztr=tf(Xtr_full[ti]); Zval=tf(Xtr_full[vi]); Zte=tf(Xte_full)
        d=Ztr.shape[1]
        dae=DAE(d).to(DEV)
        opt=torch.optim.AdamW(dae.parameters(),lr=1e-3,weight_decay=1e-5)
        Xt=torch.tensor(Ztr,device=DEV); B=2048; swap=0.20
        col_pool=Xt  # marginal pool = train-fold rows
        log(f"fold{fid}: DAE train d={d} n={len(ti)}")
        for ep in range(35):
            dae.train(); perm=torch.randperm(len(Xt),device=DEV)
            for s in range(0,len(Xt),B):
                idx=perm[s:s+B]; xb=Xt[idx]
                # swap noise: for each cell, with prob `swap` replace with a value resampled from the column
                mask=(torch.rand_like(xb)<swap)
                src=torch.randint(0,len(Xt),(xb.shape[0],xb.shape[1]),device=DEV)
                xcorr=torch.where(mask, Xt[src,torch.arange(xb.shape[1],device=DEV)], xb)
                opt.zero_grad(); rec=dae(xcorr); loss=F.mse_loss(rec,xb); loss.backward(); opt.step()
        # extract reps
        dae.eval()
        def reps(Z):
            out=[]
            with torch.no_grad():
                for s in range(0,len(Z),8192):
                    out.append(dae.encode_reps(torch.tensor(Z[s:s+8192],device=DEV)).cpu().numpy())
            return np.vstack(out).astype(np.float32)
        Rtr=reps(Ztr); Rval=reps(Zval); Rte=reps(Zte)
        # head classifier (balanced)
        seed_all(SEED+fid+1)
        head=Head(Rtr.shape[1]).to(DEV)
        cls_w=torch.tensor(len(ti)/(N_CLASSES*np.bincount(y[ti],minlength=N_CLASSES)),dtype=torch.float32,device=DEV)
        hopt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(hopt,T_max=40)
        Rt=torch.tensor(Rtr,device=DEV); yt=torch.tensor(y[ti],device=DEV,dtype=torch.long)
        best=-1; best_val=None
        for ep in range(40):
            head.train(); perm=torch.randperm(len(Rt),device=DEV)
            for s in range(0,len(Rt),B):
                idx=perm[s:s+B]; hopt.zero_grad()
                loss=F.cross_entropy(head(Rt[idx]),yt[idx],weight=cls_w); loss.backward(); hopt.step()
            sched.step()
            head.eval()
            with torch.no_grad():
                vp=F.softmax(head(torch.tensor(Rval,device=DEV)),1).cpu().numpy()
            ba=balanced_accuracy_score(y[vi],vp.argmax(1))
            if ba>best: best=ba; best_val=vp
        oof[vi]=best_val
        with torch.no_grad():
            tp=F.softmax(head(torch.tensor(Rte,device=DEV)),1).cpu().numpy()
        test_acc+=tp/len(folds)
        scores.append(best); log(f"fold{fid}: head BA={best:.6f}")
    cv=balanced_accuracy_score(y,oof.argmax(1))
    log(f"per_fold={scores}")
    log(f"OOF full BA={cv:.6f}")
    np.save(NODE_DIR/"oof.npy",oof); np.save(NODE_DIR/"test_probs.npy",test_acc)
    print(f"cv={cv:.6f}",flush=True)

if __name__=="__main__": main()
