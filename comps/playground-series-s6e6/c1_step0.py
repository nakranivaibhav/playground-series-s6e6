"""C1 Step-0: cleanlab confident-learning flags from champion OOF + locality check.
Abort the GPU retrain if flags are diffuse (not concentrated in the low-z GALAXY/STAR confusion zone)."""
import json,numpy as np,pandas as pd
from pathlib import Path
from cleanlab.filter import find_label_issues
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; L2I={l:i for i,l in enumerate(LAB)}
tr=pd.read_csv(C/"data/train.csv"); y=tr["class"].map(L2I).to_numpy(); n=len(tr)
oof=np.load(C/"nodes/node_0041/oof.npy")  # champion stacked OOF probs (577347,3)
oof=np.clip(oof,1e-9,1); oof=oof/oof.sum(1,keepdims=True)
iss=find_label_issues(y,oof,filter_by='confident_learning',return_indices_ranked_by=None)
flagged=np.where(iss)[0]
print(f"flagged {len(flagged)} / {n} ({100*len(flagged)/n:.3f}%)")
# per-class flag rate
for c,l in enumerate(LAB):
    m=(y==c); print(f"  {l:7s}: {iss[m].sum():6d} flagged / {m.sum():6d} ({100*iss[m].sum()/m.sum():.2f}%)")
# locality: redshift band x flagged
z=tr['redshift'].values
bands=[(-1,0.005),(0.005,0.1),(0.1,0.8),(0.8,1.5),(1.5,99)]
print("redshift-band flag concentration:")
for lo,hi in bands:
    bm=(z>lo)&(z<=hi); 
    if bm.sum()==0: continue
    print(f"  z({lo},{hi}]: {iss[bm].sum():6d}/{bm.sum():7d} flagged ({100*iss[bm].sum()/bm.sum():.2f}%)  share-of-all-flags {100*iss[bm].sum()/len(flagged):.1f}%")
# is it concentrated? compare flag rate in low-z (<0.1) vs overall
lowz=(z<=0.1); rate_low=iss[lowz].mean(); rate_all=iss.mean()
print(f"\nlow-z(<=0.1) flag-rate {100*rate_low:.2f}% vs overall {100*rate_all:.2f}%  -> concentration ratio {rate_low/rate_all:.2f}x")
print("ABORT if ratio ~1 (diffuse); PROCEED if >~1.5 (concentrated in confusion zone)")
np.save("/tmp/c1_flagged.npy",flagged)
