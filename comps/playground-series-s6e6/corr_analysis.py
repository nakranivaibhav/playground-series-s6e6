"""Error-correlation + decorrelated-blend analysis of the latest + off-meta models vs the best stacks.
Answers: are the off-meta / latest models DE-correlated, and does adding the decorrelated ones
(esp the DAE) help a blend even when weaker solo?"""
import json,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; NC=3
tr=pd.read_csv(C/"data/train.csv"); n=len(tr); y=tr["class"].map({l:i for i,l in enumerate(LAB)}).to_numpy()
folds=json.loads((C/"folds.json").read_text())["folds"]; fval=[np.asarray(f["val_idx"]) for f in folds]
def lp(a): return np.log(np.clip(a,1e-7,1.0))
def oo(b): return np.load(C/"nodes"/b/"oof.npy")
# reference stacks
champ=np.load(C/"nodes/node_0041/oof.npy")            # champion stacked OOF
bank=np.load(C/"refs/a1_bank_stack/stack_oof.npy")     # bank-17 stacked OOF (best)
# candidate models: off-meta exploration + this round's latest
CAND={"n0049 binchain":"node_0049","n0050 OvR":"node_0050","n0051 FT-T":"node_0051",
      "n0055 DCN":"node_0055","n0056 1D-CNN":"node_0056","n0059 cleanlab":"node_0059",
      "n0060 provenance":"node_0060","n0061 GCE-TabM":"node_0061","n0062 DAE":"node_0062"}
def err(P): return (P.argmax(1)!=y).astype(np.float32)
def ba(P): return float(np.mean([(P.argmax(1)[y==c]==c).mean() for c in range(NC)]))
e_bank=err(bank); e_champ=err(champ)
print(f"{'model':18s} {'soloBA':>8s} {'errCORR vs bank':>15s} {'errCORR vs champ':>16s} {'uniqueErr%':>11s}")
print(f"{'BANK17 stack':18s} {ba(bank):8.5f} {'--':>15s}")
for nm,nid in CAND.items():
    try: P=oo(nid)
    except: print(f"{nm}: no oof"); continue
    e=err(P)
    cb=np.corrcoef(e,e_bank)[0,1]; cc=np.corrcoef(e,e_champ)[0,1]
    # fraction of this model's errors where bank is CORRECT (potential to help)
    uniq=((e==1)&(e_bank==0)).sum()/max(1,(e==1).sum())
    print(f"{nm:18s} {ba(P):8.5f} {cb:15.3f} {cc:16.3f} {100*uniq:10.1f}%")
