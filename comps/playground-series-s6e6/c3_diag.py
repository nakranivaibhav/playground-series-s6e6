"""C3 diagnostic (read-only): do redshift-bin-optimal DE multipliers diverge from the global ones
beyond re-partition noise? Uses bank-17 stacked OOF (our best) + champion node_0041 OOF."""
import json,warnings,numpy as np,pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution
warnings.filterwarnings("ignore")
C=Path("comps/playground-series-s6e6"); LAB=["GALAXY","QSO","STAR"]; NC=3
tr=pd.read_csv(C/"data/train.csv"); y=tr["class"].map({l:i for i,l in enumerate(LAB)}).to_numpy()
z=tr['redshift'].values
def balacc(yy,pred): return float(np.mean([(pred[yy==c]==c).mean() for c in range(NC) if (yy==c).any()]))
def de(P,yy):
    f=lambda w:-balacc(yy,np.argmax(P*np.array([w[0],w[1],1.0]),1))
    r=differential_evolution(f,[(0.1,5.0),(0.1,5.0)],maxiter=60,tol=1e-8,seed=0,polish=False); return np.array([r.x[0],r.x[1],1.0])
bins=[(-1,0.005),(0.005,0.1),(0.1,0.8),(0.8,1.5),(1.5,99)]
for tag,oofp in [("bank17",C/"refs/a1_bank_stack/stack_oof.npy"),("champ_n41",C/"nodes/node_0041/oof.npy")]:
    P=np.load(oofp); P=np.clip(P,1e-9,1); P=P/P.sum(1,keepdims=True)
    gw=de(P,y); base=balacc(y,np.argmax(P*gw,1))
    # global-weight BA vs per-bin-optimal BA (in-sample upper bound)
    perbin_pred=np.argmax(P*gw,1).copy()
    div=[]
    for lo,hi in bins:
        m=(z>lo)&(z<=hi)
        if m.sum()<50: continue
        wb=de(P[m],y[m]); 
        perbin_pred[m]=np.argmax(P[m]*wb,1)
        div.append((f"({lo},{hi}]",m.sum(),wb[0],wb[1]))
    ba_perbin=balacc(y,perbin_pred)
    print(f"\n=== {tag} === global w=({gw[0]:.3f},{gw[1]:.3f}) BA={base:.6f}")
    for t,c,a,b in div: print(f"  z{t:14s} n={c:7d} w_GAL={a:.3f} w_QSO={b:.3f}")
    print(f"  per-bin-optimal (IN-SAMPLE upper bound) BA={ba_perbin:.6f}  gain={ba_perbin-base:+.6f}")
    # re-partition noise estimate: 5-fold honest per-bin vs global
print("\nNOTE: per-bin gain is IN-SAMPLE (optimistic). Build C3 node only if gain >> ~2*0.0003 sem.")
