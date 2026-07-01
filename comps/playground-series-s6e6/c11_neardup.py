import numpy as np,pandas as pd
from pathlib import Path
C=Path("comps/playground-series-s6e6")
num=['alpha','delta','u','g','r','i','z','redshift']
tr=pd.read_csv(C/"data/train.csv"); te=pd.read_csv(C/"data/test.csv")
y=tr['class'].values
print(f"train {len(tr)} test {len(te)}")
for dp in [4,3,2]:
    trk=tr[num].round(dp).astype(str).agg('|'.join,axis=1)
    tek=te[num].round(dp).astype(str).agg('|'.join,axis=1)
    tcount=trk.value_counts()
    # test rows whose rounded key exists in train
    matchmask=tek.isin(set(trk.values))
    nmatch=int(matchmask.sum())
    # label purity among matched train keys
    if nmatch>0:
        # map each train key -> majority label + purity
        lab=pd.DataFrame({'k':trk,'y':y})
        g=lab.groupby('k')['y'].agg(lambda s:s.value_counts(normalize=True).iloc[0])
        matched_keys=tek[matchmask]
        pur=g.reindex(matched_keys.values).mean()
    else: pur=float('nan')
    print(f"round {dp}dp: test-rows-with-train-match={nmatch:6d} ({100*nmatch/len(te):.3f}%)  mean-matched-train-key-purity={pur:.4f}")
# exact (full precision)
trk=tr[num].astype(str).agg('|'.join,axis=1); tek=te[num].astype(str).agg('|'.join,axis=1)
ex=int(tek.isin(set(trk.values)).sum())
print(f"EXACT full-precision test<->train matches: {ex}")
print("DECISION: actionable only if >=1000 matches at <=3dp AND purity>=0.99 AND champ misclassifies >=10% of such.")
