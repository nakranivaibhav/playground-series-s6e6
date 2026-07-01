"""A4: patch-stripped hard-vote over top public subs + our bank17, as clout/finals-slot-2 candidate."""
import glob,re
from pathlib import Path
import numpy as np,pandas as pd
from collections import Counter
C=Path("comps/playground-series-s6e6"); VB=C/"refs/vote_bank"; LAB=["GALAXY","QSO","STAR"]
te=pd.read_csv(C/"data/test.csv"); ids=te['id'].values
# top distinct-score MAIN files (strip .b/.c micro-patch variants — LB-probed, private-neutral)
files=sorted(glob.glob(str(VB/"0.97*.csv")))
main=[f for f in files if re.match(r".*/0\.97\d+\.csv$",f)]  # exclude .b/.c
main=sorted(main,key=lambda f:float(re.search(r"0\.97\d+",f).group()),reverse=True)
top=main[:7]
print("voters (public):",[Path(f).name for f in top])
def load(f):
    d=pd.read_csv(f); assert (d['id'].values==ids).all(); return d['class'].values
V=[load(f) for f in top]
# add our bank17
bank=pd.read_csv(C/"refs/a1_bank_stack/submission_bank17.csv"); assert (bank['id'].values==ids).all()
ours=bank['class'].values
allv=V+[ours]
# majority vote, ties -> bank17 (ours)
M=np.array(allv)  # (k, n)
out=[]
for j in range(M.shape[1]):
    c=Counter(M[:,j]); top2=c.most_common()
    if len(top2)>1 and top2[0][1]==top2[1][1]:
        out.append(ours[j])
    else:
        out.append(top2[0][0])
out=np.array(out)
# agreement diagnostics
print("vote vs bank17 flips:",int((out!=ours).sum()),f"({100*(out!=ours).mean():.2f}%)")
print("vote vs top-public(0.97135) flips:",int((out!=V[0]).sum()))
print("class dist:",dict(Counter(out)))
sub=pd.DataFrame({"id":ids,"class":out})
(C/"refs/a4_vote").mkdir(exist_ok=True)
sub.to_csv(C/"refs/a4_vote/submission_vote.csv",index=False)
# also save the raw top-public 0.97135 as a reference finals candidate
print("wrote refs/a4_vote/submission_vote.csv")
