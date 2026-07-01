"""Error-correlation of Mitra (n134) fold-0 OOF vs the bank, same convention as
analyze_blend: err[m] = (argmax(prob) != y); err-corr = Pearson corrcoef of the
two error-indicator vectors on the SAME rows. Rule: <0.65 = decorrelated.

Computed on the fold-0 val rows only (the OOF we have). Reports BA + err-corr vs
each reference base, so we can judge whether Mitra is weak-but-decorrelated."""
import numpy as np, pandas as pd, json
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

COMP = Path("comps/playground-series-s6e6")
CLASSES = ["GALAXY", "QSO", "STAR"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}

y_all = pd.read_csv(COMP / "data/train.csv")["class"].map(LABEL_MAP).values
idx = np.load(COMP / "nodes/node_0134/oof_fold0_idx.npy")
mitra = np.load(COMP / "nodes/node_0134/oof_fold0.npy")        # (n_val, 3) aligned to idx
y = y_all[idx]
mitra_pred = mitra.argmax(1)
mitra_err = (mitra_pred != y).astype(float)
print(f"Mitra fold-0: n={len(idx)}  BA={balanced_accuracy_score(y, mitra_pred):.6f}  "
      f"err-rate={mitra_err.mean():.4f}")

REFS = {"n070(FT-T bank)": "node_0070", "n033(TabM)": "node_0033",
        "n091(CHAMPION)": "node_0091", "n063(prev champ)": "node_0063",
        "n040(core stack)": "node_0040"}
print(f"\n{'reference':>20}  {'their_BA':>8}  {'err-corr':>8}  {'verdict':>14}")
for name, node in REFS.items():
    oof = np.load(COMP / f"nodes/{node}/oof.npy")[idx]
    rp = oof.argmax(1)
    re = (rp != y).astype(float)
    ec = np.corrcoef(mitra_err, re)[0, 1]
    verdict = "DECORRELATED" if ec < 0.65 else ("borderline" if ec < 0.72 else "entangled")
    print(f"{name:>20}  {balanced_accuracy_score(y, rp):8.4f}  {ec:8.3f}  {verdict:>14}")

# context: the wall references — weak-decorrelated bases that FAILED to stack
print("\nwall context: RBF n096 0.947/corr0.53 (stack-add -0.00005) · flux n103 0.940/corr0.485")
print("tier bases all sit at err-corr >= 0.72; <0.65 + strong enough = the only stack-add hope")
