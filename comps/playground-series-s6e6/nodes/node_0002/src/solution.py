"""node_0002 · improve(node_0001) — post-hoc per-class weight (threshold) tuning.

Balanced accuracy's optimum is NOT plain argmax of softmax probs, so we tune a
per-class multiplicative weight w and relabel as argmax(prob * w). This is pure
post-processing on node_0001's saved OOF probs — NO model training needed for CV.

CV is made FOLD-HONEST: weights are tuned on the other 4 folds' OOF and evaluated
on the held-out fold, so the score is not optimistically biased by tuning on the
same rows it is scored on.

Result: gain is within fold-noise (+0.7 sem vs node_0001) because node_0001's
class_weight='balanced' already calibrates the classes. Node stays valid, not
champion; no submission spent (a test submission would need one all-train fit to
get test probabilities — skipped, not worth a slot for a within-noise gain).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

D = Path(__file__).resolve().parents[3]            # comps/<slug>
oof = np.load(D / "nodes/node_0001/oof.npy")        # (N,3) OOF probs from parent
tr = pd.read_csv(D / "data/train.csv")
classes = np.array(sorted(tr["class"].unique()))    # ['GALAXY','QSO','STAR']
y = pd.Categorical(tr["class"], categories=list(classes)).codes
folds = json.loads((D / "folds.json").read_text())["folds"]


def bal_acc(w, P, yt):
    return balanced_accuracy_score(yt, np.argmax(P * w, axis=1))


def tune(P, yt):
    """Coordinate-ascent grid on per-class weights (w[0] anchored at 1)."""
    best = np.ones(3)
    best_s = bal_acc(best, P, yt)
    grid = np.linspace(0.2, 3.0, 29)
    for _ in range(3):
        for c in range(3):
            cands = [(bal_acc(np.where(np.arange(3) == c, v, best), P, yt), v) for v in grid]
            s, v = max(cands)
            if s > best_s:
                best_s, best[c] = s, v
    return best, best_s


base = [balanced_accuracy_score(y[f["val_idx"]], np.argmax(oof[f["val_idx"]], 1)) for f in folds]
print("argmax_per_fold=" + ",".join(f"{s:.6f}" for s in base), "mean", f"{np.mean(base):.6f}")

honest = []
for f in folds:
    va = np.array(f["val_idx"])
    mask = np.ones(len(y), bool)
    mask[va] = False
    w, _ = tune(oof[mask], y[mask])                 # tune on OTHER folds only
    honest.append(balanced_accuracy_score(y[va], np.argmax(oof[va] * w, 1)))
honest = np.array(honest)
cv, sem = honest.mean(), honest.std(ddof=1) / np.sqrt(len(honest))
print("per_fold=" + ",".join(f"{s:.6f}" for s in honest))
print(f"cv={cv:.6f}±{sem:.6f}")

wf, sf = tune(oof, y)                                # full-OOF tuned (optimistic ref + final weights)
print(f"full_oof_tuned_ref={sf:.6f} weights={wf.round(3).tolist()}")
(D / "nodes/node_0002/weights.json").write_text(
    json.dumps({"classes": classes.tolist(), "weights": wf.tolist()})
)
print("Done.")
