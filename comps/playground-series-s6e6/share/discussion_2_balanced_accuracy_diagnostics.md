# Stop reading one number: Balanced Accuracy hides where your model actually changed

**TL;DR:** Balanced Accuracy is a *macro-average of per-class recall*. Averaging three numbers into one throws away exactly the information you need to improve — *which class, and where*, your change helped or hurt. Here's a small diagnostic toolkit (per-class recall delta → confusion-matrix delta → paired flip analysis with a significance test) that turns "my CV went up 0.0003, is that real?" into a structural answer.

## Why the scalar lies to you

This competition scores `balanced_accuracy_score`:

```
BA = (recall_GALAXY + recall_QSO + recall_STAR) / 3
```

Two consequences people underuse:

1. **Each class is worth 1/3**, regardless of size. GALAXY is ~65% of the data but contributes the same 1/3 as STAR (~14%). A change that fixes 500 STAR rows is worth far more than one that fixes 500 GALAXY rows.
2. **A flat BA can hide a real, useful change.** A model can be *net-zero* on global BA while genuinely fixing a block of one class and breaking an equal-sized block of another. That model is still valuable in an ensemble — but the scalar says "wash, discard."

So when you compare two models, don't compare two numbers. Compare *structures*.

## Step 1 — per-class recall delta

The first thing to print isn't the BA. It's where the recall moved:

```python
from sklearn.metrics import recall_score
import numpy as np

def per_class_recall(y_true, y_pred, labels):
    r = recall_score(y_true, y_pred, labels=labels, average=None)
    return dict(zip(labels, r))

labels = ["GALAXY", "QSO", "STAR"]
a = per_class_recall(y, pred_A, labels)
b = per_class_recall(y, pred_B, labels)
for c in labels:
    print(f"{c:7s}  {a[c]:.4f} -> {b[c]:.4f}   Δ={b[c]-a[c]:+.4f}")
```

Now "+0.0003 BA" becomes a sentence like *"GALAXY −0.001, QSO +0.000, STAR +0.002"* — instantly you know the change is a STAR-recall play, and you can reason about whether that's robust.

## Step 2 — confusion-matrix delta

Where did the moved rows go? A 3×3 delta of `confusion(B) − confusion(A)` shows the exact migrations:

```python
from sklearn.metrics import confusion_matrix
d = confusion_matrix(y, pred_B, labels=labels) - confusion_matrix(y, pred_A, labels=labels)
print(np.array2string(d))   # rows = true class, cols = predicted; +off-diagonal = new errors
```

A healthy change concentrates positive mass on the diagonal and pulls it off one specific off-diagonal cell (e.g. the QSO↔GALAXY confusion). A change that just shuffles mass around the off-diagonals is churn.

## Step 3 — paired flip analysis + McNemar (is it signal or noise?)

This is the one most people skip. Compare the two models **row by row** on the same validation rows:

- **fixes** = rows A got wrong that B got right
- **breaks** = rows A got right that B got wrong

```python
correct_A = (pred_A == y)
correct_B = (pred_B == y)
fixes  = int((~correct_A &  correct_B).sum())
breaks = int(( correct_A & ~correct_B).sum())
```

Then ask whether `fixes` vs `breaks` is a real asymmetry or just coin-flips, with **McNemar's exact test** on the discordant pairs:

```python
from scipy.stats import binomtest
p = binomtest(min(fixes, breaks), fixes + breaks, 0.5).pvalue
print(f"fixes={fixes}  breaks={breaks}  net={fixes-breaks:+d}  McNemar p={p:.2e}")
```

`net = +12, p = 0.4` → churn, ignore it. `net = +260, p = 1e-30` → a real, concentrated block of corrections, even if global BA barely moved.

## Step 4 (the payoff) — slice it by a physical region

Because this is astronomy, **redshift** physically separates the classes (stars ≈ 0, galaxies low, quasars high). Run Step 3 *within redshift bands* and you'll often find the change is doing something real in one band and nothing elsewhere:

```python
bands = np.digitize(redshift, np.quantile(redshift, [.2,.4,.6,.8]))
for bnd in np.unique(bands):
    m = bands == bnd
    fx = int((~correct_A[m] & correct_B[m]).sum())
    bk = int(( correct_A[m] & ~correct_B[m]).sum())
    print(f"band {bnd}: fixes={fx} breaks={bk} net={fx-bk:+d}")
```

This is how you tell a *genuinely complementary* model (concentrated, significant fixes in one band — keep it for your ensemble even at flat BA) from a *wash* (fixes and breaks scattered evenly).

## Why bother?

Once you read structure instead of a scalar, three things change:
- You stop discarding flat-BA models that are actually **complementary** (ensemble gold).
- You stop promoting models whose "+0.0003" is **churn** that'll reverse on the private set.
- You aim your next experiment at a **named** weakness ("QSO recall in the high-z band") instead of guessing.

I packaged all of this into a single drop-in function (two prediction files → full structural report). I'll share the notebook in a comment — let me know if it's useful. 📊
