# %% [markdown]
# # Beyond the scalar: a structural diagnostic for Balanced Accuracy
#
# Balanced Accuracy is a **macro-average of per-class recall** — averaging three
# numbers into one hides *which class* and *where* your model actually changed.
#
# This notebook is a drop-in toolkit to compare **any two models** structurally:
#
# 1. **per-class recall delta** — which class moved
# 2. **confusion-matrix delta** — where the rows migrated
# 3. **paired flip analysis + McNemar test** — are the fixes vs breaks real or churn?
# 4. **paired bootstrap** — P(model B > model A), far finer than the "2×SEM" rule
# 5. **slice by redshift band** — find *complementary* models a flat BA would hide
#
# Point it at two out-of-fold prediction arrays (or two submission files) and read
# the structure instead of a single number. Built for Playground S6E6 but the
# functions are competition-agnostic.

# %%
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from scipy.stats import binomtest

LABELS = ["GALAXY", "QSO", "STAR"]
L2I = {c: i for i, c in enumerate(LABELS)}


# %% [markdown]
# ## The toolkit (copy these five functions into your own work)

# %%
def per_class_recall_delta(y, pred_a, pred_b, labels=LABELS):
    """Per-class recall for A and B, and the delta. The first thing to read."""
    ra = recall_score(y, pred_a, labels=labels, average=None, zero_division=0)
    rb = recall_score(y, pred_b, labels=labels, average=None, zero_division=0)
    return pd.DataFrame({"recall_A": ra, "recall_B": rb, "delta": rb - ra}, index=labels)


def confusion_delta(y, pred_a, pred_b, labels=LABELS):
    """confusion(B) - confusion(A). Rows = true class, cols = predicted class.
    Positive on the diagonal = B fixed those; positive off-diagonal = B's new errors."""
    ca = confusion_matrix(y, pred_a, labels=labels)
    cb = confusion_matrix(y, pred_b, labels=labels)
    return pd.DataFrame(cb - ca, index=[f"true_{c}" for c in labels],
                        columns=[f"pred_{c}" for c in labels])


def flip_analysis(y, pred_a, pred_b):
    """Row-level fixes (A wrong, B right) vs breaks (A right, B wrong) + McNemar p."""
    ca, cb = (pred_a == y), (pred_b == y)
    fixes = int((~ca & cb).sum())
    breaks = int((ca & ~cb).sum())
    n = fixes + breaks
    p = binomtest(min(fixes, breaks), n, 0.5).pvalue if n else 1.0
    return {"fixes": fixes, "breaks": breaks, "net": fixes - breaks, "mcnemar_p": p}


def bootstrap_p_better(y, pred_a, pred_b, n_boot=2000, seed=0):
    """Paired bootstrap of BA(B) - BA(A); returns P(B>A) and a 90% CI on the delta.
    Resamples the same rows for both models, so it's a *paired* test of the real gap."""
    rng = np.random.default_rng(seed)
    N = len(y)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, N)
        deltas[i] = (balanced_accuracy_score(y[idx], pred_b[idx])
                     - balanced_accuracy_score(y[idx], pred_a[idx]))
    return {"P(B>A)": float((deltas > 0).mean()),
            "delta_mean": float(deltas.mean()),
            "ci90": (float(np.quantile(deltas, .05)), float(np.quantile(deltas, .95)))}


def flip_by_region(y, pred_a, pred_b, region, n_bands=5):
    """Flip analysis within quantile bands of a physical variable (e.g. redshift).
    This is how you separate a genuinely complementary model from pure churn."""
    edges = np.quantile(region, np.linspace(0, 1, n_bands + 1)[1:-1])
    bands = np.digitize(region, edges)
    rows = []
    for b in np.unique(bands):
        m = bands == b
        rows.append({"band": int(b), "n": int(m.sum()), **flip_analysis(y[m], pred_a[m], pred_b[m])})
    return pd.DataFrame(rows)


def full_report(y, pred_a, pred_b, region=None, name_a="A", name_b="B"):
    print(f"BA({name_a}) = {balanced_accuracy_score(y, pred_a):.6f}")
    print(f"BA({name_b}) = {balanced_accuracy_score(y, pred_b):.6f}")
    print(f"delta        = {balanced_accuracy_score(y, pred_b)-balanced_accuracy_score(y, pred_a):+.6f}\n")
    print("── per-class recall delta ──");   print(per_class_recall_delta(y, pred_a, pred_b), "\n")
    print("── confusion delta (B − A) ──");  print(confusion_delta(y, pred_a, pred_b), "\n")
    print("── flip analysis ──");            print(flip_analysis(y, pred_a, pred_b), "\n")
    print("── bootstrap ──");                print(bootstrap_p_better(y, pred_a, pred_b), "\n")
    if region is not None:
        print("── flip by region band ──");  print(flip_by_region(y, pred_a, pred_b, region))


# %% [markdown]
# ## Worked example on the competition data
#
# We train two genuinely different models (a gradient-boosted tree and a logistic
# regression on color features), get honest out-of-fold predictions for each, and
# run the structural comparison. Watch how a small BA gap becomes a *story*.

# %%
DATA = Path("/kaggle/input/playground-series-s6e6")
if not DATA.exists():                      # local fallback
    DATA = Path("comps/playground-series-s6e6/data")
train = pd.read_csv(DATA / "train.csv")

# A few stateless color/redshift features (no fitting, leak-safe)
def fe(df):
    X = pd.DataFrame(index=df.index)
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r")]:
        X[f"{a}-{b}"] = df[a] - df[b]
    for c in ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]:
        X[c] = df[c]
    X["log1p_z"] = np.log1p(df["redshift"] - df["redshift"].min() + 1e-4)
    return X

X = fe(train).values
y = train["class"].map(L2I).values
redshift = train["redshift"].values

# %%
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

oof_lgb = np.zeros(len(y), int)
oof_lr = np.zeros(len(y), int)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for tr, va in skf.split(X, y):
    # model A: LightGBM
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                           class_weight="balanced", n_jobs=-1, verbose=-1)
    m.fit(X[tr], y[tr])
    oof_lgb[va] = m.predict(X[va])
    # model B: scaled logistic regression (a very different decision surface)
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    lr.fit(sc.transform(X[tr]), y[tr])
    oof_lr[va] = lr.predict(sc.transform(X[va]))

# decode to label strings for readable confusion tables
inv = {v: k for k, v in L2I.items()}
y_s = np.array([inv[i] for i in y])
a_s = np.array([inv[i] for i in oof_lgb])
b_s = np.array([inv[i] for i in oof_lr])

# %%
full_report(y_s, a_s, b_s, region=redshift, name_a="LightGBM", name_b="LogReg")

# %% [markdown]
# ### How to read the output
#
# - **per-class recall delta** tells you the *kind* of difference (e.g. "LogReg
#   trades GALAXY recall for QSO recall") — not just that BA fell.
# - **confusion delta** shows the exact migrations driving it.
# - **flip analysis + McNemar p**: a big `|net|` with `p ≪ 0.05` is a *real,
#   concentrated* difference; a small net with `p ≈ 0.4` is churn that'll reverse
#   out of sample.
# - **bootstrap `P(B>A)`**: I promote a candidate only at `P ≥ 0.90` **and** a
#   matching move on an untouched holdout — far safer than "is the gap > 2×SEM?".
# - **flip by region**: if the fixes concentrate in one redshift band and survive
#   on a holdout, the model is **complementary** — keep it for your ensemble even
#   when global BA looks flat. That's the gain a single scalar throws away.
#
# Swap in your own two OOF arrays (or two `submission.csv` files mapped to labels)
# and the same five functions give you the full structural read. If this helped,
# an upvote is appreciated — and happy to answer questions in the comments. 📊
