# %% [markdown]
# # The physics of the split: redshift, colors & the two categoricals (S6E6 EDA)
#
# This is a from-the-data EDA for **Predicting Stellar Class** (GALAXY / QSO / STAR).
# The goal isn't a model — it's to build the *physical intuition* that tells you
# which features carry the signal and why the rare classes are where the points are.
#
# **Roadmap**
# 1. The target: heavy imbalance + why **Balanced Accuracy** changes your priorities
# 2. **Redshift** — the single strongest, physically-grounded separator
# 3. **Colors** (magnitude differences) — the photometric fingerprint
# 4. The two **categoricals** (`spectral_type`, `galaxy_population`)
# 5. Where the classes actually get confused — aim your modeling there

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("/kaggle/input/playground-series-s6e6")
if not DATA.exists():
    DATA = Path("comps/playground-series-s6e6/data")
train = pd.read_csv(DATA / "train.csv")
print(train.shape)
train.head()

# %% [markdown]
# ## 1. The target — imbalance, and what Balanced Accuracy does to your incentives

# %%
counts = train["class"].value_counts()
print(counts)
print((counts / len(train)).round(3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
counts.plot.bar(ax=ax[0], color=["#4c72b0", "#dd8452", "#55a868"])
ax[0].set_title("Class counts (heavily imbalanced)")
(counts / len(train)).plot.pie(ax=ax[1], autopct="%1.0f%%", colors=["#4c72b0", "#dd8452", "#55a868"])
ax[1].set_ylabel("");  ax[1].set_title("Class share")
plt.tight_layout(); plt.show()

# %% [markdown]
# GALAXY ≈ 65%, QSO ≈ 20%, STAR ≈ 14%. But the metric is **Balanced Accuracy** =
# the *macro-average of per-class recall*, so each class is worth **1/3** regardless
# of size. Practical consequence: a point of **STAR** or **QSO** recall is worth far
# more than a point of GALAXY recall. Optimize for the rare classes, and always use
# `class_weight="balanced"` (or equivalent) so the majority doesn't drown them.

# %% [markdown]
# ## 2. Redshift — the physically strongest separator
#
# Redshift measures how much a spectrum is stretched by cosmic expansion:
# - **Stars** are in our own galaxy → redshift ≈ 0
# - **Galaxies** are nearby-to-moderate distance → low, positive redshift
# - **Quasars (QSO)** are extremely distant active galactic nuclei → high redshift
#
# This is real astrophysics, and it shows up cleanly in the data.

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for c, col in zip(["GALAXY", "QSO", "STAR"], ["#4c72b0", "#dd8452", "#55a868"]):
    z = train.loc[train["class"] == c, "redshift"]
    ax[0].hist(np.clip(z, -0.5, 3.0), bins=120, alpha=0.55, label=c, color=col, density=True)
ax[0].set_title("redshift by class (clipped to [-0.5, 3])"); ax[0].set_xlabel("redshift"); ax[0].legend()

# log-ish view of the near-zero stack
for c, col in zip(["GALAXY", "QSO", "STAR"], ["#4c72b0", "#dd8452", "#55a868"]):
    z = train.loc[train["class"] == c, "redshift"]
    ax[1].hist(np.clip(z, -0.01, 0.5), bins=120, alpha=0.55, label=c, color=col, density=True)
ax[1].set_title("zoom: redshift in [-0.01, 0.5]"); ax[1].set_xlabel("redshift"); ax[1].legend()
plt.tight_layout(); plt.show()

print(train.groupby("class")["redshift"].describe()[["mean", "50%", "min", "max"]])

# %% [markdown]
# STAR collapses onto ~0, QSO spreads to high values, GALAXY sits in between. A
# `log1p(redshift)` transform tames the long QSO tail and tends to help linear and
# NN models (trees don't care about monotone transforms). The overlap that remains —
# **low-redshift GALAXY vs STAR** — is exactly where the hard mistakes live (Section 5).

# %% [markdown]
# ## 3. Colors — the photometric fingerprint
#
# The `u, g, r, i, z` magnitudes are brightness in five filters (ultraviolet→infrared).
# Absolute magnitudes vary with distance/brightness; the **differences** between
# adjacent bands ("colors") encode the *shape* of the spectrum and are far more
# class-discriminative. Engineering `u-g, g-r, r-i, i-z` is the single highest-value
# feature step.

# %%
for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]:
    train[f"{a}-{b}"] = train[a] - train[b]

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, (a, b) in zip(axes, [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z")]):
    for c, col in zip(["GALAXY", "QSO", "STAR"], ["#4c72b0", "#dd8452", "#55a868"]):
        v = train.loc[train["class"] == c, f"{a}-{b}"]
        ax.hist(np.clip(v, -1, 3), bins=80, alpha=0.5, label=c, color=col, density=True)
    ax.set_title(f"{a}-{b}"); ax.set_xlabel("color")
axes[0].legend(); plt.tight_layout(); plt.show()

# %% [markdown]
# ### Color–color diagram (the classic astronomer's view)
# Plotting one color against another separates the classes into loci — the same
# structure a good model learns.

# %%
samp = train.sample(min(40000, len(train)), random_state=0)
plt.figure(figsize=(7, 6))
for c, col in zip(["GALAXY", "QSO", "STAR"], ["#4c72b0", "#dd8452", "#55a868"]):
    m = samp["class"] == c
    plt.scatter(samp.loc[m, "u-g"], samp.loc[m, "g-r"], s=3, alpha=0.25, label=c, color=col)
plt.xlim(-1, 3); plt.ylim(-1, 2.5); plt.xlabel("u-g"); plt.ylabel("g-r")
plt.title("color–color: u-g vs g-r"); plt.legend(markerscale=4); plt.show()

# %% [markdown]
# ## 4. The two categoricals

# %%
for cat in ["spectral_type", "galaxy_population"]:
    print(f"\n=== {cat} ===")
    ct = pd.crosstab(train[cat], train["class"], normalize="index").round(3)
    print(ct)
    ct.plot.bar(stacked=True, figsize=(7, 3), color=["#4c72b0", "#dd8452", "#55a868"])
    plt.title(f"P(class | {cat})"); plt.ylabel("share"); plt.legend(loc="center left", bbox_to_anchor=(1, .5))
    plt.tight_layout(); plt.show()

# %% [markdown]
# These are strong conditional signals. A leak-safe way to use them is **target
# encoding fit *inside each CV fold only*** (never on the full training set or test) —
# encode each category by its in-fold `P(class | category)`. Fitting the encoder on
# all of train is a classic, silent leak that inflates CV and shakes out on the LB.

# %% [markdown]
# ## 5. Where the classes get confused — aim here
#
# Combine the two strongest axes: redshift band × color. The dangerous overlap is
# **low-redshift objects** where GALAXY and STAR colors are similar.

# %%
low_z = train[train["redshift"] < 0.05]
print(f"low-z (<0.05) rows: {len(low_z)}  class mix:")
print((low_z["class"].value_counts(normalize=True)).round(3))

plt.figure(figsize=(7, 4))
for c, col in zip(["GALAXY", "STAR"], ["#4c72b0", "#55a868"]):
    v = low_z.loc[low_z["class"] == c, "g-r"]
    plt.hist(np.clip(v, -0.5, 2), bins=80, alpha=0.55, label=c, color=col, density=True)
plt.title("low-z GALAXY vs STAR overlap in g-r"); plt.xlabel("g-r"); plt.legend(); plt.show()

# %% [markdown]
# ## Takeaways
#
# 1. **Balanced Accuracy ⇒ chase the rare classes** (QSO, STAR); class-weight everything.
# 2. **Redshift is king** — add `log1p(redshift)`; it does most of the heavy lifting.
# 3. **Engineer colors** (`u-g, g-r, r-i, i-z`) — more discriminative than raw mags.
# 4. **Target-encode the categoricals *in-fold only*** to avoid a silent leak.
# 5. **The remaining errors live in low-z GALAXY↔STAR** — that's where extra
#    features / a complementary model earn their keep.
#
# Hope this gives a useful physical map of the data. Upvote if it helped, and ask
# anything in the comments. 🔭
