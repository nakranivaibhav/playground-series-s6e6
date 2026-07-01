# The CV mirage that cost me ~0.008 LB — and the holdout habit that catches it

**TL;DR:** I built a model with a *better* cross-validation score than my champion. It would have **dropped my leaderboard score by ~0.008** if I'd trusted it. Here's the failure mode, why a normal 5-fold CV didn't catch it, and the one cheap habit that does.

## What happened

I was hunting for gains on the two minority classes (QSO, STAR), since **Balanced Accuracy weights all three classes equally** and the easy points are in the rare classes. So I trained a *specialist* — a model tuned hard to separate the two classes that get confused most in a particular region of the feature space.

The result looked fantastic:

| model | 5-fold CV (Balanced Accuracy) |
|---|---|
| my general champion | 0.97036 |
| the new specialist | **0.97088**  ← higher! |

A clean **+0.0005 CV improvement**, leakage-checked, reproducible. Every instinct said *promote it.*

Then I submitted it as a probe. Public LB: **0.9624** — a faceplant of nearly **0.008** below the champion's 0.9712.

## Why 5-fold CV didn't catch it

The specialist learned a decision rule that fit the **specific noise pattern shared across all five training folds** in its target region. Because every fold is drawn from the same training distribution, that pattern shows up in every fold's validation slice too — so the CV mean rises. It's not classic leakage (no target bleed, no row duplication). It's **a model overfitting a localized quirk that the whole training set happens to share but the test set does not.**

A single CV scalar can't see this. It just reports "0.97088 > 0.97036, promote."

## The habit that catches it: an inviolable holdout the CV never touches

Carve out one fold (or a fixed random slice) at the very start and **never** use it for *anything* that touches model choice — not training, not feature-fitting, not hyperparameter tuning, not early-stopping. It is a stand-in for the private leaderboard that you can check as often as you like.

The rule becomes:

> A gain counts only if it shows up on **both** the working CV **and** the untouched holdout.

Re-running the specialist with that lens:

| | working CV (folds 0–3) | untouched holdout (fold 4) |
|---|---|---|
| champion | 0.9703 | 0.9705 |
| specialist | **0.9709** (better) | **0.9700** (worse!) |

The gain **reverses** on the holdout. That reversal is the mirage signature — and it's exactly what the private LB did to it. Caught for free, no submission spent.

## A second, finer guard: bootstrap the OOF rows

The holdout is binary (moved / didn't). To quantify *how confident* you should be that candidate > champion, **paired-bootstrap the out-of-fold predictions**: resample the OOF rows (with replacement) a few thousand times, recompute `BA(candidate) − BA(champion)` each time, and read off `P(candidate > champion)`.

```python
import numpy as np
from sklearn.metrics import balanced_accuracy_score as ba

def p_better(y, oof_champ, oof_cand, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(y); wins = 0
    for _ in range(n):
        idx = rng.integers(0, N, N)
        d = ba(y[idx], oof_cand[idx]) - ba(y[idx], oof_champ[idx])
        wins += d > 0
    return wins / n
```

I promote only on `P ≥ 0.90` **and** a holdout that moves the right way. The specialist scored `P ≈ 0.81` and a negative holdout delta — a clean reject. This is far finer than the usual "is the gap bigger than 2× the fold std-error?" rule, which is blind to localized effects.

## Takeaways

1. **A higher CV is a hypothesis, not a verdict.** Especially for specialists targeting a sub-region or a minority class.
2. **Keep one slice of data radioactive.** Never let it influence a single decision. It's your private-LB simulator.
3. **Promote on holdout-agreement + bootstrap confidence**, not on a single CV scalar.

Happy to share the bootstrap/holdout diagnostic code if useful — drop a comment. Hope this saves someone a wasted submission (or a shake-up). 🔭
