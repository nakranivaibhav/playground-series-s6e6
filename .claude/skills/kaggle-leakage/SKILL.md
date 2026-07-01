---
name: kaggle-leakage
description: Reference — leakage & validation discipline (preloaded into kaggle-developer, which self-gates every node it builds). The fast in-node self-check protocol — input checks BEFORE training, output checks AFTER; seconds each, never a training run. Use when validating a node's CV, deciding whether a score counts, or judging a CV↔LB gap.
allowed-tools: Bash, Read
---

# Leakage & validation discipline (reference)

A node's CV **does not count until the self-checks pass**. Leakage voids a score
regardless of how good the CV looks (CLAUDE.md hard rule 3). This skill is the
standing checklist the `kaggle-developer` applies (self-gate) to every node it
builds — including data-cleaning and feature-engineering nodes.

**Design: every check is a super-fast data/output computation (seconds) or a read
of your own code. NO check involves a training run.** There is no scanner tool —
*you* are the scanner; record the results as the gate booleans in `node.md`.

---

## Pre-train checks (BEFORE launching training — a leak caught here costs zero GPU)

Run these on the assembled feature matrix + your own `solution.py`:

1. **target not in features** *(error)* — the target, and any deterministic alias
   of it (a renamed copy, a leaked aggregate), absent from the feature list.
   Exact set-check.
2. **id / row-order not in features** *(error)* — the id column, and anything
   monotone in row order, absent.
   ```python
   assert TARGET not in features and ID_COL not in features
   ```
3. **single-feature↔target sweep** *(error on a hit)* — on a ≤50k-row sample, a
   feature with near-perfect |corr| (or single-feature AUC) vs the target
   (≥ 0.999) is a leak smell — stop and inspect, don't train.
   ```python
   s = train.sample(min(50_000, len(train)), random_state=0)
   ys = pd.factorize(s[TARGET])[0] if s[TARGET].dtype == object else s[TARGET]
   for c in features:
       x = pd.to_numeric(s[c], errors="coerce")
       if x.nunique() > 1 and abs(np.corrcoef(x.fillna(x.mean()), ys)[0, 1]) >= 0.999:
           raise SystemExit(f"leak smell: {c} ~ target")
   ```
4. **fit-inside-fold, by reading your own fold loop** *(error)* — for EVERY
   fitted transform (scaler / encoder / imputer / target-encoder / selector) AND
   every cross-row stat (kNN density, group aggregate, neighbor count): confirm
   it is computed **inside the fold loop from train-fold rows only** — never on
   full train, never on `concat([train, test])`. Walk each `fit_in_fold`
   feature-set the node consumes (`uses_data` → `data.md`) explicitly. (The final
   refit-on-all-train for the submission, AFTER the OOF loop, is correct and
   expected — the leak is a *transform* fit on full data before/during CV, or
   anything fit on test.)
5. **frozen folds** *(error)* — fold indices come from `folds.json`; never
   recomputed, never reshuffled.
6. **train↔test near-duplicates** *(warn)* — on a sample, exact / rounded-value
   matches across train↔test on the feature columns (critical for image/text).

## Post-train checks (on the OUTPUTS — no extra compute)

7. **OOF complete** *(error)* — the OOF array covers every train row exactly
   once, each row predicted by the fold that held it out; no NaN.
8. **distribution sane** *(error)* — predictions not collapsed to one value, not
   inverted, inside the target's valid range; class probabilities sum to 1.
9. **submission schema** *(error)* — `uv run tools/validate_submission.py`
   against `sample_submission.csv` (ids, header, row count).
10. **cv-too-good** *(warn → human)* — compare to the parent/baseline: a jump far
    beyond anything the lineage has produced, or a near-perfect score, goes in
    `gate_note` for human eyes BEFORE a submission is spent. A warn, not a void.

## Recording (the gate booleans in `node.md`)

```
gates: {schema_ok: ←9, oof_full: ←7, no_nan: ←7, dist_sane: ←8,
        leak_clean: ←checks 1–6 all clean, cv_too_good: ←10, passed}
```
- Any *error*-level failure → `gates.leak_clean: false`, `leak: VOID`,
  `status: buggy` — the CV does not count, no matter its value; the fix is a
  **debug** child, not a re-score.
- `passed` is true only when every required gate is true (`cv_too_good: true` is
  a warn the human eyeballs, never a blocker).

---

## The rules (do not negotiate these)

1. **Leakage voids a score, full stop.** An error-level failure marks the node
   `buggy` and its CV does not count — regardless of how good the CV is.
2. **A CV↔LB gap is a DIAGNOSTIC ONLY — never an auto-demote.** Trust a
   well-built local CV over the small, noisy public LB (CLAUDE.md rule 6). Log
   the gap in the journal, surface it in a Decision Card, and only investigate
   (re-read folds for a group/time mismatch; consider a one-off adversarial-
   validation diagnostic) — never silently swap the champion because the LB
   disagreed.
3. **Adversarial validation is NOT a standing gate.** One-off diagnostic only,
   when a large unexplained CV↔LB gap appears; never demote a node on its output.
4. **Every node clears the self-checks before its CV counts** — cleaning and
   feature-engineering nodes included. A feature that "improves CV" but fails
   fit-inside-fold is buggy, not good.
5. **Folds are frozen.** Read `folds.json`; never call `make_folds.py` again
   mid-run. Every transform fits inside the train fold only.
6. **Group / temporal leakage is prevented upstream by construction** —
   `tools/make_folds.py` + the frozen `folds.json` (a group never straddles
   folds; time-series folds are past-only) — not re-checked per node. No leakage
   check may require a training run (shuffled-label controls stay removed).
