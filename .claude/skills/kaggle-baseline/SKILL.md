---
name: kaggle-baseline
description: Stage 3 — build the dumbest defensible baseline (mean/median or base-rate) as node_0000, score it on the frozen folds, promote to champion, then make the first real submission to prove the pipe end-to-end. Use when folds.json exists and there's no champion yet, or the human says "baseline" / "first submission" / "/kaggle-baseline".
argument-hint: <slug>
allowed-tools: Bash, Read, Write, Edit
---

# kaggle-baseline — the dumb baseline that becomes the first champion

Goal: build the **simplest prediction that cannot be wrong about the schema**, prove
it scores under our own frozen CV, validate the submission file, promote it to
`champion/`, and spend submission #1 on it. This proves the data → CV → submission
→ Kaggle pipe works *before* a single model is trained. If this stage is green,
every later node only changes the prediction, never the plumbing.

`<slug>` comes from `$1` (or the only `comps/` dir). All paths below are
`comps/<slug>/...`. Dates are ALWAYS `date -u` — never typed.

## Preconditions (read, don't assume)
1. `Read comps/<slug>/spec.md` — pull the fenced ```yaml machine block:
   `target_col`, `id_col`, `metric`, `metric_direction` (minimize|maximize),
   `task_type` (regression|classification_*), and the sample-submission value
   column name(s).
2. `Read comps/<slug>/validation.md` and confirm `comps/<slug>/folds.json` exists.
   If folds.json is missing, STOP — go run `/kaggle-validate` first.
3. `Read comps/<slug>/progress.md`; confirm `validation` is ticked and `baseline`
   is not. Confirm `comps/<slug>/data/{train.csv,test.csv,sample_submission.csv}`
   exist (whatever the comp's filenames are — use the ones in spec.md).
4. `Read comps/<slug>/config.md` for the autonomy mode (gates the submit step).
5. If `comps/<slug>/champion/` already has a `submission.csv`, a champion exists —
   this skill is for the FIRST one only. Stop and report.

## Step 1 — create node_0000 and its node.md
```bash
slug=<slug>; node=comps/$slug/nodes/node_0000
mkdir -p $node/src
NOW=$(date -u +%Y-%m-%dT%H:%MZ)
```
Write `$node/node.md` from the CLAUDE.md `node.md` template (frontmatter +
`## plan` body, **no checkboxes**), with `op: draft`, `parents: [root]`,
`family: baseline`, `status: running`, `stage: proposed`, `metric`/`direction`
from spec.md, `created: $NOW`. Leave the metric/gate fields null for now —
they get filled in as the node progresses (`cv`, `sem`, `folds`, `baseline_cv`,
the `gates:` booleans, `leak`); there is **no** `metrics.md` and
**no** `gate_report.md`, those values live in this frontmatter. Fill the
`## plan` body:
- **built on:** root (nothing inherited — this is the floor)
- **change:** the constant rule (which constant + why, 1–2 lines, e.g. "predict
  the global train median for every test row")
- **hypothesis:** "establishes the schema-correct data→CV→submit pipe and a floor
  CV every later node must beat"
- **target:** the official metric + direction

This file is the `proposed` artifact — the frontmatter already reads
`stage: proposed`. Advance `stage` only after each later artifact exists
(artifact-then-mark).

Append one line to `comps/$slug/journal.md`:
`$NOW node_0000 draft(root) baseline — predict <mean|median|base-rate> · status=running`.

## Step 2 — write the baseline solution (`$node/src/solution.py`)
The script is **self-contained** and does TWO jobs in one run: (a) compute CV
under folds.json, (b) emit `submission.csv`. Pick the constant by task:

| task_type | constant predicted for every row | why |
|---|---|---|
| regression, metric=RMSE/MAE | train **median** of target | median ~minimizes MAE, robust for RMSE |
| regression, metric=RMSLE | `expm1(mean(log1p(y_train)))` | RMSLE optimum is the log-space mean |
| classification (proba metric, e.g. AUC/logloss) | train **base rate** `mean(y==1)` per class | constant prob = the prior |
| classification (label metric, e.g. accuracy/F1) | **majority class** | the mode is the label-metric optimum |

Hard rules the script obeys:
- The constant is fit **inside each train fold only** (recompute per fold from the
  fold's train rows), then scored on that fold's `val_idx`. Never fit on full
  train for the CV number — that is the fit-inside-fold discipline even for a
  constant, and it keeps the leakage scan honest.
- Score with the comp's **official metric** in its **official direction** (read
  from spec.md). Print `cv=<mean>±<sem>` on its own line, plus per-fold values.
- For the *submission*, the constant is fit on **all** train rows (that's correct —
  test is never in the fit) and broadcast to every test id.
- Output columns must byte-match `sample_submission.csv`'s header and id set.
- `features = []` (a constant uses no features) — so the target/id-not-in-features
  self-check is trivially true.

Sketch (adapt names from spec.md; do not hardcode `Id`/`target`):
```python
import json, numpy as np, pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[3]             # comps/<slug>  (src→node→nodes→<slug>)
slug, TARGET, IDC = "<slug>", "<target_col>", "<id_col>"
tr = pd.read_csv(D/"data/train.csv"); te = pd.read_csv(D/"data/test.csv")
samp = pd.read_csv(D/"data/sample_submission.csv")
folds = json.loads((D/"folds.json").read_text())["folds"]
def const_from(y):      # the dumb predictor, fit on given rows only
    return float(np.median(y))                      # swap per table above
def metric(y, p):       # official metric, official direction handled by caller
    return float(np.sqrt(np.mean((y-p)**2)))        # swap per spec.md
scores=[]
for f in folds:
    va = np.array(f["val_idx"]); mask=np.ones(len(tr),bool); mask[va]=False
    c = const_from(tr.loc[mask, TARGET].values)     # FIT INSIDE FOLD
    scores.append(metric(tr.loc[va, TARGET].values, np.full(va.shape, c)))
cv=float(np.mean(scores)); sem=float(np.std(scores,ddof=1)/np.sqrt(len(scores)))
print("per_fold="+",".join(f"{s:.6f}" for s in scores)); print(f"cv={cv:.6f}±{sem:.6f}")
c_full = const_from(tr[TARGET].values)              # full-fit ok: test not used
sub = samp.copy(); val_cols=[c for c in samp.columns if c!=IDC]
sub[IDC]=te[IDC].values
for col in val_cols: sub[col]=c_full
sub.to_csv(D/"nodes/node_0000/submission.csv", index=False)
print("wrote submission.csv rows", len(sub))
```
For multi-column classification submissions (one prob column per class), set each
class column to that class's base rate; for a single-prob binary column, use
`mean(y==1)`. Match `sample_submission.csv` exactly.

## Step 3 — run it (capture log, artifact-then-mark)
Fast (seconds), so run foreground; only background per the CLAUDE.md marker
pattern if it ever takes minutes.
```bash
uv run python $node/src/solution.py > $node/train.log 2>&1; echo "exit=$?"
grep -E "cv=|Traceback|Error|Killed" $node/train.log
```
No traceback → `train.log` exists, so advance `stage: built` in node.md's
frontmatter. Then write the CV numbers straight into the **node.md frontmatter**
(no `metrics.md`): set `cv: <mean>`, `sem: <sem>`, `folds: [<per-fold scores>]`,
`baseline_cv: <mean>` (this constant *is* the baseline) — the score is computed
within this build step, so `stage` stays `built`. There is no separate unit-test
gate for a constant — the fast self-checks + validate are the gates.

## Step 4 — fast self-checks (constant baseline passes trivially)
No tool runs here — these are the developer-style in-node self-checks (the
`kaggle-leakage` checklist), all true by construction for a constant (no features
to leak through):
- **target/id not in features** — trivially true: `features = []`.
- **OOF complete** — every fold's `val_idx` got a prediction (the loop covers all
  folds, no row skipped).
- **no NaN** — the constant is finite; no NaN/inf in the OOF scores or submission.
- **distribution sane** — predictions are a single finite constant (expected).
- **schema valid** — confirmed by `tools/validate_submission.py` in Step 5.

Record the result in the node.md `gates:` frontmatter: set `leak_clean: true`
and the structural booleans (`schema_ok`, `oof_full`, `no_nan`, `dist_sane`,
`cv_too_good`) accordingly; set `leak: clean`. (If solution.py ever accidentally
used a feature/id, the target/id-not-in-features check would fail — fix solution.py,
don't override the gate: set `leak: VOID` and the failing boolean false; the CV
does not count.)

## Step 5 — validate the submission file (the schema gate)
```bash
uv run tools/validate_submission.py \
  --submission $node/submission.csv \
  --sample comps/$slug/data/sample_submission.csv --id <id_col>
echo "valid_exit=$?"
```
Must print `OK:` and exit 0. Any `INVALID:` line (column/row/id/NaN/inf) → fix
solution.py and rerun Steps 3–5. On `OK:`, finalize the node.md `gates:`
frontmatter (no `gate_report.md`): set `schema_ok: true`, and `passed: true` only
once every required gate boolean is true. Advance `stage: reviewed`.

## Step 6 — create graph.md and make node_0000 the champion
This is the first valid node, so it is the champion by definition (best valid CV).
1. Create `comps/$slug/graph.md` (the map — there is no `tree.md`) from the
   CLAUDE.md template: a header line (`metric: <metric> (<direction>) · champion:
   node_0000 (cv <cv> · lb —) · updated $(date -u +%F)`), a Mermaid `graph LR`
   with the root→node_0000 edge and the champion styled, and a `## nodes` table
   whose last column is the node-record path:
   ````markdown
   # <slug> — experiments
   metric: <metric> (<direction>) · champion: node_0000 (cv <cv> · lb —) · updated <date -u +%F>

   ```mermaid
   graph LR
       root --> node_0000[node_0000 · baseline · <cv>]:::champ
       classDef champ fill:#cfc,stroke:#070;
   ```

   ## nodes
   | node | what it is | cv | lb | status | detail |
   |------|------------|----|----|--------|--------|
   | node_0000 | baseline · constant <mean\|median\|base-rate> | <cv> | — | champion | `nodes/node_0000/node.md` |
   ````
2. Byte-copy into `champion/` (cp, never symlink — CLAUDE.md semantics):
```bash
mkdir -p comps/$slug/champion
cp -r $node/src comps/$slug/champion/src
cp $node/submission.csv comps/$slug/champion/submission.csv
```
3. Write `comps/$slug/champion/README.md`: node_0000, the constant used, `cv=…`,
   metric+direction, and "first champion — dumb baseline, proves the pipe."
4. In `$node/node.md` frontmatter set `status: champion`,
   `decided: $(date -u +%Y-%m-%dT%H:%MZ)`, and advance `stage: decided`. Append a
   `journal.md` line: `<NOW> node_0000 → champion cv=<…> (<metric> <direction>)`.

## Step 7 — SUBMIT GATE (spends 1 of the daily limit)
A real submission is irreversible + rate-limited → it is a **hard human gate**
except in `full_auto`. Check the budget first (derived, never stored); the daily
limit comes from spec.md's `daily_submission_limit` (asked from the human at
kaggle-start), never a literal:
```bash
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' comps/$slug/spec.md)
uv run tools/kaggle_io.py budget --ledger comps/$slug/submissions.md --limit "$lim"
```
Render the card in the CLAUDE.md Decision Card format, with this
stage-specific content:
- **stage:** baseline · first submission
- **What's going on:** the dumbest possible prediction is built and passes our own checks.
- **Found / propose:**
  - node_0000 = constant <mean|median|base-rate>, cv=<…> (<metric>)
  - submission.csv validated against sample_submission (schema OK)
  - this is a dry run of the whole pipe — not a real model yet
- **Why:** proves data→CV→submit→Kaggle works before we spend effort modelling.
- **Cost:** ~0 compute · spends submission <used+1>/<lim> today (resets 00:00 UTC)
- `interactive` / `auto_except_submit`: **wait** for approval (the submit gate is
  human in both). On "skip", leave node_0000 as champion, do NOT submit, mark
  progress and stop.
- `full_auto`: proceed without waiting.

On approval (and only if `remaining > 0`), hand off to the submit skill so the
ledger/poll logic lives in one place:
```
/kaggle-submit <slug> --node node_0000 --message "node_0000 baseline cv=<cv> (<metric>)"
```
The kaggle-submit skill appends the UTC row to `submissions.md`, polls for the
public score, and logs the CV↔LB gap (surfaced, never auto-acted). When it
returns, in node.md set `lb: <public score>` and `submitted: <date -u +%F>` (a
submission is recorded by these fields — there is no `submitted` stage; `stage`
stays `decided`), and update the `lb` cell of the `graph.md` `## nodes` row; note
the public score + gap in `journal.md`. (If a 403 comes back, that's
rules-not-accepted / unverified, NOT bad creds — surface the human gate, don't
retry around it.)

## Step 8 — close the stage
Tick `baseline` in `comps/$slug/progress.md` and regenerate its derived header
(`today (UTC)=$(date -u +%F)`, `submissions=<used>/<lim>` where `lim` is spec.md's
`daily_submission_limit`, `deadline … days_left`).
Final readout to the human: champion = node_0000, local CV, public score (if
submitted) and the CV↔LB gap, and that the pipe is proven end-to-end — next is
`/kaggle-experiment` (real models).

## Guardrails
- Never advance `stage` before its named artifact exists (artifact-then-mark):
  `proposed → built → reviewed → decided` (a submission is recorded by the
  `submitted:` date + `lb:` fields, not a stage).
- A server-rejected submission does NOT burn the daily quota — safe to fix and
  resubmit; only an *accepted* submit counts.
- Do not add features, models, or tuning here — that is `/kaggle-experiment`.
  One atomic thing: the dumbest constant, proven end-to-end.
- If `solution.py` takes minutes (huge test set), run it backgrounded with the
  marker-file pattern (`DONE=/tmp/${slug}_node_0000.done`) per CLAUDE.md.
