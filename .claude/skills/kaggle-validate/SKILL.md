---
name: kaggle-validate
description: Stage 2 — freeze the CV scheme ONCE: pick the leak-correct split from spec.md, carve an inviolable holdout, write folds.json + validation.md. Use when `validation` is the next stage, or the human says "freeze the CV" / "set up validation".
argument-hint: <slug>
allowed-tools: Bash, Read, Write
---

# /kaggle-validate — freeze the CV, once and immutably

You are at the **validation gate**. The question this stage answers, in plain
language: **"does my private grading match how Kaggle grades?"** Get this wrong
and every CV number downstream is a lie. You freeze the scheme here and **never
re-split**. A later node that re-runs `make_folds.py` with a different seed (or
scheme) is **auto-rejected** — the split seed is part of the frozen contract,
not a tunable.

`<slug>` is the comp folder name. If `$ARGUMENTS` is empty, read the only dir
under `comps/` (or ask which one). Everything below uses `comps/<slug>/`.

## 0 · Preconditions (don't skip)
- `comps/<slug>/spec.md` and `comps/<slug>/eda.md` must exist and the `eda`
  stage box in `progress.md` must be ticked. If EDA isn't done, stop and tell
  the human to run `/kaggle-eda` first — you cannot pick a leak-correct scheme
  without knowing the data's group/time structure.
- `comps/<slug>/data/train.csv` must exist (downloaded + unzipped).
- Get the date once, from the shell (never from memory):
  ```bash
  NOW=$(date -u +%Y-%m-%dT%H:%MZ)
  ```

## 1 · Parse the spec machine block
`spec.md` ends with a fenced ```` ```yaml ```` machine block of key fields. Read
it and pull the values that drive the split. Canonical key names (what
kaggle-start writes):

| spec key | maps to flag | meaning |
|---|---|---|
| `target_col` | `--target` | label column (required for stratified) |
| `task_type` | `--task-type` | `classification*` ⇒ stratified, else kfold |
| `group_key` | `--group-key` | a unit that must not straddle folds (patient, image, store) |
| `time_col` | `--time-col` | ordering column ⇒ expanding-window timeseries |
| `metric` | — | the official metric — you justify the scheme against THIS |
| `id_col` | — | id column (for the holdout note + leakage awareness) |

```bash
sed -n '/^```yaml/,/^```/p' comps/<slug>/spec.md
```
If a key is absent in the spec, treat it as unset (don't invent a group/time
column). The tool's auto-pick priority is **timeseries > group > stratified >
kfold** — exactly the leak-correct order, so prefer letting it auto-pick rather
than forcing `--scheme` unless EDA found a reason the auto-pick is wrong (record
that reason in `validation.md`).

Decide `--n-splits` (default **5**; use 3 if a fold would be tiny, more only if
the data is large and folds stay well-populated) and `--seed` (default **42** —
**this seed controls ONLY the split**, nothing else).

## 2 · Run the freeze
Auto-pick (preferred — pass only the fields the spec set):
```bash
uv run tools/make_folds.py \
  --train comps/<slug>/data/train.csv \
  --out   comps/<slug>/folds.json \
  --target <target> --task-type <task_type> \
  --group-key <group_key> --time-col <time_col> \
  --n-splits 5 --seed 42
```
Omit `--group-key` / `--time-col` / `--target` when the spec didn't set them.
Add `--scheme <kfold|stratified|group|timeseries>` ONLY to override the
auto-pick, and only with a written reason. The tool writes
`folds.json = {scheme, n_splits, seed, n_rows, folds:[{fold, val_idx:[...]}]}`
with positional indices into the (for timeseries, time-sorted) train frame.

## 3 · Carve the inviolable holdout
The holdout is a slice of train **never touched in training or feature-fit** — a
final honesty check distinct from the CV folds. Derive it deterministically from
the frozen folds so it's reproducible, and **record its exact indices in
`validation.md`** so any node can assert-exclude them.

- **timeseries:** the holdout is the **tail** — take the val indices of the
  **last** fold (the most-future block). Never random — that would leak future
  into the rest.
- **group:** the holdout is **all rows of the groups in the last fold's val
  set** — whole groups, never partial, or it leaks across the group boundary.
- **stratified / kfold:** the holdout is the **last fold's val indices** (a
  clean ~1/n_splits stratified or random slice).

```bash
uv run python - <<'PY'
import json, pathlib
slug = "<slug>"
f = json.loads(pathlib.Path(f"comps/{slug}/folds.json").read_text())
hold = sorted(f["folds"][-1]["val_idx"])
print(f"scheme={f['scheme']} n_splits={f['n_splits']} n_rows={f['n_rows']}")
print(f"holdout = last fold ({len(hold)} rows = {len(hold)/f['n_rows']:.1%})")
print("first/last idx:", hold[0], hold[-1])
PY
```
The remaining `n_splits-1` folds are the working CV. Be explicit in
`validation.md` that the holdout rows are excluded from every `.fit(` —
encoders, scalers, target-encoders, feature stats, model fit. (This is enforced
later by the developer's fast leakage self-checks — the `kaggle-leakage` skill;
here you just record the contract.)

## 4 · Write `validation.md` (the WHY, in plain language)
Explain, for a smart non-specialist, **why this scheme reproduces the official
grade**. Cover, in prose:
- **official metric** (from spec) and that local CV is scored with the *same*
  metric and the *same* direction — your private grading mirrors Kaggle's.
- **why this split** is leak-correct for THIS data: timeseries because train is
  past and test is future (no peeking ahead); group because a `<group_key>`
  appearing in both train and val would let the model memorise the unit; else
  stratified to keep class balance per fold; else plain kfold.
- **the holdout**: which rows (range/count/%), how derived, and the standing
  rule that it is never seen during fit/feature-engineering.
- **the freeze**: scheme + n_splits + seed are fixed; `folds.json` is the
  contract. **Re-splitting with another seed/scheme is auto-rejected** — note
  this explicitly.
- **any override**: if you passed `--scheme`, justify it against what EDA found.

Template:
```markdown
# Validation — comps/<slug>  (frozen <NOW>)
official metric: <metric>  (direction: <maximize|minimize>)

## Scheme
<scheme> · <n_splits> splits · seed 42 · n_rows <N> → folds.json

Why it matches the official grade: <plain-language paragraph tying split to
metric + data structure; cite the EDA finding that forced group/time if any>.

## Holdout (inviolable — never fit on)
- rows: last fold's val set — <count> rows (<pct>%), idx <first>…<last>
- derivation: <tail / whole-groups / last-fold slice>
- rule: excluded from every .fit() and all feature statistics, forever.

## Frozen contract
scheme + n_splits + seed are immutable. folds.json is the source of truth.
A node that re-splits with a different seed or scheme is auto-rejected.
```

## 5 · Tick the box and render the Decision Card
Artifact-then-tick: only after `folds.json` and `validation.md` both exist, tick
the `validation` stage box in `comps/<slug>/progress.md` and append one
timestamped line to `journal.md`:
```bash
echo "| $NOW | validation | froze <scheme> ${NSPLITS}-fold seed=42, holdout=<count> rows | folds.json validation.md |" \
  >> comps/<slug>/journal.md
```
Then render `📋 validation` in the **CLAUDE.md Decision Card format** (this is a
**human gate** outside `full_auto`); stage-specific content:
- *What's going on*: I froze how I'll grade myself locally before training anything.
- *Found / propose* bullets: scheme + n_splits + seed (split-only) · the
  inviolable holdout (count, %, never fit on) · local CV uses the official
  metric · re-splitting later is auto-rejected (frozen contract).
- *Why*: so my private CV tracks Kaggle's private grade — no shake-up.
- *Cost*: seconds · no compute · 0 submissions.
- Stage-specific responses to honor besides the standard buttons:
  `[Change scheme]` and `[Change n_splits/seed]`.
In `interactive` / `auto_except_submit`, **wait** for approval (validation is a
gate). In `full_auto`, proceed to `/kaggle-baseline`. If the human changes the
scheme/seed, re-run step 2 once and re-freeze — that's allowed only **here**,
before the freeze is final; after this stage the seed is locked.

## Guardrails
- `uv run` for everything; dates from `date -u` only.
- Never random-split a timeseries comp; never let a group straddle folds.
- Don't fabricate a group/time column the spec/EDA didn't establish.
- If `make_folds.py` errors (e.g. stratified without `--target`), read the
  message, fix the flag, re-run — don't hand-roll a split.
