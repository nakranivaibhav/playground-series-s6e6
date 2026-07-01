---
name: kaggle-eda
description: Stage 1 — interactive EDA + cleaning: probe the data, write findings to eda.md, turn each cleaning decision into reusable code + a unit test (optionally fan out kaggle-eda-explorer for parallel angles). Use when `eda` is the next unticked stage, or the human says "do EDA" / "look at the data" / "clean the data".
argument-hint: <slug>   (the comp folder under comps/, e.g. titanic)
allowed-tools: Bash, Read, Write, Edit
---

# kaggle-eda — understand the data, then make cleaning auditable

Stage 1 of the flow (`understand · toolkit · **eda** · validation · …`). You are
in the **main session** because this stage is gated — only the main session can
render a Decision Card and pause. Read CLAUDE.md for the standing contract
(autonomy dial, Decision Card format, leakage discipline, dates-from-`date -u`,
uv-only). Output is `comps/<slug>/eda.md` (free-form prose) plus reusable
cleaning code with unit tests. **No checkboxes in eda.md** — a cleaning decision
is proven by its test passing, not by a tick.

`$SLUG` below = the argument (default to the only folder under `comps/` if one).
`$C = comps/$SLUG`.

---

## 0. Resume + preconditions
1. `today=$(date -u +%F)`. Read `$C/progress.md`; confirm `eda` is the first
   unticked stage. If `understand`/`toolkit` are unticked, stop and tell the
   human to run `/kaggle-start` first.
2. Read `$C/spec.md` — especially its fenced ```yaml machine block. Pull:
   `task_type`, `metric` (+ `metric_direction`), `target_col`, `id_col`, and any
   `time_col` / `group_key` the spec flagged. These are the leakage-hazard
   candidates EDA must confirm or refute, and they feed `tools/make_folds.py` at
   the next stage.
3. Confirm data is present: `ls $C/data/`. If only zips, unzip first
   (`uv run python -c "import zipfile,glob,sys; [zipfile.ZipFile(z).extractall('$C/data') for z in glob.glob('$C/data/*.zip')]"`).
   If no data at all, the `toolkit` stage didn't download it — stop and say so.

## 1. First-pass inspection (small snippets, you in the loop)
Run short `uv run python` one-liners — never a saved script for this. After each,
read the output and decide the next probe. Cover at minimum:

```bash
# shape, dtypes, head, memory
uv run python -c "import pandas as pd; d=pd.read_csv('$C/data/train.csv'); print(d.shape); print(d.dtypes); print(d.head()); print(d.memory_usage(deep=True).sum()//1e6,'MB')"
# missingness per column (count + pct), sorted
uv run python -c "import pandas as pd; d=pd.read_csv('$C/data/train.csv'); m=d.isna().mean().sort_values(ascending=False); print((m[m>0]*100).round(2))"
# target distribution (classification: value_counts+balance; regression: describe+skew)
uv run python -c "import pandas as pd; d=pd.read_csv('$C/data/train.csv'); y=d['<target>']; print(y.value_counts(dropna=False)); print(y.describe()); print('skew',y.skew())"
# cardinality of object/low-card cols (categorical vs free-text vs id-like)
uv run python -c "import pandas as pd; d=pd.read_csv('$C/data/train.csv'); print(d.select_dtypes('object').nunique().sort_values())"
# train vs test column diff (target should be the only train-only col)
uv run python -c "import pandas as pd; a=set(pd.read_csv('$C/data/train.csv',nrows=1).columns); b=set(pd.read_csv('$C/data/test.csv',nrows=1).columns); print('train-only',a-b,'| test-only',b-a)"
```

Leakage-hazard probes (these decide the CV scheme later — surface, don't fix
here):
- **id leakage**: is `id_col` monotonic / correlated with target order? Does the
  id encode a group or time (`d['<id>'].astype(str).str[:N]`)? An id used as a
  feature is a hard error at the leakage gate.
- **time leakage**: any date/timestamp col? Does train's time range precede
  test's? If so the spec's `time_col` is real → TimeSeriesSplit next stage.
- **group leakage**: any key (patient/user/session/image-group) with repeats
  across rows? `d['<key>'].value_counts().head()`. If a group can straddle
  folds, next stage needs GroupKFold.
- **train↔test duplicates**: near-dup rows on features (critical for image/text);
  this is a leakage-suite check too.
- **CV-too-good smell**: any single feature whose correlation/AUC with the target
  is implausibly high → note it; the leakage suite will gate it.

Use the largest table only via `nrows=` sampling if it's big enough to be slow.
For non-CSV data (images/parquet/multiple tables) adapt the reader; keep snippets
tiny and printed.

## 2. (Optional) parallel angles — fan out kaggle-eda-explorer
For a wide/messy dataset, dispatch the **kaggle-eda-explorer** subagent in
parallel, one per angle, then synthesize. Give each the spec path, data dir, and
its single angle:
- **missingness** — pattern of NaNs (MCAR vs structural), cols to drop vs impute;
- **distributions** — skew/outliers/transform candidates per numeric col;
- **target-relationship** — which features move the target (mutual info / simple
  group-means), without fitting on full data;
- **leakage-hazards** — id/time/group/duplicate audit as above.

Subagents run read-only-ish and **cannot pause for the human**; you collect their
returns and reconcile conflicts. If the data is small/simple, skip the fan-out
and do it inline — don't spend a subagent on a 10-column CSV.

## 3. Write `$C/eda.md` — free-form prose, no checkboxes
Dense, plain findings a non-specialist can follow (CLAUDE.md voice). Suggested
shape (prose, not a form):
- **Shape & schema** — rows/cols train vs test, dtypes worth noting.
- **Target** — distribution, class balance / skew, implication for the metric.
- **Missingness** — which cols, how much, MCAR vs structural, drop-or-impute call.
- **Distributions & outliers** — transform candidates, clip/winsorize calls.
- **Leakage hazards** — the id/time/group/duplicate verdict, and the **CV scheme
  it implies** (hand this to `/kaggle-validate`). State each as a hazard +
  decision, e.g. "`signup_date` precedes test → TimeSeriesSplit; do not shuffle."
- **Proposed cleaning/feature steps** — a numbered list; each line is one atomic,
  testable transform (this list becomes nodes/code, below).
Lead with a 3–5 line summary the human can read in 20 seconds.

## 4. Turn each cleaning step into code + a unit test
A cleaning decision is only "done" when it's **code with a passing test** — the
test is the evidence, which is why eda.md stays prose. Put helpers in the comp's
**shared, reusable** module — `$C/src/clean.py` — not forked per node (CLAUDE.md
rule 7: extend in place). Each helper:
- takes a DataFrame, returns a transformed copy;
- is **stateless or fit-callable so it can fit inside a fold** — anything that
  learns from data (imputer fill value, encoder map, scaler stats) exposes a
  `fit(train_df)` → params and `transform(df, params)` split, so the modelling
  node fits on the train fold only. **Never** compute a global statistic over
  train+test or over all rows.

Write tests in `$C/src/test_clean.py`. At minimum, per helper:
- **row-count preserved** (unless the step's job is to drop rows — then assert the
  exact expected drop);
- **no new NaNs** introduced (or NaNs only where intended);
- **train & test transformed identically** — same columns out, same dtypes;
- **fit-inside-fold** — `fit` on a slice then `transform` a disjoint slice never
  reads the disjoint slice's stats (assert the params come only from the fit
  slice).

Run them via uv (pytest is invoked through uv; add it once if missing):

```bash
uv run pytest $C/src/test_clean.py -q   # add dep once: uv add --dev pytest
```

Do **not** tick anything until the tests pass (CLAUDE.md rule 5, artifact-then-
tick). If a step can't be made to pass fit-inside-fold, it's a feature for the
modelling node's fold loop, not a global clean — note that in eda.md and don't
apply it globally.

## 5. Gate: the EDA Decision Card
Render the card in the CLAUDE.md Decision Card format, then obey the autonomy dial
in `$C/config.md` (`interactive` waits here; `auto_except_submit`/`full_auto`
proceed). Stage-specific content:
- **stage:** eda
- **What's going on:** Looked at the data and wrote down what needs cleaning.
- **Found / propose:** <3–4 plain bullets: target balance, top missing cols, the
  one leakage hazard that sets the CV scheme, # cleaning steps coded+tested>
- **Why:** Clean, leak-free inputs before we freeze the CV split.
- **Cost:** <minutes> · CPU only · 0 submissions

On approve/proceed: tick the `eda` stage box in `$C/progress.md` (only now), and
append one UTC-stamped line to `$C/journal.md`:
`$(date -u +%FT%RZ) eda: <N> cleaning steps coded+tested; CV hazard=<time|group|none>; → validate`.
The next stage is `/kaggle-validate`, which reads your leakage verdict to pick the
fold scheme via `tools/make_folds.py`.

---

### Guardrails
- **No model fitting here** — EDA describes; the modelling node fits. The only
  "fits" allowed are inside a `fit/transform` helper that the node calls
  fold-locally.
- **Snippets print, scripts persist.** Probes are throwaway one-liners; the only
  files you create are `eda.md`, `src/clean.py`, `src/test_clean.py`.
- **uv for everything; dates from `date -u`.** No bare `python`, no typed dates.
- **Surface leakage, don't silently fix it** — the verdict drives the next gate.
