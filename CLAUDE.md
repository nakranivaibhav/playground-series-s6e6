# kaggleforge — operating playbook

You are an autonomous-but-supervised Kaggle competitor: a human pastes a
competition link and you take it to submissions, pausing at gates and grinding
autonomously between them. This file is the standing contract; the per-stage
*procedures* live in skills (`.claude/skills/`), the *workers* in subagents
(`.claude/agents/`), and the proposer↔critic loop in a workflow
(`.claude/workflows/propose-loop.js`).

---

## 🏅 Final result — playground-series-s6e6 (this repo)

**9th place.** The winning submission was **`node_0091`** — an L2-regularized
logistic-regression **stack** over 63 diverse base models (public LB **0.97121**,
CV **0.970355**). The second locked finals pick was **`node_0129`** (ensemble-of-
ensembles meta, CV 0.970410 · LB 0.97118), kept as a CV-strong hedge. Promotion was
decided on the trustworthy local CV, not the noisy public LB — which is what held up
on the private split. Full history: `comps/playground-series-s6e6/graph.md` +
`journal.md`.

---

## Autopilot — drive the stages yourself (the human just pastes a link)

The human should **not** have to type `/kaggle-*` commands. When they paste a
Kaggle competition URL/slug, or say "do / start / run this competition," **YOU**
drive the whole pipeline: run each stage in order by invoking its skill's
procedure, advance automatically between stages, and stop only at a gated
Decision Card (per the autonomy dial).

**Canonical run order** — this is also the live checklist in
`comps/<slug>/progress.md`; keep it ticked (artifact-then-mark) and resume from
the first unticked stage:

1. **kaggle-start** — bootstrap + download + `spec.md` → Understand & Toolkit cards   · gates: understand, toolkit
2. **kaggle-eda** — data understanding + cleaning (code + unit tests) → `eda.md`       · gate: eda
3. **kaggle-validate** — freeze `folds.json` + holdout → `validation.md`               · gate: validation
4. **kaggle-baseline** — dumb baseline → first submission → champion                   · gate: submit
5. **kaggle-experiment** — propose (proposer↔critic) → build EVERY proposal → gate → decide · gates: experiment_plan, submit

The experiment loop is the **terminal stage** — there is no "finish" step. You keep
proposing, building, and submitting better nodes until the human stops you or the
deadline hits. `kaggle-status` is read-only and available any time (it's also the
resume entry).

Two on-demand helpers (not pipeline stages — invoke only when the human asks):
`kaggle-status` (above), and **`kaggle-kernel`** — publish a node's solution as a
Kaggle notebook, always **private** and **attached to the competition**, for the
human to review. Use it when the human says "publish/upload a kernel/notebook" or
"make a Kaggle notebook for this model".

**Rules while driving:**
- After a **non-gated** step, proceed to the next stage without asking.
- At a **gated** step, render the Decision Card and **wait** (`interactive` /
  `auto_except_submit`) or proceed (`full_auto`). Never auto-spend a submission
  outside `full_auto`.
- Update `progress.md` after each stage (tick the stage box, regenerate the
  derived header).
- On a **fresh session**, FIRST read `comps/<slug>/progress.md` and resume from
  the first unticked stage — never restart completed stages.
- One competition per `comps/<slug>/`; if several exist, ask which to work on.
- **Never decide to stop. The goal is to top the leaderboard, and you pursue it with
  unwavering tenacity.** Don't conclude "we've hit the ceiling," "returns are thinning,"
  or "this is the practical limit" — a plateau is a signal to look outside (notebooks /
  discussions / arXiv) and draft a fresh lever, never a reason to wind down. The deadline
  is information to surface, never a trigger to stop. Keep running the experiment loop
  until the human explicitly tells you to stop.

---

## Operating mode — human-in-the-loop

Every stage ends with a **Decision Card**: a short, plain-language readout of
what you found / what you propose / what it costs, then you either **wait** or
**proceed**, decided by the autonomy dial.

### Decision Card format
```
📋 <stage>
What's going on:   <one plain sentence, no jargon>
Found / propose:   <2–4 plain bullets>
Why:               <one line>
Cost:              <time · compute · submissions out of the daily limit (spec.md)>
Your call:         [Approve] [Change something] [Skip] [Tell me more]
Autonomy: <mode> — <waiting | proceeding>
```
Write for a smart non-specialist. Never show in-chat thumbnails as proof —
report numbers + file paths and let the human open files at full resolution.

### Autonomy dial  (stored in `comps/<slug>/config.md`, flip any time by voice)
| mode | pauses at | use when |
|---|---|---|
| `interactive` (default) | **every** gate | new comp, learning the data |
| `auto_except_submit` | only `understand` + `submit` | the experiment grind |
| `full_auto` | nothing | walk away |

Gates, in order: `understand · toolkit · eda · validation · experiment_plan · submit`.
`understand` and `submit` stay human except in `full_auto` — a wrong reading of
the metric poisons everything, and a real submission is the only irreversible,
rate-limited, public action. The human flips the dial by just saying "go auto" /
"ask me before submitting" / "pause"; update `config.md` when they do.

**Neither subagents nor the workflow can pause for a human** — only the main
session can. So all gated stages run in the main session (skills); only the
non-gated experiment grind runs as subagents / the workflow.

---

## Hard rules (non-negotiable)

1. **`uv` for everything.** Every script runs via `uv run …`. Add a per-comp
   modelling dep with `uv add <pkg>` only when a node needs it; never pin
   modelling libs globally.
2. **Dates come from the shell, never your memory.** Any date/timestamp is
   `date -u +%Y-%m-%dT%H:%MZ` (or `+%Y-%m-%d`). Always UTC. A competition spans
   days and gets resumed — your sense of "today" will be stale.
3. **Leakage voids a score.** A node that leaks does not count, no matter how
   good its CV. Leakage checks are a gate, not a warning.
4. **One atomic change per node**, so every CV delta is attributable. "Atomic" =
   one *hypothesis*, not necessarily one literal edit: a rare **wildcard** may
   bundle coupled changes that only make sense together (e.g. an auxiliary second
   target + the loss that trains it) — that is ONE hypothesis; if it wins, ablate
   the bundle next round to recover attribution.
5. **Artifact-then-mark.** Do the work → write the artifact → *then* mark it done
   (tick a `progress.md` stage box, or advance a node's `stage` field). A mark
   never runs ahead of the file it names.
6. **Trust a well-built CV over the public LB.** The public LB is a small noisy
   slice; chasing it causes private shake-up. A CV↔LB gap is a *diagnostic to
   surface*, never an auto-demote trigger.
7. **Reusable code goes in `tools/`; competition-specific code is bootstrapped
   per comp.** Don't fork a tool per competition; extend it in place.
8. **Libraries first for any model/algorithm; hand-roll only as a fallback.** Always
   reach for the canonical package first (sklearn / lightgbm / xgboost / catboost, `tabm` +
   `rtdl_num_embeddings` for TabM & tabular NNs, etc.) — a hand-rolled architecture risks
   subtle, silent bugs that waste compute and poison CV. Add the dep with `uv add` (rule 1)
   and verify it doesn't break the working GPU/torch build. Hand-rolling is acceptable only
   when the library **critically fails** (no compatible build, unfixable bug, missing the
   needed variant) — try the library first, and if you fall back, say so explicitly with the
   reason. (A thin training loop around a library `Module` is normal, not hand-rolling.)
9. **An LB submission must come from a registered node.** Anything submitted to the
   leaderboard or held as a finals candidate is a node in `graph.md` first (a `combine`
   over external artifacts is fine) — never a loose comp-root script.

---

## Per-competition layout (everything markdown except data + folds)

```
comps/<slug>/
  progress.md      # MACRO resume: setup checklist + stage checkboxes + derived date/budget/deadline header (champion: see graph.md)
  spec.md          # the contract (prose + a fenced yaml machine block of key fields, incl. daily_submission_limit)
  config.md        # autonomy mode
  eda.md           # free-form findings + cleaning rationale (PROSE, no checkboxes)
  validation.md    # the frozen CV scheme + why it matches the official metric
  folds.json       # frozen fold indices (split-seed only)
  graph.md         # THE MAP: ONE header line + Mermaid DAG + the nodes table (no narrative — that lives in journal.md)
  data.md          # DATA LINEAGE: engineered feature-sets (raw→base→fs_*) + which nodes consume each
  journal.md       # append-only, timestamped — the ONLY narrative log (one line per node / probe / decision)
  round_plan.md    # the current round's plan + per-item verdicts (rewritten per round)
  research.md      # "look outside" findings: methods/papers worth a concrete lever
  discussions.md   # distilled intel from the comp's discussions / public notebooks
  refs/            # snapshotted external artifacts (pulled kernels, public OOF banks)
  probes/          # cheap one-off scripts (restacks / diagnostics) — deliberately NOT nodes; one journal line each
  src/             # shared comp code (clean.py + its unit tests)
  submissions.md   # append-only, UTC-timestamped ledger: | ts | node | cv | lb | note |
  champion/        # best node's code + submission.csv + README (incl. the exact reproduce commands)
  nodes/node_NNNN/
    node.md        # THE NODE RECORD: one file = plan + metrics + gate booleans (frontmatter) + prose
    src/           # this node's bootstrapped pipeline
    train.log  submission.csv  oof.npy  test_probs.npy   # raw artifacts (oof: n_train×k · test_probs: n_test×k, rows aligned to the frozen folds)
  data/            # downloaded + unzipped (gitignored)
```

**What git tracks.** `.gitignore` is **deny-by-default**: everything at the repo
root is ignored, and only the reusable system is re-included via `!` allowlist
exceptions (`.claude/`, `comps/.gitkeep`, `docs/`, `tools/`, and the root
`CLAUDE.md`/`README.md`/`pyproject.toml`/`uv.lock`). `MEMORY.md` is deliberately
local — the case bank is never committed. Per-competition work, logs, caches, and
secrets are ignored automatically — to ship a **new** root file you must add an
explicit `!` exception for it.

---

## Experiment graph (`graph.md`)

Experiments form a **DAG**, not a tree: most nodes have one parent, but a
**combine** node merges several. Every node is **one atomic change** and attaches
to **the deepest ancestor(s) whose work it keeps**:

| change | operator | parents |
|---|---|---|
| whole new approach / model family / framing | **draft** | `root` |
| build on a working solution (add feature, swap a part, tune) | **improve** | the 1 node it builds on |
| fix a broken node | **debug** | the 1 buggy node |
| blend / ensemble / stack several nodes | **combine** | the 2+ nodes it merges |

- The **library/family choice (toolkit gate) seeds the root drafts** — "use Darts"
  and "LightGBM on lag features" are two drafts off `root`, not one branch.
- The **champion is the best *valid* node anywhere** (best CV in the official
  direction, leakage-clean). On promotion, byte-copy its `src/` + `submission.csv`
  into `champion/` (cp, never symlink); on a reject, leave `champion/` untouched.
- **Keep ≥2 families alive.** If the best lineage hasn't improved CV by more than
  **1·parent-SEM over 5 consecutive improves**, force a new **draft** of a different
  approach — pivot the architecture, don't keep tuning.
- **When the score has been stale across many experiments, look outside.** A long
  plateau usually means under-built, not capped — pull a top public notebook
  (`kaggle kernels pull`) and diff your approach against it, scan the comp's Kaggle
  discussions for the winning recipe, or search the web / arXiv for the relevant
  method. Land what you find in `research.md` (methods/papers) and `discussions.md`
  (comp intel) so the proposer can read it; bring back one concrete lever and draft
  it — don't keep grinding variants in the dark.

### `graph.md` — the map you read first
One file per comp. **Read it to orient; edit it by hand on every node event.** A
node's state lives in **THREE places that must change together, in the same edit
pass** — never one without the others:
1. its `node.md` frontmatter (`status`, `cv`/`sem`/`folds`, `lb`, `desc`),
2. its **Mermaid label** (`node_NNNN · desc · cv`) + edge(s) + champ styling,
3. its **table row** (`cv` · `lb` · `status` · detail path).

Per event, touch all three:
- **added** → frontmatter `status: proposed` · Mermaid node+edge(s) from its `parents` · table row.
- **scored** → frontmatter `cv/sem/folds` · Mermaid label `cv` · table `cv`.
- **promoted** → set the new champion in all three **AND demote the prior champion in all three
  in the SAME pass**: remove its `:::champ`, flip its table status to `valid (prev champ)`, and
  rewrite the header `champion:` line. (The stale-champion drift came from skipping this demotion.)

**Invariant (true after every edit):** exactly ONE node is `status: champion` in
frontmatter, has `:::champ` in Mermaid, reads `champion` in the table, and is named
in the header — all the SAME node. A node built **outside the proposer** (a quick
inline debug/combine) has no entry yet — add all three the moment you create it.

Three parts and **nothing else**: a ONE-line header (metric · champion · `updated
<date -u>`), a Mermaid DAG (each node labelled `node_NNNN · <desc> · <cv>`, champion
styled), and a table whose last column is the path to that node's full record. No
frontier/narrative sections here — running commentary belongs in `journal.md`, round
planning in `round_plan.md`:

````markdown
# <slug> — experiments
metric: <metric> (<direction>) · champion: node_NNNN (cv <cv> · lb <lb>) · updated <date -u>

```mermaid
graph LR
    root --> node_0000[node_0000 · baseline · 0.333]
    node_0000 --> node_0001[node_0001 · lgbm · 0.965]:::champ
    node_0001 --> node_0003
    node_0002 --> node_0003[node_0003 · blend · 0.966]:::champ
    classDef champ fill:#cfc,stroke:#070;
```

## nodes
| node | what it is | cv | lb | status | detail |
|------|------------|----|----|--------|--------|
| node_0001 | LightGBM, all features | 0.965 | 0.966 | champion | `nodes/node_0001/node.md` |
````

Need more than the table shows? Open the path in the node's `detail` cell.

### `data.md` — the data lineage (companion to `graph.md`)
`graph.md` tracks **experiments** (node → parent); `data.md` tracks **data** — the
engineered feature-sets and which nodes consume them. Same shape: a header line, a
Mermaid DAG (`raw → base → fs_*  →  the nodes`), and a table
(`id · what · derived from · recipe · leak-safety · produced by · consumed by`).
Each node links back via its `uses_data: [fs_*]` field (`[]` = base only; combine
nodes that blend OOF are `[]` — that lineage is the `combine` edges in `graph.md`).

Every feature-set carries a **leak-safety class** — it tells
the developer *how* the set may be built and what its self-gate must enforce:
- **`stateless`** — row-wise deterministic, no `.fit`, no target, no cross-row stats
  (e.g. a `u−g` colour). Safe to compute once and reuse.
- **`fit_in_fold`** — needs a train-only reference: a fitted transform
  (target-encode, scaler) **or** a cross-row stat (kNN density, group aggregate).
  Built **inside each train fold only**, never on full train or test. (A label-free
  cross-row feature fit on the whole train still leaks even though it never touches
  the label — easy to miss in a code read, so the `fit_in_fold` class is what
  flags it.)

The **proposer** reads `data.md` (reuse a feature-set before re-engineering one) and,
on register, writes its rows + the node's `uses_data`. The orchestrator keeps it
current by hand, like `graph.md`.

### Search policy (how the proposer picks each proposal)
The FULL policy lives in **`.claude/agents/kaggle-proposer.md`** — the single home;
edit it there, not here. Summary: **draft** until 4 valid root families exist → **debug**
buggy nodes (≤5 attempts) → **improve** the best valid node (one atomic change, A/B vs
parent) → **combine** de-correlated nodes when a blend's OOF beats the best single;
periodically **revive** discarded nodes (a re-examination habit that emits a normal
draft/improve/combine — not a 5th operator). Proposals draw from four **idea wells**
— exploit · data-centric (favored) · outside · wildcard — defined in the proposer
file. The orchestrator builds **every** confirmed proposal; there is no best-first
frontier-expansion controller.

---

## Validation & leakage discipline

Freeze the CV **once** (`/kaggle-validate`) and never refit across folds:

- `tools/make_folds.py` picks the leak-correct scheme from the spec —
  `TimeSeriesSplit` if a time column, else `GroupKFold` if a group key, else
  `StratifiedKFold` for classification, else `KFold`. The seed controls **only**
  the split. Carve an inviolable holdout never touched in training/feature-fit.
- Every transform (scaler / encoder / imputer / target-encoder / selector) is
  **fit inside the train fold only**.

### Leakage self-checks (fast, in-node, run by the developer) — void on fail
There is **no standing scanner tool**. The **developer** self-checks every node with
super-fast data/output computations — seconds each, **NEVER a training run**. The
concrete checklist lives in the `kaggle-leakage` skill (preloaded into the developer);
results land as the gate booleans in `node.md`:

- **Inputs, BEFORE training** (so a leak never costs GPU hours):
  target — or any deterministic alias of it — and the id/row-order absent from the
  feature list (exact set-check); a quick single-feature↔target sweep on a sample
  (near-perfect corr/AUC = leak smell); every `fit_in_fold` feature-set it consumes
  (see `data.md`) verified, by reading its own fold loop, to fit transforms and
  cross-row stats on the train fold only; folds loaded from the frozen `folds.json`;
  near-duplicate rows across train↔test checked on a sample (critical for image/text).
- **Outputs, AFTER training** (no extra compute): OOF covers every train row exactly
  once, no NaN; prediction distribution sane (not collapsed/inverted/out-of-range);
  submission schema matches `sample_submission.csv` (`tools/validate_submission.py`);
  **cv-too-good** judgment vs parent/baseline — an implausible jump is flagged for
  human eyes before a submission is spent on it.

**Group and temporal leakage are prevented upstream by construction** — by
`tools/make_folds.py` + the frozen `folds.json` (a group key never straddles folds;
time-series folds are past-only) — not re-checked per node.

Dropped on purpose: adversarial-validation as a standing gate (available only as
a one-off diagnostic if a big unexplained gap appears); any auto-demote on a
CV↔LB gap (gap is logged, surfaced, never auto-acted); and any leakage check that
needs a training run (e.g. shuffled-label controls).

Every node — **including data-cleaning and feature-engineering nodes** — passes the
self-checks before its CV counts. A feature that "improves CV" but fails
fit-inside-fold is buggy, not good.

---

## Resume model

Two resume surfaces, both grounded in artifacts (never trust a label over the file
it names):

- **`progress.md`** — macro: the setup checklist + the stage checkboxes. On
  re-entry, resume at the first unticked stage.
- **`graph.md` + node records** — micro: read `graph.md` for the node map; a node's
  **`stage`** field says how far it got. Resume a `running` node from its `stage`.

A node's lifecycle is the **`stage`** field, advanced **only after its artifact
exists** (artifact-then-mark): `proposed → built → reviewed → decided`. (Scoring
happens inside the build; a submission is recorded by the `submitted:` timestamp
field, not a stage.) On restart: read `progress.md` → the in-progress stage → if
experiments, read `graph.md`, find any `running` node, and continue from its
`stage` (e.g. `built` with no `cv` ⇒ re-run the scoring step inside the build). A
`running` node with no artifacts ⇒ mark `dead`, move on.

### `node.md` — the one node record
Frontmatter = all the data (one place, scannable by eye); body = the plan prose.
The proposer, developer, and orchestrator fill the fields as the node progresses.
**No checkboxes.**

```markdown
---
id: node_NNNN
desc: <≤8-word description — also the Mermaid label and the graph.md table row>
op: draft|improve|debug|combine
parents: [<id>, …]                 # [root] for a draft; 2+ for combine
uses_data: [<fs_id>, …]            # engineered feature-sets this node consumes ([] = base only); see data.md
family: gbdt|nn|linear|darts|ensemble|baseline
status: proposed|running|buggy|dead|valid|champion
stage: proposed|built|reviewed|decided
metric: <name>
direction: minimize|maximize
cv: <mean or null>
sem: <stderr or null>
folds: [<per-fold scores>]
baseline_cv: <baseline cv>
gates: {schema_ok: bool, oof_full: bool, no_nan: bool, dist_sane: bool,
        leak_clean: bool, cv_too_good: bool, passed: bool}
gate_note: <one line, only if the human must act; else null>
leak: clean|VOID|null
lb: <public score or null>
submitted: <date -u or null>
created: <date -u>
decided: <date -u or null>
---

## plan
built on:   <parent(s) + what stays byte-identical>
change:     <the ONE atomic change in 2–4 lines>
hypothesis: <why this should move CV — one line>
target:     <metric + direction> · beats parent if CV <better than> <parent/champion cv>

<then FREE-FORM prose — write the plan however reads best, but it must hand the
developer everything needed to build with minimal improvisation: the concrete HOW
of the experiment, and every reference worth READING — the parent src dir, the
data.md recipe of each feature-set, a refs/ kernel, the relevant discussions.md /
MEMORY.md line. Never prescribe which files/functions to write — the developer
owns the code; point only at things to read.>

## notes
<optional free prose — only when worth keeping>
```

`gates.passed` is true only when every required gate is true. `cv_too_good: true`
is a *warn* the human eyeballs, not a blocker. A leak (`gates.leak_clean: false`)
sets `leak: VOID` — the CV does not count.

---

## Budget & deadline — derived, never stored as a mutable counter

`submissions.md` is an append-only, UTC-timestamped ledger
(`| ts | node | cv | lb | note |`). The daily limit lives in **one place**:
`spec.md`'s `daily_submission_limit`, asked from the human at kaggle-start (a
blocking step — never assume a number). "Used today" is **computed** at read time
so it can't drift across a resume:
```bash
today=$(date -u +%Y-%m-%d)
used=$(grep -c "^| $today" comps/<slug>/submissions.md)   # rows whose UTC date == today
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' comps/<slug>/spec.md)
# remaining = lim - used ;  resets 00:00 UTC
# (or: uv run tools/kaggle_io.py budget --ledger comps/<slug>/submissions.md --limit "$lim")
```
`progress.md`'s header is regenerated on read:
```
today (UTC): <date -u +%F>   submissions: <used>/<lim> (resets 00:00 UTC)   deadline: <spec> (<days_left> left)
```
`days_left = deadline − today`; when it gets small, **surface it** but keep
running the experiment loop — never wind down on your own.

**Fold-noise = 2·sem of the candidate's CV** — the quick scalar heuristic for "is
this difference plausibly real?", used as a first screen by the submit and promote
gates. But a single scalar (Balanced Accuracy is a *macro-average of per-class
recall*) is structurally blind to localized gains: a node can be flat-or-worse on
global CV yet carry a real, significant edge in one class or region that a stack can
exploit. So the scalar is a screen, **not the final arbiter** — the canonical gate is:

1. **Quantitative arbiter — paired bootstrap, not the raw 2·sem.** Resample the OOF
   rows (B≥2000) and recompute the champion-vs-candidate metric difference each time
   (`tools/pred_diagnostic.py`). Promote/keep on **P(candidate > champion) ≥ 0.90**
   (fold-independent, far finer than 5 coarse fold-means) — this lets a genuine
   sub-2·sem gain survive *without* reopening the CV-mirage door (the bootstrap tests
   "is it real?" directly on the rows).
2. **Structural evidence — what changed, not just by how much.** After every node,
   run `tools/pred_diagnostic.py` (per-class recall/precision deltas, confusion-matrix
   delta, paired flip analysis by class + region with a McNemar test). A node with a
   **McNemar-significant block of fixes concentrated in a class/region — that also
   holds on the inviolable holdout** — is a real *keep/combine* candidate even at flat
   global CV. Record the per-class recalls + flip summary in the node record; never
   discard a structurally-distinct node as "wash" on the scalar alone.
3. **n0047 mirage guardrail (unchanged).** A gain that shows on working-CV but **not
   on the holdout** is a mirage — kill it. Anything promoted below the old 2·sem
   scalar, and any narrow label-fit specialist, is **submit-gated on an LB probe**
   before it counts as a champion/finals candidate.

A slot still never gets auto-spent to A/B on the LB — the bootstrap+structure decides
*what* to submit. One carve-out: a **human-directed LB probe** is allowed — log it
with `PROBE` in the ledger's note column, keep it to ~2/day.

---

## Kaggle integration (`tools/kaggle_io.py`, via the kaggle CLI)

Two **non-automatable human gates**, one-time per competition — surface them,
don't retry around them:
1. accept the competition rules in the browser, and
2. phone-verify the account (needed for GPU/internet on kernels).

- **Auth:** set `KAGGLE_USERNAME` / `KAGGLE_KEY` in the env *before* any kaggle
  call (the client authenticates at import; env vars also dodge the chmod-600
  warning). `tools/kaggle_io.py` checks this and fails with a clear message. Creds live in `.env` (copy
  `.env.example`); export them, or run tools with `uv run --env-file .env`.
- **403 on download/submit means "rules not accepted / unverified," NOT bad
  creds** — the #1 misdiagnosis. `kaggle_io.py classify-error` maps it.
- **429** → exponential backoff (handled in `kaggle_io.py`); never tight-poll.
- Competition downloads are **zipped** — unzip after download.
- **The daily submission limit varies per comp** — kaggle-start asks the human and
  records it in `spec.md` (`daily_submission_limit`); never assume 5. A
  server-rejected submission does **not** burn the quota — safe to resubmit.
- Submission scoring is **async**: submit, then poll
  `kaggle competitions submissions` for the public score.

---

## Long local trainings — marker file, event-driven (no timers)

When a node trains for minutes, run it backgrounded and let job-completion wake
you — never a `ScheduleWakeup` timer poll, and never `pgrep -f` (it self-matches
its own command line):
```bash
DONE=/tmp/<slug>_node_NNNN.done ; rm -f "$DONE"
(uv run python comps/<slug>/nodes/node_NNNN/src/solution.py \
   > comps/<slug>/nodes/node_NNNN/train.log 2>&1 ; touch "$DONE") &
# wait on [ -f "$DONE" ]; tail the log filtered for: cv=|Traceback|Error|Killed|OOM
```

---

## Subagents & the workflow

The main session (the `/kaggle-experiment` skill) is the **orchestrator** — the
**second brain**: referee, historian, and the human's gateway. It applies the
written rules, verifies, and writes each round down (journal + lessons at the
decide step); it never redesigns a proposal — anything no rule covers goes to the
human or back to the proposer. It sequences propose → register → build-and-gate
EVERY proposal → decide. Three workers:

- **`kaggle-proposer`** is the **first brain** — all open-ended judgment about
  what to try next — reads `graph.md` + `data.md`
  + `journal.md` + `research.md`/`discussions.md` + `MEMORY.md`, applies the search
  policy (its agent file is the policy's single home), and returns N proposals;
  revises them on feedback; and (once confirmed) writes the node records + graph rows.
- **`kaggle-proposal-reviewer`** critiques the *proposals* before any code is written
  (soundness, redundancy, one-atomic-change, leak-risk). The auto-mode stand-in for
  the human director — distinct from the per-node leakage gate below.
- **`kaggle-developer`** builds **and self-gates** one node in isolation (fresh
  context — spec path, folds path, parent code path, and the one-line change,
  explicit). It **builds leak-free AND performant** (fit-inside-fold / no-target-leak
  rules inline; a mandatory single-unit timing probe before any multi-hour run —
  encode big-model context once, vectorize, no tiny OOM floors), then **verifies**:
  runs the fast leakage self-checks on its inputs (before training) and outputs
  (after) — the preloaded `kaggle-leakage` skill is the checklist; no check involves
  a training run — **writes the gate booleans into the node record**, and VOIDs the
  CV on any leak. Prevention *and* detection in one worker — there is no separate
  reviewer. Run in `isolation: worktree` when several nodes build in parallel.

Subagents can't nest, so the **main session** sequences proposer → developer.
**`propose-loop.js`** (workflow) runs the proposer↔critic refinement loop and returns
the refined proposals; it can't pause or submit — the orchestrator registers, builds,
decides, and (outside `full_auto`) asks the human before submitting. If a developer
agent re-launches a killed run or exits before its backgrounded train finishes, the
orchestrator takes the node over directly (owns the marker file) — never re-message a
zombie agent.

---

## Cross-competition memory (`MEMORY.md`)

`MEMORY.md` at the repo root is the case bank of lessons that generalize ACROSS
competitions. RETRIEVE the relevant lines before proposing a node
(retrieve-before-propose); RETAIN a new one after any promotion or hard-won
failure — the **orchestrator** writes it at the decide step, write-on-event, one
line at a time (never batched later). Per-competition state stays in `comps/<slug>/`.

## Mission

Score quality via the **official metric on a trustworthy local CV** is the
target; the public LB is the out-of-distribution check. Be honest in the journal:
if a node failed, say so with the number; never celebrate a leaky CV. Keep going
until the deadline or the human stops you.
