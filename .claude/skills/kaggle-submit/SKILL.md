---
name: kaggle-submit
description: Budget-gated Kaggle submission with async public-score poll — computes today's budget from the UTC ledger against spec.md's daily_submission_limit, blocks past the limit, renders the SUBMIT card, submits, polls the score, then logs it to submissions.md + graph.md with the CV↔LB gap. Use when a valid node's CV beats the last-submitted CV by more than fold-noise (2·sem), or a stage reaches the `submit` gate.
argument-hint: <slug> <node_id>   e.g. titanic node_0007
allowed-tools: Bash, Read, Write, Edit
---

# /kaggle-submit — budget-gated submit + async poll

Spend ONE of the day's submission slots (limit = `daily_submission_limit` in
`spec.md` — the single source) on the single best-justified node, then poll for
its public score. **CV decides WHAT to submit; the LB is never an A/B target**
(Hard rule 6). A slot only goes to a node that beats the last-submitted CV by
more than fold-noise (2·sem — CLAUDE.md "Budget & deadline").

Resolve `<slug>` and `<node_id>` from args. If `<node_id>` is omitted, use the
current champion (`comps/<slug>/champion/`). All paths below are repo-relative.

---

## 0 · Preconditions (read, don't retry around the human gates)

- `comps/<slug>/nodes/<node_id>/submission.csv` exists, and its `node.md`
  frontmatter shows `leak: clean` and a non-null `cv` (i.e. it was built and
  scored). A node that hasn't cleared the leakage self-checks **cannot** be
  submitted — leakage voids the score (Hard rule 3). If `leak` isn't `clean` or
  `cv` is null, stop and say so.
- `KAGGLE_USERNAME` / `KAGGLE_KEY` are in the env (the tool fails with a clear
  message otherwise). A 403 here means **rules-not-accepted / unverified**, not
  bad creds — surface the human gate, don't retry.

Validate the file before spending anything on it (a malformed CSV wastes a slot):
```bash
slug=<slug>; node=<node_id>
ndir=comps/$slug/nodes/$node
id_col=$(grep -E '^id_col:' comps/$slug/spec.md | awk '{print $2}')   # from spec machine block
uv run tools/validate_submission.py \
  --submission $ndir/submission.csv \
  --sample     comps/$slug/data/sample_submission.csv \
  --id         "${id_col:-id}"
```
Non-zero exit ⇒ the CSV is malformed; fix the node, do **not** submit.

---

## 1 · Budget — derived from UTC timestamps, never a counter

```bash
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' comps/$slug/spec.md)
[ -n "$lim" ] || { echo "spec.md lacks daily_submission_limit — kaggle-start must ask the human; stop"; }
uv run tools/kaggle_io.py budget --ledger comps/$slug/submissions.md --limit "$lim"
# prints:  <YYYY-MM-DD>  <used>/<lim> used  (<remaining> remaining, resets 00:00 UTC)
```
- `remaining == 0` ⇒ **the daily limit is spent — block**. Print when the next
  slot frees (`00:00 UTC`) and stop. Do not call submit.
- The count is recomputed from the ledger every time, so it can't drift across a
  resume. Never store or trust a mutable counter.

---

## 2 · CV gate — only submit a node that beats the last submitted CV

The last submitted CV is the `cv` column of the **last row** of
`comps/$slug/submissions.md` (empty ledger ⇒ this is the first/baseline submit,
which always passes). **Fold-noise = 2·sem** of the candidate's CV (the `sem:`
field in its `node.md`) — the one canonical definition, same bar as the promote
gate (CLAUDE.md "Budget & deadline").

Submit only if, **in the official metric's improving direction**:
```
| candidate_cv − last_submitted_cv |  >  2·sem      (the fold-noise band)
```
If the candidate is within fold-noise of what's already on the LB, **do not
spend a slot** — it's an LB A/B, which Hard rule 6 forbids. Say so and stop.
(Allowed exceptions, noted explicitly in the card: end-of-comp final-ensemble
submits, and a **human-directed LB probe** — logged with `PROBE` in the ledger's
note column, kept to ~2/day.)

Also honor the **CV-too-good tripwire**: if this node's CV jumped implausibly vs
its parent, that's flagged for human eyes *before* a slot is spent — surface it
in the card rather than auto-submitting, regardless of autonomy mode.

---

## 3 · SUBMIT Decision Card (gated except `full_auto`)

The `submit` gate is human in `interactive` and `auto_except_submit`; only
`full_auto` proceeds without waiting (read the mode from `comps/$slug/config.md`).
This costs **1 of the daily limit**.

```
📋 submit
What's going on:   Node <node_id> (<one-line change>) beats the last submitted CV — spending a slot to see it on the public board.
Found / propose:   • candidate CV <cv> ± <sem> vs last submitted <last_cv> (Δ <delta>, > 2·sem fold-noise)
                   • <used>/<lim> used today, <remaining> remaining (resets 00:00 UTC)
                   • file validates against sample_submission; leakage self-checks clean
                   • <deadline> — <days_left> days left
Why:               CV cleared the fold-noise band; the LB is the OOD check, not the selector.
Cost:              ~1–2 min · no compute · 1 of the daily <lim> submissions
Your call:         [Approve] [Change something] [Skip] [Tell me more]
Autonomy: <mode> — <waiting | proceeding>
```
Compute `<days_left>` and every date from the shell (`date -u`), never memory:
```bash
deadline=$(grep -E '^deadline:' comps/$slug/spec.md | awk '{print $2}')
days_left=$(( ( $(date -u -d "$deadline" +%s) - $(date -u +%s) ) / 86400 ))
```
In `interactive` / `auto_except_submit`: **wait** here. Proceed only on approve
(or in `full_auto`).

---

## 4 · Submit (a server-rejected submit does NOT burn the quota)

```bash
msg="$node cv=$cv"
uv run tools/kaggle_io.py submit $slug --file $ndir/submission.csv --message "$msg"
```
- Exit 0 ⇒ accepted by the server, now scoring asynchronously → go poll.
- Non-zero ⇒ classify before reacting; a **server-rejected** submission is safe
  to resubmit (it didn't burn the slot):
  ```bash
  uv run tools/kaggle_io.py classify-error --text "<the stderr line>"
  ```
  `rules_not_accepted` (403) ⇒ surface the human browser/verify gate, stop.
  `rate_limited` (429) ⇒ already backed off by the tool; if still failing, wait
  and retry once. `auth` ⇒ env vars; stop. Only append a ledger row **after** an
  accepted submit (exit 0) — never on a rejected one.

---

## 5 · Poll the async public score (event-driven, no tight loop)

Scoring is async. Poll `submissions` with a marker-file waiter so you wake when a
public score appears, not on a timer — and never tight-poll (the tool backs off
429s, but you should still space reads):
```bash
DONE=/tmp/${slug}_${node}_scored.done ; rm -f "$DONE"
(
  for i in $(seq 1 20); do                       # ~ up to 10 min, 30s spacing
    out=$(uv run tools/kaggle_io.py submissions $slug)
    # newest row first; "complete" + a numeric publicScore means it finished
    echo "$out" | grep -iE 'complete' | grep -qE '[0-9]' && { echo "$out" > /tmp/${slug}_${node}_sub.txt; break; }
    sleep 30
  done
  touch "$DONE"
) &
# wait on [ -f "$DONE" ]; then read /tmp/${slug}_${node}_sub.txt for the public score
```
Pull the **public score** for *this* submission (match the `$msg` / newest row)
into `lb`. If still `pending` after the window, record `lb=pending` and note that
the row will be backfilled on the next poll — don't block the loop.

---

## 6 · Append the ledger row — EXACT format the budget reader counts

The `budget` subcommand counts a row iff it `startswith("| <today-UTC>")`.
Append **after** an accepted submit, with the timestamp from the shell. The 5th
column is a free-text `note` (empty for a normal submit; `PROBE` for a
human-directed LB probe; never jam notes into the `lb` cell):
```bash
ts=$(date -u +%FT%RZ)                 # e.g. 2026-06-05T14:07Z  (UTC, minute precision)
printf '| %s | %s | %s | %s | %s |\n' "$ts" "$node" "$cv" "$lb" "$note" >> comps/$slug/submissions.md
```
This produces exactly: `| <date -u +%FT%RZ> | node_NNNN | <cv> | <lb> | <note> |`.
If the ledger has no header yet, write it once first (header rows don't start
with `| <date>` so they're never miscounted):
```
| ts (UTC)            | node      | cv     | lb      | note |
|---------------------|-----------|--------|---------|------|
```

---

## 7 · Advance the stage, log the gap (artifact-then-mark, never auto-demote)

1. In `comps/$slug/nodes/$node/node.md` frontmatter, **only now** that the row
   exists (Hard rule 5 — artifact then mark), set:
   - `lb: <public score>` (or `lb: pending` if the poll window closed unscored)
   - `submitted: <date -u +%F>`
   (The `stage` ladder ends at `decided` — a submission is recorded by these two
   fields, not by a stage value.)
2. Update that node's row in `comps/$slug/graph.md` — its `lb` cell (and
   `status`, if this submit promoted it to `champion`). The Mermaid label keeps
   the node's `cv`; the table carries the `lb`.
3. Append one timestamped line to `comps/$slug/journal.md`:
   ```
   <date -u +%FT%RZ>  $node  submit  cv=$cv  lb=$lb  gap=$(cv−lb)  used=<used+1>/<lim>
   ```
4. **Log the CV↔LB gap as a diagnostic, never an auto-demote** (Hard rule 6). A
   large gap is something to *surface to the human* (and consider a one-off
   adversarial-validation diagnostic next round), not a reason to change the
   champion. The champion is decided by CV in `graph.md`; submitting does not
   re-rank it.

---

## Done — closing readout

State, in plain language: which node was submitted, its CV, the public score (or
`pending`), the CV↔LB gap, and how many of the day's slots remain. Point to the
ledger (`comps/$slug/submissions.md`) and the node dir for full detail. If the
budget was already exhausted, say so and when the next slot opens (00:00 UTC).
