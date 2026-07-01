---
name: kaggle-status
description: Read-only readout of where a competition stands, and the resume entry point. Use when the user asks "where are we / status / what's next", or on any session restart to rebuild state. Never trains, submits, or edits — only reads and reports.
argument-hint: "[slug]  (omit to use the only comp under comps/, or the most recently touched)"
allowed-tools: Bash, Read
---

# /kaggle-status — read-only readout + resume entry point

You report where one competition stands, in plain language, and (because this is
the resume path) you end by stating **exactly where to resume**. You change
**nothing**: no training, no submitting, no editing `progress.md` / `graph.md` /
`node.md`. Regenerating the header below is a *report you print to chat*, not a
file write.

## 0 · Resolve the slug
- If `$ARGUMENTS` names a slug, use `comps/<slug>/`.
- Else list `comps/*/` (skip `.gitkeep`). One comp → use it. Several → pick the
  most recently modified (`ls -dt comps/*/ | head -1`) and say which you chose.
- If `comps/<slug>/progress.md` is missing, stop and report: "No comp bootstrapped
  yet — run `/kaggle-start <url>`." Do not invent state.

Set `C=comps/<slug>` for the steps below.

## 1 · Regenerate the derived header (print, don't write)
Everything dated is derived live from the shell so a resume can't read stale state.

```bash
today=$(date -u +%F)
# budget: DERIVED from the append-only ledger; limit from spec.md (the single source)
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' "$C/spec.md")
uv run tools/kaggle_io.py budget --ledger "$C/submissions.md" --limit "${lim:?spec.md lacks daily_submission_limit — kaggle-start must ask the human}"
# cross-check the count the same way the contract defines it:
used=$(grep -c "^| $today" "$C/submissions.md" 2>/dev/null || echo 0)
```
- Read the **deadline** from `$C/spec.md` — it lives in the fenced yaml machine
  block (a `deadline:` field, ISO date). Grep it out; if absent, report "deadline: n/a".
- Compute days left with the shell (never by hand), guarding a missing deadline:
```bash
dl=$(grep -oiE 'deadline:[[:space:]]*([0-9]{4}-[0-9]{2}-[0-9]{2})' "$C/spec.md" \
      | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}' | head -1)
if [ -n "$dl" ]; then
  days_left=$(( ( $(date -u -d "$dl" +%s) - $(date -u -d "$today" +%s) ) / 86400 ))
fi
```
- Print the header exactly in the contract's shape:
```
today (UTC): <today>   submissions: <used>/<lim> (resets 00:00 UTC)   deadline: <dl|n/a> (<days_left|?> left)
```
If `days_left` is small (≤ ~2), add one line: "Deadline near." Surface it as
information only; never wind down on your own — keep running experiments to climb
the leaderboard until the human explicitly stops you.

## 2 · Current stage (from progress.md)
Read `$C/progress.md`. The stage checkboxes are the macro sequence
(`understand · toolkit · eda · validate · baseline · experiment`; the *gate*
sequence in CLAUDE.md is a separate vocabulary). Report:
- the **first unticked** stage = where the comp currently is;
- the autonomy mode from `$C/config.md` (`interactive` / `auto_except_submit` /
  `full_auto`) and what it pauses at.

## 3 · The graph (only if at/after the experiment stage)
Read `$C/graph.md` (skip if it doesn't exist yet — pre-experiment comp). The
`## nodes` table carries one row per node (`status` column); the header line names
the champion. Report:
- **counts by status**: `proposed · running · buggy · valid · champion · dead`
  (one tally line);
- the **champion node**: its id, its CV (with the official metric name + direction
  from `validation.md`/`spec.md`), and its **public LB** (the row's `lb` cell, or
  `submissions.md` if newer). If CV and LB diverge, state the gap as a *diagnostic
  to surface*, never an auto-demote — per the trust-CV rule;
- the search frontier in one line: how many `valid` roots (families) are alive,
  and whether any `running`/`buggy` nodes are open.

## 4 · Recent journal
`tail -n 6 "$C/journal.md"` (append-only, one timestamped line per node). Echo
those lines verbatim under "Recent activity" — they're the densest history.

## 5 · Resume pointer (this is the entry point)
Follow the resume model end-to-end and state **one concrete next action**:
1. From `progress.md`, take the first unticked stage.
2. If that stage is **before** experiments → next action is "run `/<that-stage's
   skill>`" (e.g. unchecked `validation` → `/kaggle-validate`).
3. If at the **experiment** stage → read `graph.md` to rebuild the frontier, then:
   - If a node is `running`: open `$C/nodes/<id>/node.md`, read its `stage` field
     (`proposed → built → reviewed → decided`), and say "resume node `<id>` at:
     `<next stage after `stage`>` → `<the artifact that stage produces>`" (e.g.
     `built` with null `cv` ⇒ re-run the scoring step inside the build). If that
     node's `stage` is past `proposed` **but no artifacts are on disk**, say it's
     stale → "mark `<id>` dead and pick the next operator" (don't resume a ghost).
   - If nothing is `running`: apply the search policy (single home:
     `.claude/agents/kaggle-proposer.md`) to name the next operator; state which
     and why in a sentence.
4. If `submissions: <lim>/<lim>` for today, add: "submission budget spent — resets
   00:00 UTC; CV work can continue, no submit until reset."

Verify a node's `stage` against the artifacts it implies before trusting it
(artifact-then-mark): a `stage` past the file that proves it is a lie — report the
mismatch instead of believing it.

## 6 · The readout (print this, then stop)
No Decision Card, no gate, no waiting — `/kaggle-status` is read-only and always
just reports. Use plain language for a smart non-specialist; give **file paths**,
not in-chat thumbnails.

```
📍 <slug> — status
<header line from §1>
Stage:       <current stage> · autonomy <mode>
Graph:       <N> nodes — <proposed p · running r · buggy b · valid v · champion 1 · dead d>
Champion:    <node id> · CV <metric>=<val> (<dir>) · LB <lb|not scored>  → champion/
Recent:      (last journal lines)
  <line>
  <line>
Next:        <the one concrete resume action from §5>
```

Keep it tight. End after printing — do not proceed into the named next stage; the
human (or the autonomy dial in the relevant skill) drives that.
