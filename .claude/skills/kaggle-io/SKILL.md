---
name: kaggle-io
description: Reference for tools/kaggle_io.py — auth, the two one-time human gates, and every subcommand (download/submit/submissions/leaderboard/budget/classify-error). Use to download data, make or poll a submission, read the leaderboard, check the budget, or decode a Kaggle error (especially a 403). Every Kaggle call goes through this wrapper.
allowed-tools: Bash, Read
---

# kaggle-io — the one door to Kaggle

`tools/kaggle_io.py` is the **only** place that talks to Kaggle. Never call the
raw `kaggle` CLI and never hit the Kaggle HTTP API directly — those skip the
auth check, the zip-unpack, the 429 backoff, and the error mapping, and they
make failures non-uniform. Every skill (start, eda, baseline, submit, status)
routes through `uv run tools/kaggle_io.py …`.

This skill is a **reference**: it does not run a stage. Read it when you need a
copy-paste command or have to explain a Kaggle failure to the human.

---

## 1. Auth — set env BEFORE any call

The kaggle client authenticates **at import**, so the env must be set in the
same shell that runs the tool. Set both, then call:

```bash
export KAGGLE_USERNAME="$KAGGLE_USERNAME" KAGGLE_KEY="$KAGGLE_KEY"   # from the human's secrets
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' comps/<slug>/spec.md)
uv run tools/kaggle_io.py budget --ledger comps/<slug>/submissions.md --limit "$lim"
```

- The tool's `ensure_auth()` accepts **either** `KAGGLE_USERNAME`+`KAGGLE_KEY`
  in the env **or** a `~/.kaggle/kaggle.json`. Env vars are preferred — they
  also dodge the chmod-600 warning. If neither is present the tool exits with a
  clear message; that is an **auth** problem, not a rules problem.
- Never print the key. Never write it into any `comps/` file or the journal.

---

## 2. Two one-time HUMAN gates (non-automatable — surface, don't retry)

These cannot be scripted. If you hit them, **stop and put them in a Decision
Card** for the human — do not loop trying to work around them:

1. **Accept the competition rules** in the browser on the comp's *Rules* page.
2. **Phone-verify** the Kaggle account (required for GPU / internet on kernels,
   and for some comps to download or submit at all).

A 403 on download or submit almost always means one of these two is undone —
**not** bad credentials (see the error map below).

---

## 3. Subcommands (copy-paste)

Slugs are the competition slug from the URL, e.g.
`https://www.kaggle.com/competitions/titanic` → `titanic`.

### download — pull + unzip competition data
Downloads are **zipped**; the tool unzips every `*.zip` into `--dest` and
deletes the archive. Re-running is safe.
```bash
uv run tools/kaggle_io.py download <slug> --dest comps/<slug>/data
```
On success prints `unzipped …` per archive then `data ready in comps/<slug>/data`.
`comps/<slug>/data/` is gitignored — never commit it.

### submit — make a submission
```bash
uv run tools/kaggle_io.py submit <slug> \
  --file comps/<slug>/champion/submission.csv \
  --message "node_0007 cv=0.1234"
```
Submission scoring is **async** — `submit` only enqueues. Poll with
`submissions` for the public score. Budget-gate **before** calling this (see
`/kaggle-submit`): the per-comp daily limit comes from spec.md's
`daily_submission_limit` (asked from the human at kaggle-start), resets 00:00 UTC.
A *server-rejected* submission does **not** burn quota — safe to fix and resubmit.
After a real submit, append a UTC-timestamped row to `comps/<slug>/submissions.md`
(5 columns: `| ts | node | cv | lb | note |`).

### submissions — list past submissions + their scores (poll here)
```bash
uv run tools/kaggle_io.py submissions <slug>
```
Maps to `kaggle competitions submissions -c <slug> -v`. Use this to poll for the
public score after a submit; don't tight-loop — re-run on a sane interval.

### leaderboard — top of the public LB
```bash
uv run tools/kaggle_io.py leaderboard <slug>
```
Maps to `kaggle competitions leaderboard -c <slug> -s`. The public LB is a small
noisy slice — a CV↔LB gap is a *diagnostic to surface*, never an auto-demote
trigger (CLAUDE.md rule 6).

### budget — today's used/remaining, DERIVED from the ledger
`--limit` is **required** (no default) — the limit comes from spec.md's
`daily_submission_limit` (asked from the human at kaggle-start), never a literal:
```bash
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' comps/<slug>/spec.md)
uv run tools/kaggle_io.py budget --ledger comps/<slug>/submissions.md --limit "$lim"
```
Counts rows in `submissions.md` whose UTC date == today (today via the tool's
own `datetime.now(timezone.utc)`, matching the `date -u` rule). The count is
**computed at read time**, never stored, so it can't drift across a resume.
Prints e.g. `2026-06-05  2/<lim> used  (<lim>−2 remaining, resets 00:00 UTC)`. A
row counts only if it starts with `| <YYYY-MM-DD` — keep the ledger in that
5-column format (`| ts | node | cv | lb | note |`).

### classify-error — decode a Kaggle error string
```bash
uv run tools/kaggle_io.py classify-error --text "403 Forbidden"   # -> rules_not_accepted
```
Use it to turn raw stderr into an actionable category before you tell the human
anything. Pipe a captured stderr in via `--text "$err"`.

---

## 4. Error map (the #1 misdiagnosis is the 403)

`classify-error` (and `run_kaggle`'s inline mapping) bucket stderr into:

| category | trigger | what it actually means / do |
|---|---|---|
| `rules_not_accepted` | **403**, forbidden, "must accept", "competition rules" | **NOT bad creds.** Human must accept rules + phone-verify (the two gates above). Surface in a card; don't retry. |
| `rate_limited` | **429**, too many requests | Handled internally: exponential backoff (2s, 4s, …, up to 5 tries). Never tight-poll yourself. |
| `auth` | **401**, unauthorized, missing kaggle.json / `KAGGLE_KEY` | Set `KAGGLE_USERNAME` + `KAGGLE_KEY` in the env, then retry. |
| `not_found` | **404**, not found | Wrong/typo'd slug — check the URL. |
| `ok` / `unknown` | empty / unmatched | `ok` = empty text; `unknown` = real text that matched nothing — read it and decide. |

Key point to repeat to the human when it comes up: **403 means rules-not-accepted
/ unverified, not bad credentials.** Spending time regenerating the API key on a
403 is wasted — the fix is the browser gates.

429 is the only category the tool retries automatically. Everything else returns
the error (with a one-line human hint appended to stderr) for you to act on.

---

## 5. Self-test (offline; no network, no auth)

Verifies the error map, the 429 backoff schedule, and the budget derivation
without touching Kaggle:
```bash
uv run tools/kaggle_io.py --selftest   # prints "kaggle_io selftest OK"
```
Run it after any edit to `tools/kaggle_io.py`, or as a quick sanity check that
the wrapper is wired correctly before a stage depends on it.

---

## Quick recall
- One door: `uv run tools/kaggle_io.py …` — never the raw CLI/API.
- Env first: `KAGGLE_USERNAME` + `KAGGLE_KEY` before any call.
- Two human gates: accept rules + phone-verify — surface, don't retry.
- 403 = rules/verification, **not** creds. 429 = handled. 401 = creds.
- Downloads unzip themselves. Submits are async — poll `submissions`.
- Budget is derived from `submissions.md`, never a stored counter.
