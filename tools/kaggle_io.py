#!/usr/bin/env python3
"""Reusable Kaggle CLI wrapper — the one place that talks to Kaggle.

Thin, stateless, competition-agnostic. Orchestration lives in skills; this just
makes the kaggle CLI safe to drive headlessly: env-auth check, zip-aware
download, async-aware submit/poll, exponential backoff on 429, hard error
mapping (403 == rules-not-accepted, NOT bad creds), and a timestamp-derived
daily-budget reader.

The kaggle CLI is a runtime dependency: `uv add kaggle`. Auth via env:
    export KAGGLE_USERNAME=... KAGGLE_KEY=...

Usage:
    uv run tools/kaggle_io.py download <slug> --dest comps/<slug>/data
    uv run tools/kaggle_io.py submit <slug> --file sub.csv --message "node_7 cv=0.12"
    uv run tools/kaggle_io.py submissions <slug>
    uv run tools/kaggle_io.py leaderboard <slug>
    uv run tools/kaggle_io.py budget --ledger comps/<slug>/submissions.md \
        --limit <daily_submission_limit from spec.md — the single source of truth>
    uv run tools/kaggle_io.py classify-error --text "403 Forbidden"
    uv run tools/kaggle_io.py --selftest
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# --- error mapping ---------------------------------------------------------
_ERROR_PATTERNS = [
    ("rate_limited", r"\b429\b|too many requests|rate.?limit"),
    ("rules_not_accepted", r"\b403\b|forbidden|you must accept|competition rules|not.*participat"),
    ("auth", r"\b401\b|unauthorized|invalid.*credential|could not find kaggle\.json|KAGGLE_KEY"),
    ("not_found", r"\b404\b|not found|does not exist"),
]


def classify_error(text: str) -> str:
    """Map kaggle CLI stderr to an actionable category.

    403 maps to ``rules_not_accepted`` because that is what it almost always
    means in practice (the human hasn't accepted the comp rules / verified),
    not a credentials problem — the #1 misdiagnosis.
    """
    t = (text or "").lower()
    for label, pat in _ERROR_PATTERNS:
        if re.search(pat, t):
            return label
    return "unknown" if t.strip() else "ok"


_HUMAN_HINT = {
    "rules_not_accepted": (
        "403 — accept the competition rules in the browser (and phone-verify the "
        "account). This is NOT a credentials problem."
    ),
    "auth": "Set KAGGLE_USERNAME and KAGGLE_KEY in the env, then retry.",
    "rate_limited": "Rate limited (429) — backing off.",
    "not_found": "Competition/resource not found — check the slug.",
}


# --- auth + subprocess -----------------------------------------------------
def ensure_auth() -> None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return
    sys.exit(
        "kaggle auth missing: export KAGGLE_USERNAME and KAGGLE_KEY "
        "(or place ~/.kaggle/kaggle.json), then retry."
    )


def run_kaggle(args: list[str], *, retries: int = 5, _sleep=time.sleep) -> subprocess.CompletedProcess:
    """Run `kaggle <args>` with exponential backoff on rate limiting."""
    ensure_auth()
    delay = 2.0
    last = None
    for attempt in range(retries):
        # invoke via the module, not the `kaggle` entry-point script — the latter's
        # shebang can go stale if the venv is relocated; `python -m kaggle` always works.
        last = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)
        if last.returncode == 0:
            return last
        kind = classify_error(last.stderr or last.stdout)
        if kind == "rate_limited" and attempt < retries - 1:
            _sleep(delay)
            delay *= 2
            continue
        hint = _HUMAN_HINT.get(kind, "")
        sys.stderr.write((last.stderr or last.stdout or "") + ("\n" + hint if hint else "") + "\n")
        return last
    return last


# --- subcommands -----------------------------------------------------------
def cmd_download(slug: str, dest: str) -> int:
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    r = run_kaggle(["competitions", "download", "-c", slug, "-p", str(dest_p)])
    if r.returncode != 0:
        return r.returncode
    for z in dest_p.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest_p)
        z.unlink()
        print(f"unzipped {z.name}")
    print(f"data ready in {dest_p}")
    return 0


def cmd_submit(slug: str, file: str, message: str) -> int:
    r = run_kaggle(["competitions", "submit", "-c", slug, "-f", file, "-m", message])
    sys.stdout.write(r.stdout)
    return r.returncode


def cmd_passthrough(verb: list[str]) -> int:
    r = run_kaggle(verb)
    sys.stdout.write(r.stdout)
    return r.returncode


def read_budget(ledger: str, limit: int) -> dict:
    """Count today's (UTC) submissions from the append-only markdown ledger.

    A row counts as a submission if it starts with `| <YYYY-MM-DD`. The count is
    DERIVED, never stored, so it can't drift across a resume. `limit` comes from
    spec.md's `daily_submission_limit` — there is deliberately no default here.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    used = 0
    p = Path(ledger)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.startswith(f"| {today}"):
                used += 1
    return {"today": today, "used": used, "limit": limit, "remaining": max(0, limit - used)}


def cmd_budget(ledger: str, limit: int) -> int:
    b = read_budget(ledger, limit)
    print(f"{b['today']}  {b['used']}/{b['limit']} used  ({b['remaining']} remaining, resets 00:00 UTC)")
    return 0


# --- selftest --------------------------------------------------------------
def _selftest() -> int:
    assert classify_error("HTTP 429 Too Many Requests") == "rate_limited"
    assert classify_error("403 Forbidden") == "rules_not_accepted"
    assert classify_error("You must accept the competition rules") == "rules_not_accepted"
    assert classify_error("401 Unauthorized") == "auth"
    assert classify_error("Could not find kaggle.json") == "auth"
    assert classify_error("404 Not Found") == "not_found"
    assert classify_error("") == "ok"
    assert classify_error("some weird thing") == "unknown"

    # backoff retries on 429 then gives up, using an injected no-op sleep
    calls = {"n": 0}

    def fake_run(args, capture_output, text):
        calls["n"] += 1
        return subprocess.CompletedProcess(args, 1, "", "429 too many requests")

    import builtins  # noqa
    orig = subprocess.run
    subprocess.run = fake_run  # type: ignore
    os.environ.setdefault("KAGGLE_USERNAME", "x")
    os.environ.setdefault("KAGGLE_KEY", "y")
    try:
        slept = []
        run_kaggle(["competitions", "list"], retries=3, _sleep=lambda d: slept.append(d))
        assert calls["n"] == 3, calls
        assert slept == [2.0, 4.0], slept  # exp backoff, last attempt doesn't sleep
    finally:
        subprocess.run = orig  # type: ignore

    # budget derivation
    import tempfile
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with tempfile.TemporaryDirectory() as d:
        led = Path(d) / "submissions.md"
        led.write_text(
            f"| {today}T10:00Z | node_1 | 0.12 |\n"
            f"| {today}T11:00Z | node_2 | 0.11 |\n"
            f"| 2000-01-01T00:00Z | old | 0.99 |\n"
        )
        b = read_budget(str(led), 5)
        assert b["used"] == 2 and b["remaining"] == 3, b
    print("kaggle_io selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selftest", action="store_true")
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("download"); d.add_argument("slug"); d.add_argument("--dest", required=True)
    s = sub.add_parser("submit"); s.add_argument("slug"); s.add_argument("--file", required=True); s.add_argument("--message", required=True)
    sm = sub.add_parser("submissions"); sm.add_argument("slug")
    lb = sub.add_parser("leaderboard"); lb.add_argument("slug")
    bg = sub.add_parser("budget"); bg.add_argument("--ledger", required=True); bg.add_argument("--limit", type=int, required=True, help="daily_submission_limit from spec.md")
    ce = sub.add_parser("classify-error"); ce.add_argument("--text", required=True)

    a = p.parse_args(argv)
    if a.selftest:
        return _selftest()
    if a.cmd == "download":
        return cmd_download(a.slug, a.dest)
    if a.cmd == "submit":
        return cmd_submit(a.slug, a.file, a.message)
    if a.cmd == "submissions":
        return cmd_passthrough(["competitions", "submissions", "-c", a.slug, "-v"])
    if a.cmd == "leaderboard":
        return cmd_passthrough(["competitions", "leaderboard", "-c", a.slug, "-s"])
    if a.cmd == "budget":
        return cmd_budget(a.ledger, a.limit)
    if a.cmd == "classify-error":
        print(classify_error(a.text)); return 0
    p.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
