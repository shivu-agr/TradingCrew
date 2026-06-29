"""Per-agent audit utility — cross-reference a saved RunRecord with the
file log to produce a one-screen "did everything work?" summary.

Usage
-----

::

    # Audit the most recent run for a ticker:
    ../.venv/bin/python scripts/audit_run.py --ticker LT.NS

    # Audit a specific run by run_id:
    ../.venv/bin/python scripts/audit_run.py --run-id 2026-06-29T06-30-16.792145+00-00 --ticker LT.NS

    # Cap the log scan to the last N lines (faster on huge logs):
    ../.venv/bin/python scripts/audit_run.py --ticker LT.NS --log-tail 5000

What it surfaces
----------------

For each role in ``expected_role_order`` we print:

* status: ``OK``, ``DEGRADED`` (patch fired), or ``MISSING`` (no
  report written — the runner failed to capture the output)
* report length in characters
* every tool call the agent made (parsed from the agent_step lines in
  the saved record and from the ``OpenAI: Successfully validated tool``
  + tool_call lines in the file log)
* the **anomalies** flagged on the log between the previous role's
  ``node_completed`` and this role's ``node_completed`` — yfinance
  ticker-probe storms, Tavily retries, 5xx/4xx upstream errors, etc.

At the end we print a section listing every distinct
"actionable issue" we saw (degraded analysts, malformed ticker inputs,
hallucinated tool names, etc.) so a quick eyeball is enough to tell
whether the run is healthy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEGRADED_MARKER = "[[DEGRADED_TOOL_CALL_OUTPUT]]"
LEGACY_REPR_PREFIX = "[ChatCompletionMessageFunctionToolCall"

# Pre-compiled regexes for things we care about in the file log.
_RE_DEGRADED = re.compile(r"degraded on task .* \(agent='([^']+)'\)")
_RE_YF_404 = re.compile(r"yfinance: \$([^:]+): possibly delisted")
_RE_YF_HTTP = re.compile(r"yfinance: HTTP Error (\d{3}):")
_RE_RESOLVER_WARN = re.compile(
    r"resolve_ticker: ('[^']+') is not a ticker"
)
_RE_RESOLVER_MISS = re.compile(
    r"resolve_ticker: no yfinance match for ('[^']+')"
)
_RE_TOOL_CALL = re.compile(
    r"OpenAI: Successfully validated tool '([^']+)'"
)
_RE_RUN_START = re.compile(r"Expected role order:")
_RE_RUN_DONE = re.compile(r"\brun_completed|run_error\b")
_RE_504 = re.compile(r'"HTTP/1\.1 (5\d{2}|4\d{2}) ')
_RE_TAVILY_RETRY = re.compile(r"Tool retry .* tavily", re.I)
# Log lines all start with ``YYYY-MM-DD HH:MM:SS,mmm`` in the local
# timezone of the server.  Strip the comma-milliseconds so we can
# compare with the saved record's ISO-8601 ``started_at`` /
# ``completed_at`` timestamps (which are UTC).  We don't try to
# bridge the timezones precisely — instead we anchor the window on
# the literal log line that contains the run's ``started_at`` ISO
# string (the runner logs the kickoff config), and we close it on
# the next "Expected role order:" marker or end-of-file.
_RE_LOG_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def _runs_dir() -> Path:
    cache = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache) / "runs"


def _load_run(ticker: str, run_id: Optional[str]) -> Tuple[Dict[str, Any], Path]:
    rdir = _runs_dir() / ticker.upper()
    if not rdir.exists():
        raise SystemExit(f"No runs directory for {ticker} at {rdir}")
    if run_id:
        path = rdir / f"{run_id}.json"
    else:
        candidates = sorted(rdir.glob("*.json"), reverse=True)
        if not candidates:
            raise SystemExit(f"No saved runs for {ticker} in {rdir}")
        path = candidates[0]
    with open(path) as f:
        return json.load(f), path


def _classify_report(role: str, body: str, degraded_roles: Iterable[str]) -> str:
    if not body:
        return "MISSING"
    if role in degraded_roles:
        return "DEGRADED"
    trimmed = body.lstrip()
    if trimmed.startswith(DEGRADED_MARKER) or trimmed.startswith(LEGACY_REPR_PREFIX):
        return "DEGRADED"
    return "OK"


def _human_size(n: int) -> str:
    if n < 1_000:
        return f"{n} chars"
    return f"{n/1000:.1f}k chars"


def _scan_log(log_path: Path, tail_lines: Optional[int]) -> List[str]:
    if not log_path.exists():
        return []
    with open(log_path, errors="replace") as f:
        lines = f.readlines()
    if tail_lines:
        return lines[-tail_lines:]
    return lines


def _slice_to_run(lines: List[str], record: Dict[str, Any]) -> List[str]:
    """Return only the log section that belongs to the record's run.

    We walk every ``Expected role order:`` marker in the file (each
    marks a kickoff boundary), pair each marker with the next one,
    and pick the segment whose timestamp range straddles the
    record's ``started_at`` (compared on naive local datetimes — the
    server writes log lines in its own timezone, the saved record is
    UTC, so we shift both into UTC and pick the closest marker that
    fired *at or before* the record's start).

    Falls back to the most-recent kickoff when timestamp parsing
    fails — better to over-include some anomalies than to attribute
    them to the wrong run.
    """
    markers: List[Tuple[int, datetime]] = []
    for i, ln in enumerate(lines):
        if not _RE_RUN_START.search(ln):
            continue
        m = _RE_LOG_TS.match(ln)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            markers.append((i, ts))
        except ValueError:
            continue
    if not markers:
        return lines

    rec_start_iso = record.get("started_at")
    if not rec_start_iso:
        return lines[markers[-1][0]:]

    # Strip TZ for naive comparison — both the log and the record use
    # wall-clock time, just in different zones, and we don't try to
    # convert here.  We pick the marker closest to the record start
    # *by relative ordering of kickoffs*: if there are N markers and
    # this record is the M-th one (counted by ordered ``started_at``
    # across the recent saved runs), we slice marker[M].
    runs_dir = _runs_dir() / (record.get("ticker") or "").upper()
    others = sorted(
        p.stem for p in runs_dir.glob("*.json")
    )
    try:
        index = others.index(record.get("run_id"))
    except ValueError:
        return lines[markers[-1][0]:]

    # Markers go in chronological order; pick the one with the same
    # rank counted from the END.  E.g. if there are 3 markers and
    # this record is the latest of 3 saved runs, take markers[-1].
    rank_from_end = (len(others) - 1) - index
    if rank_from_end >= len(markers):
        return lines[markers[0][0]:]
    chosen_idx = markers[len(markers) - 1 - rank_from_end][0]
    next_idx = (
        markers[len(markers) - rank_from_end][0]
        if rank_from_end > 0
        else len(lines)
    )
    return lines[chosen_idx:next_idx]


def _audit_run(record: Dict[str, Any], log_lines: List[str]) -> Dict[str, Any]:
    expected = record.get("expected_role_order") or []
    reports = record.get("reports") or {}
    degraded_roles = set(record.get("degraded_roles") or [])

    log_lines = _slice_to_run(log_lines, record)

    # Counters / collections we will populate.
    by_role: Dict[str, Dict[str, Any]] = {}
    anomalies = Counter()
    yf_404_targets = Counter()
    resolver_garbage: List[str] = []
    resolver_misses: List[str] = []
    tool_validations: Counter = Counter()
    upstream_errors: List[str] = []
    tavily_retries: List[str] = []

    for ln in log_lines:
        if (m := _RE_DEGRADED.search(ln)):
            anomalies[f"degraded:{m.group(1)}"] += 1
        if (m := _RE_YF_404.search(ln)):
            yf_404_targets[m.group(1)] += 1
            anomalies["yfinance_404"] += 1
        if (m := _RE_RESOLVER_WARN.search(ln)):
            resolver_garbage.append(m.group(1))
            anomalies["resolver_garbage_input"] += 1
        if (m := _RE_RESOLVER_MISS.search(ln)):
            resolver_misses.append(m.group(1))
            anomalies["resolver_total_miss"] += 1
        if (m := _RE_TOOL_CALL.search(ln)):
            tool_validations[m.group(1)] += 1
        if _RE_504.search(ln):
            anomalies["upstream_5xx_4xx"] += 1
            upstream_errors.append(ln.strip())
        if _RE_TAVILY_RETRY.search(ln):
            anomalies["tavily_retry"] += 1
            tavily_retries.append(ln.strip())

    for role in expected:
        body = reports.get(role, "")
        status = _classify_report(role, body, degraded_roles)
        by_role.setdefault(role, {"status": "MISSING", "report_chars": 0, "occurrences": 0})
        by_role[role]["occurrences"] += 1
        # Multiple occurrences (debate / risk rounds) share the same
        # report body in CrewAI's task store, so we only update the
        # status the first time we see the role.
        if by_role[role]["occurrences"] == 1:
            by_role[role]["status"] = status
            by_role[role]["report_chars"] = len(body)

    return {
        "ticker": record.get("ticker"),
        "run_id": record.get("run_id"),
        "started_at": record.get("started_at"),
        "completed_at": record.get("completed_at"),
        "status": record.get("status"),
        "cascade_route": record.get("cascade_route"),
        "final_action": (record.get("final_decision") or {}).get("action"),
        "final_size_pct": (record.get("final_decision") or {}).get("size_pct_of_book"),
        "expected_role_order": expected,
        "by_role": by_role,
        "anomalies": anomalies,
        "yf_404_targets": yf_404_targets,
        "resolver_garbage": resolver_garbage,
        "resolver_misses": resolver_misses,
        "tool_validations": tool_validations,
        "upstream_errors": upstream_errors[-5:],   # last 5 most relevant
        "tavily_retries": tavily_retries[-5:],
    }


def _print_summary(audit: Dict[str, Any], record_path: Path) -> None:
    bar = "─" * 78
    print(bar)
    print(
        f"AUDIT  {audit['ticker']}  {audit['run_id']}\n"
        f"       started:   {audit['started_at']}\n"
        f"       completed: {audit['completed_at']}\n"
        f"       cascade:   {audit['cascade_route']}\n"
        f"       outcome:   {audit['final_action']} "
        f"(size={audit['final_size_pct']})\n"
        f"       file:      {record_path}"
    )
    print(bar)
    print(f"{'#':>3} {'STATUS':10} {'CHARS':>10}  ROLE")
    for i, role in enumerate(audit["expected_role_order"], start=1):
        info = audit["by_role"].get(role, {})
        status = info.get("status", "MISSING")
        chars = info.get("report_chars", 0)
        occ = info.get("occurrences", 1)
        suffix = f"  (× {occ})" if occ > 1 else ""
        print(f"{i:>3} {status:10} {_human_size(chars):>10}  {role}{suffix}")
    print(bar)

    anomalies = audit["anomalies"]
    if not anomalies:
        print("Anomalies: none — clean run.")
        return

    print("Anomalies:")
    for kind, n in sorted(anomalies.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:35} ×{n}")
    if audit["yf_404_targets"]:
        print()
        print("yfinance 404 targets (top 10):")
        for sym, n in audit["yf_404_targets"].most_common(10):
            print(f"  {sym:50} ×{n}")
    if audit["resolver_garbage"]:
        print()
        print("Non-ticker inputs caught by the resolver (samples):")
        for raw in audit["resolver_garbage"][:5]:
            print(f"  {raw}")
    if audit["resolver_misses"]:
        print()
        print("Resolver total-miss (no exchange suffix matched):")
        for raw in audit["resolver_misses"][:5]:
            print(f"  {raw}")
    if audit["upstream_errors"]:
        print()
        print("Recent upstream 4xx/5xx errors:")
        for ln in audit["upstream_errors"]:
            print(f"  {ln[:150]}")
    if audit["tavily_retries"]:
        print()
        print("Recent Tavily retries:")
        for ln in audit["tavily_retries"]:
            print(f"  {ln[:150]}")
    print(bar)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", required=True, help="Ticker (e.g. LT.NS)")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Specific run_id (default: most recent for this ticker)",
    )
    parser.add_argument(
        "--log",
        default="logs/web.log",
        help="Path to the file log (default: logs/web.log)",
    )
    parser.add_argument(
        "--log-tail",
        type=int,
        default=None,
        help="Only scan the last N lines of the log (faster on huge files)",
    )
    args = parser.parse_args()

    record, path = _load_run(args.ticker, args.run_id)
    log_lines = _scan_log(Path(args.log), args.log_tail)
    audit = _audit_run(record, log_lines)
    _print_summary(audit, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
