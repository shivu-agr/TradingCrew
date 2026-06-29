"""Run history — persist every UI run for replay and audit.

Each completed kickoff (whether short-circuited in CRISIS, full debate,
critic-revised, or error'd) is written to::

    $TRADINGCREW_CACHE_DIR/runs/{TICKER}/{ISO_TS}.json

Plus a single index line is appended to ``$TRADINGCREW_CACHE_DIR/runs/index.jsonl``
so the API can answer "most recent N runs across tickers" without
walking the directory tree.

The record is the union of every UI-relevant event the runner emits:
the run config, cascade status, final decision (post-critic), action
proposal, execution result, reflection records, and episode metadata.
This lets the UI re-render the *exact* state a previous run produced
without re-running it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _runs_dir() -> Path:
    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache_dir) / "runs"


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` to ``path`` atomically (write tmp + rename).

    Guards against partial writes when the process is killed mid-write —
    a half-written ``runs/AAPL/2026-06-06T...json`` would otherwise crash
    the next attempt to list run history.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


@dataclass
class RunRecord:
    """The full audit trail of one UI kickoff, as written to disk."""

    run_id: str
    ticker: str
    started_at: str
    completed_at: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    # M4 — regime + routing
    cascade_status: Optional[Dict[str, Any]] = None
    cascade_route: Optional[str] = None  # FULL_DEBATE / CRISIS_OVERRIDE / RISK_HEAVY

    # CrewAI — narrative outputs
    expected_role_order: List[str] = field(default_factory=list)
    reports: Dict[str, str] = field(default_factory=dict)
    # Roles whose final answer came back as the degraded placeholder
    # (see ``trading_crew/_patches.py``).  Persisted so the UI's Recent
    # Runs panel can flag the record on rehydrate, and so post-hoc
    # audits can spot which analysts came back empty without having
    # to grep the per-role report bodies for the marker string.
    degraded_roles: List[str] = field(default_factory=list)
    final_decision: Optional[Dict[str, Any]] = None
    final_decision_source: Optional[str] = None   # "crew" / "critic" / "cascade_override"

    # M4 — critic
    reflection_records: Optional[Dict[str, Any]] = None

    # M1 — typed proposal
    action_proposal: Optional[Dict[str, Any]] = None
    action_proposal_markdown: Optional[str] = None

    # M2 + M5 — pipeline
    execution_result: Optional[Dict[str, Any]] = None

    # M3 — episode
    episode_meta: Optional[Dict[str, Any]] = None

    # Counters / errors
    tool_calls: int = 0
    error: Optional[str] = None
    status: str = "running"  # running / completed / error

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_run(record: RunRecord) -> Path:
    """Write ``record`` to disk and append an index line.

    Returns the file path.  Never raises — failures are logged so a
    storage error never aborts the kickoff.
    """
    runs_dir = _runs_dir()
    file_path = runs_dir / record.ticker.upper() / f"{record.run_id}.json"
    try:
        _atomic_write_json(file_path, record.to_dict())
    except Exception as exc:
        logger.exception("Failed to write run record %s: %s", record.run_id, exc)
        return file_path

    # Append to the global index (best-effort — index is regenerable).
    index_path = runs_dir / "index.jsonl"
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "a") as f:
            f.write(json.dumps({
                "run_id": record.run_id,
                "ticker": record.ticker.upper(),
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "status": record.status,
                "final_action": (record.final_decision or {}).get("action"),
                "final_size_pct": (record.final_decision or {}).get("size_pct_of_book"),
                "cascade_route": record.cascade_route,
                "path": str(file_path),
            }, default=str) + "\n")
    except Exception:
        logger.exception("Failed to append to run index")
    return file_path


def list_recent_runs(limit: int = 20, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the most-recent ``limit`` run-index entries, newest first.

    When ``ticker`` is set, only entries for that ticker are returned.
    The index is append-only so we read the *whole* file in memory — at
    ~200 bytes per line this stays cheap up to ~50k runs.
    """
    index_path = _runs_dir() / "index.jsonl"
    if not index_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ticker and entry.get("ticker", "").upper() != ticker.upper():
                    continue
                entries.append(entry)
    except Exception:
        logger.exception("Failed to read run index")
        return []
    entries.sort(key=lambda e: e.get("started_at", ""), reverse=True)
    return entries[:limit]


def load_run(run_id: str, ticker: str) -> Optional[Dict[str, Any]]:
    """Load a full run record by id + ticker.  Returns None if missing."""
    file_path = _runs_dir() / ticker.upper() / f"{run_id}.json"
    if not file_path.exists():
        return None
    try:
        with open(file_path) as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load run %s", run_id)
        return None


def load_latest_run(ticker: str) -> Optional[Dict[str, Any]]:
    """Return the most recent full run record for ``ticker`` (or None)."""
    recent = list_recent_runs(limit=1, ticker=ticker)
    if not recent:
        return None
    entry = recent[0]
    return load_run(entry["run_id"], entry["ticker"])


def clear_runs(ticker: Optional[str] = None) -> Dict[str, Any]:
    """Delete saved run records and rewrite the index.

    When ``ticker`` is ``None`` we clear the *entire* runs history —
    every per-ticker subdirectory and the global index.  When ``ticker``
    is set we only drop records for that ticker and rewrite the index
    in place so other tickers stay intact.

    The deletion is mildly defensive:

    * We move the per-ticker directory aside to ``runs/.trash/{ticker}-{ts}``
      instead of immediately ``rm -rf``-ing it.  A user who hits the UI
      button by mistake can recover by renaming the trash folder back.
    * On a full clear we move ``runs/`` itself to ``runs.trash-{ts}/``.
      A new empty ``runs/`` is created so subsequent kickoffs still
      write cleanly.

    Returns ``{"removed": N, "trash_path": "..."}`` so the UI can show
    a confirmation toast (``N`` is the number of dropped index lines).
    """
    runs_dir = _runs_dir()
    if not runs_dir.exists():
        return {"removed": 0, "trash_path": None}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trash_root = runs_dir / ".trash"

    if ticker is None:
        # Move the whole runs/ tree aside and start fresh.  Counting the
        # index lines first gives the UI an accurate "removed N runs"
        # number without having to enumerate the dirs.
        index_path = runs_dir / "index.jsonl"
        removed = 0
        if index_path.exists():
            try:
                with open(index_path) as f:
                    removed = sum(1 for line in f if line.strip())
            except Exception:
                logger.exception("Failed to count index entries before clear")
        backup = runs_dir.parent / f"runs.trash-{timestamp}"
        try:
            os.rename(runs_dir, backup)
        except Exception:
            logger.exception("Failed to move %s aside; aborting clear", runs_dir)
            return {"removed": 0, "trash_path": None}
        # Recreate the empty runs directory so future writes work.
        runs_dir.mkdir(parents=True, exist_ok=True)
        return {"removed": removed, "trash_path": str(backup)}

    # Per-ticker clear: move that ticker's directory into runs/.trash/
    # and rewrite the index keeping only entries for other tickers.
    ticker_dir = runs_dir / ticker.upper()
    removed = 0
    trash_path: Optional[Path] = None
    if ticker_dir.exists():
        trash_root.mkdir(parents=True, exist_ok=True)
        trash_path = trash_root / f"{ticker.upper()}-{timestamp}"
        try:
            os.rename(ticker_dir, trash_path)
        except Exception:
            logger.exception("Failed to move %s aside; aborting clear", ticker_dir)
            return {"removed": 0, "trash_path": None}

    index_path = runs_dir / "index.jsonl"
    if index_path.exists():
        kept_lines: List[str] = []
        try:
            with open(index_path) as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        entry = json.loads(stripped)
                    except json.JSONDecodeError:
                        kept_lines.append(stripped)
                        continue
                    if entry.get("ticker", "").upper() == ticker.upper():
                        removed += 1
                        continue
                    kept_lines.append(stripped)
            # Atomic rewrite — same pattern as the per-run writer so a
            # crash mid-rewrite can't leave the index half-written.
            tmp = index_path.with_suffix(index_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                if kept_lines:
                    f.write("\n".join(kept_lines) + "\n")
            os.replace(tmp, index_path)
        except Exception:
            logger.exception("Failed to rewrite index after clear of %s", ticker)
    return {
        "removed": removed,
        "trash_path": str(trash_path) if trash_path else None,
    }
