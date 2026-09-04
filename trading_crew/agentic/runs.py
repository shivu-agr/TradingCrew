"""Run history — persist every UI run for replay and audit.

Each completed kickoff (whether short-circuited in CRISIS, full debate,
critic-revised, or error'd) is stored as one row in the ``runs`` table of the
shared ``prediction.db`` (SQLite) — a header JSON plus indexed columns for
the recent-runs panel. Agent report bodies live in ``run_reports`` and are
loaded when a tile is opened.  This replaces the former
``$TRADINGCREW_CACHE_DIR/runs/{TICKER}/{run_id}.json`` files + ``index.jsonl``.

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
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LEGACY_RUNS_MIGRATED = False


def _legacy_runs_dir() -> Path:
    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache_dir) / "runs"


def _migrate_legacy_runs() -> None:
    """Import legacy ``runs/{TICKER}/{run_id}.json`` files into the DB once,
    then move the tree aside so it's no longer read."""
    global _LEGACY_RUNS_MIGRATED
    if _LEGACY_RUNS_MIGRATED:
        return
    _LEGACY_RUNS_MIGRATED = True
    runs_dir = _legacy_runs_dir()
    if not runs_dir.is_dir():
        return
    try:
        from web.backend import prediction_db as db

        count = 0
        for path in runs_dir.glob("*/*.json"):
            if path.name.endswith(".tmp"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — skip a corrupt record
                continue
            if isinstance(data, dict) and data.get("run_id") and data.get("ticker"):
                db.upsert_run(data)
                count += 1
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        runs_dir.rename(runs_dir.parent / f"runs.migrated-{ts}")
        if count:
            logger.info("imported %d legacy run records into the DB", count)
    except Exception as exc:  # noqa: BLE001 — migration is best-effort
        logger.warning("legacy run-record import failed: %s", exc)


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
    # Most-recent output per role.  For roles that fire over multiple
    # debate / risk rounds (Bullish Researcher, Bearish Researcher, the
    # three Risk personas) this holds ONLY the last round — the full
    # per-round history lives in ``reports_by_round`` below.  Kept for
    # backward compatibility with existing run records and downstream
    # consumers that just want "the latest take".
    reports: Dict[str, str] = field(default_factory=dict)
    # Full per-round audit.  Key = role name (same as ``reports`` keys),
    # value = list of ``{"round": N, "output": "...", "degraded": bool}``
    # entries in chronological order.  Single-round roles (analysts,
    # research manager, trader, compliance, PM) get exactly one entry;
    # multi-round roles get one entry per round.  Lets the Reports tab
    # render round 1 / round 2 / … side by side so the user can see
    # how the debate actually evolves instead of only the final stance.
    reports_by_round: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Roles whose final answer came back as the degraded placeholder
    # (see ``trading_crew/_patches.py``).  Persisted so the UI's Recent
    # Runs panel can flag the record on rehydrate, and so post-hoc
    # audits can spot which analysts came back empty without having
    # to grep the per-role report bodies for the marker string.
    degraded_roles: List[str] = field(default_factory=list)
    final_decision: Optional[Dict[str, Any]] = None
    final_decision_source: Optional[str] = None   # "crew" / "critic" / "cascade_override"

    # Report trend signals — the structured lift of the reports desk's
    # Firefly-style signal card (LLM-read filing signals + deterministic
    # financial-trend signals).  ``{signals: [...], net_bias, net_confidence,
    # bias_score}``.  Persisted so the dashboard can render the card post-run
    # without re-running the desk.
    report_signals: Optional[Dict[str, Any]] = None

    # Price-structure signals — the deterministic OHLCV-computed card
    # (VOL_STATE / PRICE_POSITION / REGIME_AGE / DIP_BUY / HURST) plus the
    # numeric read (vol_pct, regime_age_days, dip_buy, size_risk_mult, …).
    # Persisted so the dashboard can render the card post-run.
    price_signals: Optional[Dict[str, Any]] = None

    # Per-desk bias board — each analyst desk's directional lean
    # (``[{desk, bias}, ...]``) handed directly to the PM.  Persisted so the
    # Decision-node panel can render it post-run without re-running the crew.
    desk_bias_board: Optional[List[Dict[str, Any]]] = None

    # Full evidence blackboard (per-desk reports + filled request_evidence
    # addenda).  UI shows the full text; debate tools see a compact TLDR view.
    evidence_blackboard: Optional[Dict[str, Any]] = None

    # Shared bull/bear scratch pad (analyst TLDRs + debate turns + notes).
    debate_scratchpad: Optional[Dict[str, Any]] = None

    # Post-PM alignment — desk-board / computed-trend disagreement applied
    # to the PM action (HOLD on split/majority-oppose, STRONG cap).
    signal_alignment: Optional[Dict[str, Any]] = None

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


def write_run(record: RunRecord) -> None:
    """Persist ``record`` to the ``runs`` table.

    Never raises — a storage error is logged so it can't abort the kickoff.
    """
    try:
        from web.backend import prediction_db as db

        payload = record.to_dict()
        db.upsert_run(payload)
        try:
            db.record_from_run(payload)
        except Exception:  # noqa: BLE001 — index write must not abort the audit
            logger.exception(
                "Failed to index run %s into today's predictions", record.run_id
            )
    except Exception as exc:  # noqa: BLE001 — persistence must not abort a run
        logger.exception("Failed to write run record %s: %s", record.run_id, exc)


def list_recent_runs(limit: int = 20, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the most-recent ``limit`` run-index rows, newest first."""
    try:
        from web.backend import prediction_db as db

        _migrate_legacy_runs()
        return db.list_runs(limit=limit, ticker=ticker)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to list recent runs")
        return []


def load_run(run_id: str, ticker: str, full: bool = False) -> Optional[Dict[str, Any]]:
    """Load a run by id + ticker.  Default is the header (no report bodies)."""
    try:
        from web.backend import prediction_db as db

        _migrate_legacy_runs()
        return db.get_run(run_id, ticker, full=full)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load run %s", run_id)
        return None


def load_latest_run(ticker: str, full: bool = False) -> Optional[Dict[str, Any]]:
    """Return the most recent run header for ``ticker`` (or None)."""
    recent = list_recent_runs(limit=1, ticker=ticker)
    if not recent:
        return None
    entry = recent[0]
    return load_run(entry["run_id"], entry["ticker"], full=full)


def clear_runs(ticker: Optional[str] = None) -> Dict[str, Any]:
    """Delete run records from the DB (all, or one ticker's).

    Returns ``{"removed": N, "trash_path": None}`` — the field is kept for
    API/UI compatibility; DB deletes are not moved to a filesystem trash.
    """
    try:
        from web.backend import prediction_db as db

        _migrate_legacy_runs()
        removed = db.clear_runs(ticker=ticker)
        return {"removed": int(removed or 0), "trash_path": None}
    except Exception:  # noqa: BLE001
        logger.exception("Failed to clear runs")
        return {"removed": 0, "trash_path": None}
