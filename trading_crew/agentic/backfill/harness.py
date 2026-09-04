"""Backfill harness — generate historical, point-in-time decision labels.

The unit of work is a *cell*: one ``(ticker, as_of_date)`` pair.  For each
cell we run the crew under a strict :class:`AsOfContext`, bridge the PM's
``PortfolioDecision`` into an ``ActionProposal``, and write a
``source="backfill"`` episode.  After all cells complete we run one
outcome-resolution pass (outside any as_of scope, so the resolver can see
the post-decision tape needed to price the realised return).

State/resumability
------------------
A job is identified deterministically by a hash of its config, so restarting
the same config resumes where it stopped.  A JSON checkpoint at
``$TRADINGCREW_CACHE_DIR/backfill/state/<job_id>.json`` records each cell's
status; ``done`` cells are skipped on resume, and episodes are idempotent
(``EpisodicMemory.add`` upserts by ``episode_id``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config + cells
# ---------------------------------------------------------------------------


@dataclass
class BackfillConfig:
    """Everything needed to define (and deterministically identify) a run."""

    tickers: List[str]
    start_date: str  # YYYY-MM-DD (inclusive)
    end_date: str    # YYYY-MM-DD (inclusive)
    spacing_days: int = 30
    horizon_days: int = 90
    debate_rounds: int = 1
    risk_rounds: int = 1
    resolve: bool = True          # run the L2 outcome-resolution pass at the end
    reflect: bool = False         # generate LLM reflections during resolution
    cell_gap_seconds: float = 5.0  # polite gap between cells (web scraping)
    only_resolvable: bool = True  # skip cells whose outcome window is not yet in the past

    def normalized_tickers(self) -> List[str]:
        seen: List[str] = []
        for t in self.tickers:
            u = (t or "").strip().upper()
            if u and u not in seen:
                seen.append(u)
        return seen

    def fingerprint(self) -> Dict[str, object]:
        return {
            "tickers": self.normalized_tickers(),
            "start_date": self.start_date,
            "end_date": self.end_date,
            "spacing_days": self.spacing_days,
            "horizon_days": self.horizon_days,
        }


@dataclass
class BackfillCell:
    ticker: str
    as_of: str  # YYYY-MM-DD

    @property
    def cell_id(self) -> str:
        return f"{self.ticker.upper()}@{self.as_of}"


def job_id_for(config: BackfillConfig) -> str:
    """Stable job id derived from the run-defining config fields.

    Two runs with the same tickers / window / spacing / horizon share a
    checkpoint so a stop→resume continues the same job.  Cosmetic knobs
    (gaps, debate rounds) are intentionally excluded from the fingerprint.
    """
    blob = json.dumps(config.fingerprint(), sort_keys=True).encode("utf-8")
    return "bf-" + hashlib.sha256(blob).hexdigest()[:12]


def _parse_date(d: str) -> datetime:
    return datetime.strptime(d[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)


def generate_cells(config: BackfillConfig, *, now: Optional[datetime] = None) -> List[BackfillCell]:
    """Enumerate every ``(ticker, as_of)`` cell in the configured window.

    When ``only_resolvable`` is set, cells whose outcome window
    (``as_of + horizon_days``) has not yet elapsed are dropped — they
    couldn't be labeled now anyway.
    """
    now = now or datetime.now(timezone.utc)
    start = _parse_date(config.start_date)
    end = _parse_date(config.end_date)
    step = max(1, int(config.spacing_days))
    dates: List[str] = []
    cur = start
    while cur <= end:
        if config.only_resolvable and (cur + timedelta(days=config.horizon_days)) >= now:
            cur += timedelta(days=step)
            continue
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=step)
    cells: List[BackfillCell] = []
    for ticker in config.normalized_tickers():
        for d in dates:
            cells.append(BackfillCell(ticker=ticker, as_of=d))
    return cells


# ---------------------------------------------------------------------------
# Checkpoint / state
# ---------------------------------------------------------------------------


def _state_dir() -> Path:
    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache_dir) / "backfill" / "state"


def _state_path(job_id: str) -> Path:
    return _state_dir() / f"{job_id}.json"


def load_state(job_id: str) -> Dict[str, dict]:
    """Load the per-cell status map for ``job_id`` (empty if none yet)."""
    path = _state_path(job_id)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("cells", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(job_id: str, config: BackfillConfig, cells: Dict[str, dict]) -> None:
    path = _state_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "job_id": job_id,
        "config": config.fingerprint(),
        "updated_ts": datetime.now(timezone.utc).isoformat(),
        "cells": cells,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Episode recording (headless — mirrors runner._record_episode but backfill)
# ---------------------------------------------------------------------------


def _memory_path() -> Path:
    cache_dir = os.environ.get("TRADINGCREW_CACHE_DIR") or os.path.expanduser("~/.trading_crew")
    return Path(cache_dir) / "memory" / "episodes.jsonl"


def _record_backfill_episode(memory, ticker: str, proposal, decision_ts: str) -> str:
    """Write a PENDING ``source="backfill"`` episode; returns its id."""
    from trading_crew.agentic.memory import Episode, OutcomeStatus, Regime
    from trading_crew.agentic.versioning import (
        code_git_sha,
        prompts_hash,
        workflow_version,
    )

    try:
        d = datetime.fromisoformat(decision_ts.replace("Z", "+00:00"))
    except ValueError:
        d = datetime.now(timezone.utc)
    outcome_dt = d + timedelta(days=max(proposal.horizon_days, 1))
    ep = Episode(
        episode_id=f"{ticker.upper()}-{decision_ts}",
        symbol=ticker.upper(),
        decision_ts=decision_ts,
        state_summary=proposal.rationale[:1000],
        regime=Regime.UNKNOWN,
        action_proposal=proposal.model_dump(),
        outcome_ts=outcome_dt.isoformat(),
        outcome_status=OutcomeStatus.PENDING,
        embargo_days=max(proposal.horizon_days, 1),
        workflow_version=workflow_version(),
        prompts_hash=prompts_hash(),
        code_git_sha=code_git_sha(),
        source="backfill",
    )
    memory.add(ep)
    return ep.episode_id


def _run_one_cell(cell: BackfillCell, config: BackfillConfig, memory) -> str:
    """Run the crew for one cell under a PIT scope; return the episode id."""
    from trading_crew import TradingCrew
    from trading_crew.agentic.bridge import portfolio_decision_to_action_proposal
    from trading_crew.data_sources import pit

    # Resolve to the canonical symbol so the episode matches live runs.
    ticker = cell.ticker
    try:
        from trading_crew.market_context import resolve_ticker

        resolved = resolve_ticker(ticker)
        if resolved:
            ticker = resolved
    except Exception:  # noqa: BLE001
        pass

    as_of_dt = _parse_date(cell.as_of)
    decision_ts = as_of_dt.isoformat()

    # Memory (L1 retrieval) is disabled in backfill: its embargo is keyed to
    # *now*, not to as_of, so retrieving a lesson that only resolved after the
    # decision date would leak the future into a historical decision.
    with pit.as_of_scope(as_of_dt, mode="backfill", strict=True):
        tc = TradingCrew(
            ticker=ticker,
            debate_rounds=config.debate_rounds,
            risk_rounds=config.risk_rounds,
            memory=False,
        )
        result = tc.crew().kickoff(inputs={"ticker": ticker, "trade_date": cell.as_of})
        decision = getattr(result, "pydantic", None)
        if decision is None:
            raise RuntimeError("crew produced no structured PortfolioDecision")
        proposal = portfolio_decision_to_action_proposal(
            decision, symbol=ticker, decision_ts=decision_ts
        )
        return _record_backfill_episode(memory, ticker, proposal, decision_ts)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_backfill(
    config: BackfillConfig,
    *,
    should_stop: Optional[Callable[[], bool]] = None,
    on_log: Optional[Callable[[dict], None]] = None,
    memory=None,
    now: Optional[datetime] = None,
) -> dict:
    """Run (or resume) a backfill job; return a summary dict.

    ``should_stop`` is polled between cells for cooperative cancellation.
    ``on_log`` receives per-cell progress dicts.  ``memory`` / ``now`` are
    injected for tests.
    """
    from trading_crew.agentic.memory import EpisodicMemory

    should_stop = should_stop or (lambda: False)
    log = on_log or (lambda payload: None)
    memory = memory or EpisodicMemory(_memory_path())

    job_id = job_id_for(config)
    cells = generate_cells(config, now=now)
    state = load_state(job_id)

    total = len(cells)
    done = sum(1 for c in cells if state.get(c.cell_id, {}).get("status") == "done")
    log({"event": "start", "job_id": job_id, "total_cells": total, "already_done": done})

    processed_ticker_set = set()
    stopped = False
    for idx, cell in enumerate(cells):
        if should_stop():
            stopped = True
            log({"event": "stopped", "at_cell": idx, "cell_id": cell.cell_id})
            break
        prior = state.get(cell.cell_id, {})
        if prior.get("status") == "done":
            continue

        state[cell.cell_id] = {"status": "running", "ts": datetime.now(timezone.utc).isoformat()}
        _save_state(job_id, config, state)
        log({"event": "cell_start", "cell_id": cell.cell_id, "index": idx, "total": total})

        t0 = time.time()
        try:
            episode_id = _run_one_cell(cell, config, memory)
            state[cell.cell_id] = {
                "status": "done",
                "episode_id": episode_id,
                "ts": datetime.now(timezone.utc).isoformat(),
                "elapsed_sec": round(time.time() - t0, 2),
            }
            processed_ticker_set.add(cell.ticker.upper())
            log({"event": "cell_done", "cell_id": cell.cell_id, "episode_id": episode_id})
        except Exception as exc:  # noqa: BLE001 — one bad cell must not kill the run
            logger.exception("backfill cell failed: %s", cell.cell_id)
            state[cell.cell_id] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            log({"event": "cell_failed", "cell_id": cell.cell_id, "error": str(exc)})
        _save_state(job_id, config, state)

        # Polite gap for web-scraping sources (on top of the screener throttle).
        if config.cell_gap_seconds > 0 and idx < total - 1 and not should_stop():
            time.sleep(config.cell_gap_seconds)

    # Outcome resolution runs OUTSIDE any as_of scope so the resolver can see
    # the post-decision tape it needs to price the realised return.
    resolved_count = 0
    if config.resolve and not stopped:
        resolved_count = _resolve_backfill(config, memory, log)

    summary = {
        "job_id": job_id,
        "total_cells": total,
        "done": sum(1 for c in cells if state.get(c.cell_id, {}).get("status") == "done"),
        "failed": sum(1 for c in cells if state.get(c.cell_id, {}).get("status") == "failed"),
        "resolved": resolved_count,
        "stopped": stopped,
    }
    log({"event": "finished", **summary})
    return summary


def _resolve_backfill(config: BackfillConfig, memory, log) -> int:
    """Resolve pending backfill episodes into alpha-labeled training rows."""
    from trading_crew.agentic.reflection import resolve_pending_episodes

    llm = None
    if config.reflect:
        try:
            from trading_crew._common import get_llm

            llm = get_llm(temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("backfill reflection LLM unavailable (%s); resolving without it", exc)
            llm = None

    resolved = 0
    for ticker in config.normalized_tickers():
        try:
            records = resolve_pending_episodes(
                ticker=ticker,
                memory=memory,
                llm=llm,
                skip_llm=(llm is None),
            )
            n = sum(1 for r in records if r.status == "RESOLVED")
            resolved += n
            log({"event": "resolved", "ticker": ticker, "n_resolved": n})
        except Exception as exc:  # noqa: BLE001
            logger.warning("backfill resolution failed for %s: %s", ticker, exc)
    return resolved
