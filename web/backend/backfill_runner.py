"""Background controller for the offline backfill harness (Phase 2).

Mirrors the RLMF LoRA job pattern: a single-tenant background thread with a
cooperative stop flag and a pollable status snapshot.  The heavy lifting
lives in ``trading_crew.agentic.backfill``; this module only manages the
thread lifecycle + a rolling log for the UI.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_crew.agentic.backfill import BackfillConfig, generate_cells, job_id_for

logger = logging.getLogger(__name__)


@dataclass
class _BackfillJob:
    job_id: str
    config: Dict[str, Any]
    status: str = "running"  # running | completed | stopped | failed
    started_ts: float = field(default_factory=time.time)
    finished_ts: Optional[float] = None
    total_cells: int = 0
    processed: int = 0
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    thread: Optional[threading.Thread] = None
    stop_flag: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "config": self.config,
            "status": self.status,
            "active": self.status == "running",
            "started_ts": self.started_ts,
            "elapsed_sec": (self.finished_ts or time.time()) - self.started_ts,
            "total_cells": self.total_cells,
            "processed": self.processed,
            "summary": self.summary,
            "error": self.error,
            "logs": self.logs[-300:],
        }


_LOCK = threading.RLock()
_JOB: Optional[_BackfillJob] = None


def start_backfill(
    *,
    tickers: List[str],
    start_date: str,
    end_date: str,
    spacing_days: int = 30,
    horizon_days: int = 90,
    debate_rounds: int = 1,
    risk_rounds: int = 1,
    cell_gap_seconds: float = 5.0,
    resolve: bool = True,
    reflect: bool = False,
    include_unresolvable: bool = False,
) -> Dict[str, Any]:
    """Start a background backfill run (single-tenant). Returns a snapshot."""
    global _JOB
    tickers = [t.strip() for t in (tickers or []) if t and t.strip()]
    if not tickers:
        raise ValueError("at least one ticker is required")
    if not start_date or not end_date:
        raise ValueError("start_date and end_date are required (YYYY-MM-DD)")

    with _LOCK:
        if _JOB is not None and _JOB.status == "running":
            raise ValueError(f"A backfill is already running (job {_JOB.job_id}).")

    config = BackfillConfig(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        spacing_days=spacing_days,
        horizon_days=horizon_days,
        debate_rounds=debate_rounds,
        risk_rounds=risk_rounds,
        cell_gap_seconds=cell_gap_seconds,
        resolve=resolve,
        reflect=reflect,
        only_resolvable=not include_unresolvable,
    )
    job = _BackfillJob(
        job_id=job_id_for(config),
        config=config.fingerprint(),
        total_cells=len(generate_cells(config)),
    )

    def _worker() -> None:
        from trading_crew.agentic.backfill import run_backfill

        def _on_log(payload: dict) -> None:
            job.logs.append({**payload, "ts": datetime.now(timezone.utc).isoformat()})
            if payload.get("event") in ("cell_done", "cell_failed"):
                job.processed += 1

        try:
            summary = run_backfill(config, should_stop=job.stop_flag.is_set, on_log=_on_log)
            job.summary = summary
            job.status = "stopped" if summary.get("stopped") else "completed"
        except Exception as exc:  # noqa: BLE001
            logger.exception("backfill job failed (%s)", job.job_id)
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
        finally:
            job.finished_ts = time.time()

    thread = threading.Thread(target=_worker, name=f"backfill-{job.job_id}", daemon=True)
    job.thread = thread
    with _LOCK:
        _JOB = job
    thread.start()
    return job.snapshot()


def stop_backfill() -> Dict[str, Any]:
    with _LOCK:
        job = _JOB
    if job is None or job.status != "running":
        return {"stopped": False, "reason": "no active backfill"}
    job.stop_flag.set()
    return {"stopped": True, "job_id": job.job_id}


def get_backfill_status() -> Dict[str, Any]:
    with _LOCK:
        job = _JOB
    if job is None:
        return {"active": False, "status": "idle", "job_id": None, "logs": []}
    return job.snapshot()
