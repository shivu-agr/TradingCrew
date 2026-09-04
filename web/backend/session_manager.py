"""In-memory multi-run session manager for the trading dashboard.

Motivation
----------
The original ``/ws/analyze`` endpoint tied a run 1:1 to a WebSocket:
closing the tab cancelled the CrewAI kickoff mid-flight, and there was
no way to fan a run out to multiple viewers or resume from a page
reload.  Everything happening in the analysis was ephemeral — either
you had the tab open the whole time or the run was lost.

This module de-couples "the run" from "the viewer".  A single
``SessionManager`` (created once at FastAPI startup) owns every
in-flight and recently-completed run in the process.  A run is driven
by an ``AnalysisRunner`` on a background task and has an authoritative
list of events on the session — WebSocket clients subscribe (and
un-subscribe) to that list, get an atomic replay of everything that's
already happened, and then tail live events until the run terminates
or they disconnect.  Disconnecting a viewer *never* cancels the run.

Concurrency
-----------
Multiple sessions can execute in parallel — bounded by
``TRADINGCREW_MAX_CONCURRENT_RUNS`` (default 4).  vLLM decode
throughput scales sub-linearly with active batch size, so 3-4
concurrent kickoffs on 1×H100 with Qwen3-32B roughly triples the
aggregate tokens/sec vs a purely sequential queue.  Any starts beyond
the cap wait on an ``asyncio.Semaphore`` and progress from
``queued`` → ``running`` when a slot frees up.

The CrewAI event bus is a *process-global* singleton, so runs would
otherwise cross-contaminate each other's tool events.  ``AnalysisRunner``
now records its own crew's agent-id set at kickoff and filters
incoming tool events by ``event.agent_id`` — see ``_is_own_event``
in ``runner.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = {RunStatus.COMPLETED, RunStatus.ERROR, RunStatus.CANCELLED}


# Sentinel that the dispatcher pushes into subscriber queues once the
# run has drained its final event.  Distinguished from real events by
# its type so WS consumers know to close cleanly instead of loop-blocking.
_EOF = {"type": "__eof__"}


@dataclass
class RunSession:
    """A single trading-analysis run and its viewer fan-out.

    All mutation happens on the FastAPI event loop; the ``AnalysisRunner``
    itself runs its CrewAI kickoff on a worker thread and pushes events
    back onto its own ``asyncio.Queue`` (see ``AnalysisRunner.events``)
    which the dispatcher drains here on the loop.
    """

    run_id: str
    ticker: str
    cfg: Dict[str, Any]
    # Session trace id for log group-by + per-session search metrics.  A batch
    # passes its ``job_id`` here (all its tickers share it); a standalone run
    # falls back to its own ``run_id`` (set in ``SessionManager.start``).
    trace_id: str = ""
    status: RunStatus = RunStatus.QUEUED
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    subscribers: Set[asyncio.Queue] = field(default_factory=set)
    # Snapshot fields extracted from events for the /api/runs/active list
    # so a page-reload can render the sidebar without replaying every
    # event on every session.
    final_action: Optional[str] = None
    error: Optional[str] = None
    agent_count: Optional[int] = None
    task_count: Optional[int] = None
    # Runtime handles — kept off the wire in ``snapshot()``.
    runner: Any = None
    run_task: Optional[asyncio.Task] = None
    dispatcher_task: Optional[asyncio.Task] = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)

    def snapshot(self) -> Dict[str, Any]:
        """Serializable snapshot for /api/runs/active."""
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "ticker": self.ticker,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "final_action": self.final_action,
            "error": self.error,
            "event_count": len(self.events),
            "subscribers": len(self.subscribers),
            "agent_count": self.agent_count,
            "task_count": self.task_count,
            "cfg": {
                k: v for k, v in self.cfg.items()
                if k in ("debate_rounds", "risk_rounds", "asset_class", "run_mode", "quality_review", "compliance", "llm_preset", "embedding_preset", "book", "trade_date", "decision_ts")
            },
        }


class SessionManager:
    """Process-wide store of active + recently-completed runs."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        max_concurrent: int = 4,
        max_total: int = 20,
        runner_factory: Optional[Callable[[asyncio.AbstractEventLoop], Any]] = None,
    ):
        self._loop = loop
        self._sessions: Dict[str, RunSession] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._max_total = max_total
        # Tests inject a lightweight fake runner via ``runner_factory``
        # so we don't need CrewAI in the test env.  Production imports
        # the real ``AnalysisRunner`` lazily below to keep the top of
        # this module free of heavy transitive imports.
        self._runner_factory = runner_factory

    # ---- runner factory ----------------------------------------------------
    def _make_runner(self) -> Any:
        if self._runner_factory is not None:
            return self._runner_factory(self._loop)
        # Deferred import — runner.py pulls in the full crew stack.
        from .runner import AnalysisRunner
        return AnalysisRunner(self._loop)

    # ---- public API --------------------------------------------------------
    async def start(self, ticker: str, cfg: Dict[str, Any]) -> RunSession:
        """Register a new run and schedule it in the background.

        Returns immediately with a session in ``QUEUED`` or ``RUNNING``
        state; the semaphore inside ``_drive`` gates progression to
        RUNNING so bursts beyond ``max_concurrent`` don't overload
        vLLM's decode batch.
        """
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
        # A batch tags every ticker with the shared job_id (via cfg["trace_id"])
        # so search metrics + logs group per batch; a lone run traces to itself.
        trace_id = str(cfg.get("trace_id") or run_id)
        session = RunSession(run_id=run_id, ticker=ticker, cfg=cfg, trace_id=trace_id)
        self._sessions[run_id] = session
        self._evict()
        session.run_task = asyncio.create_task(
            self._drive(session), name=f"run:{run_id}"
        )
        return session

    def get(self, run_id: str) -> Optional[RunSession]:
        return self._sessions.get(run_id)

    def list(self) -> List[Dict[str, Any]]:
        """Snapshots of every known session, newest first."""
        return sorted(
            [s.snapshot() for s in self._sessions.values()],
            key=lambda x: x["started_at"],
            reverse=True,
        )

    def subscribe(self, run_id: str):
        """Attach a viewer.  Returns ``(session, replay, queue)``.

        ``replay`` is a *shallow copy* of events already emitted, safe
        to iterate without holding a lock (asyncio doesn't preempt
        between adjacent sync statements, so the copy + set-insert
        pair below is atomic w.r.t. the dispatcher).
        """
        session = self._sessions.get(run_id)
        if session is None:
            return None, [], None
        q: asyncio.Queue = asyncio.Queue()
        replay = list(session.events)
        session.subscribers.add(q)
        # If the run has already terminated we still let the caller
        # replay the buffer, but we drop an EOF into the queue so it
        # doesn't hang waiting for a live tail.
        if session.status in TERMINAL_STATUSES:
            q.put_nowait(_EOF)
        return session, replay, q

    def unsubscribe(self, session: RunSession, q: asyncio.Queue) -> None:
        session.subscribers.discard(q)

    async def cancel(self, run_id: str) -> bool:
        s = self._sessions.get(run_id)
        if s is None or s.status in TERMINAL_STATUSES:
            return False
        s.status = RunStatus.CANCELLED
        if s.runner is not None:
            try:
                s.runner.cancel()
            except Exception:
                logger.exception("runner.cancel() failed for %s", run_id)
        return True

    def remove(self, run_id: str) -> bool:
        """Evict a *terminal* session from memory.  Refuses to remove
        an active run — callers must ``cancel()`` first."""
        s = self._sessions.get(run_id)
        if s is None:
            return False
        if s.status not in TERMINAL_STATUSES:
            return False
        del self._sessions[run_id]
        return True

    # ---- internals ---------------------------------------------------------
    async def _drive(self, session: RunSession) -> None:
        """Own the full life-cycle of one session."""
        from trading_crew.observability.trace import reset_trace_id, set_trace_id

        # Bind the trace id on THIS task's context before any child task /
        # thread is spawned — asyncio.create_task (dispatcher), asyncio.to_thread
        # (kickoff) and CrewAI's Task.execute_async all copy the context, so the
        # id rides all the way down to the analyst search calls for free.
        _trace_token = set_trace_id(session.trace_id or session.run_id)
        try:
            async with self._semaphore:
                if session.status == RunStatus.CANCELLED:
                    # User cancelled while queued — never start the runner.
                    session.completed_at = datetime.now(timezone.utc)
                    self._finalize_no_run(session)
                    return
                session.status = RunStatus.RUNNING
                runner = self._make_runner()
                session.runner = runner
                # Dispatcher fans events out to buffer + subscribers.
                session.dispatcher_task = asyncio.create_task(
                    self._dispatch(session), name=f"dispatch:{session.run_id}"
                )
                try:
                    await runner.run(session.ticker, session.cfg)
                except Exception as exc:
                    logger.exception("runner.run failed for %s", session.run_id)
                    # If the runner explodes we still need to unblock
                    # the dispatcher so viewers don't hang forever.
                    session.error = str(exc)
                    try:
                        # Push a synthetic error into the runner's event
                        # queue so _dispatch tears down cleanly.
                        runner.events.put_nowait({"type": "error", "message": str(exc)})
                    except Exception:
                        pass
                # Wait for the dispatcher to drain to a terminal event.
                if session.dispatcher_task is not None:
                    try:
                        await asyncio.wait_for(session.dispatcher_task, timeout=15.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "Dispatcher for %s did not drain in 15s; cancelling",
                            session.run_id,
                        )
                        session.dispatcher_task.cancel()
        finally:
            reset_trace_id(_trace_token)
            self._finalize(session)

    async def _dispatch(self, session: RunSession) -> None:
        """Drain the runner's event queue onto session state + viewers."""
        runner = session.runner
        while True:
            event = await runner.events.get()
            session.events.append(event)
            # Snapshot the important bits so /api/runs/active can show
            # a rich sidebar without replaying the whole buffer.
            t = event.get("type")
            if t == "run_started":
                session.agent_count = event.get("agent_count")
                session.task_count = event.get("task_count")
            elif t == "final_decision":
                dec = event.get("decision") or {}
                if isinstance(dec, dict):
                    session.final_action = dec.get("action") or session.final_action
            elif t == "error":
                session.error = event.get("message")
            # Fan out to any live viewers.  ``put_nowait`` because
            # subscriber queues are unbounded — falling behind on the
            # wire never blocks the analysis.
            for q in list(session.subscribers):
                try:
                    q.put_nowait(event)
                except Exception:
                    logger.exception(
                        "put_nowait on subscriber queue failed (%s)", session.run_id
                    )
            if t in ("run_completed", "error"):
                # Push EOF so tailing subscribers can break cleanly
                # (in addition to receiving the run_completed/error).
                for q in list(session.subscribers):
                    try:
                        q.put_nowait(_EOF)
                    except Exception:
                        pass
                return

    def _finalize(self, session: RunSession) -> None:
        """Called after the driver task exits (normal or error)."""
        if session.status == RunStatus.RUNNING:
            # Prefer the more specific status derived from the event
            # stream if the runner told us it errored.
            session.status = (
                RunStatus.ERROR if session.error else RunStatus.COMPLETED
            )
        session.completed_at = datetime.now(timezone.utc)
        # Anyone still subscribed learns the truth.  Idempotent:
        # _dispatch already pushed EOF on run_completed/error, but a
        # cancelled-while-queued run never called _dispatch.
        for q in list(session.subscribers):
            try:
                q.put_nowait(_EOF)
            except Exception:
                pass
        session.done_event.set()
        # Hand this run's freed heap back to the OS so a long batch's RSS
        # doesn't creep upward run-after-run (see trading_crew.memutil).
        try:
            from trading_crew.memutil import release_memory

            release_memory()
        except Exception:  # noqa: BLE001 - cleanup must never break a run
            pass

    def _finalize_no_run(self, session: RunSession) -> None:
        """Cancelled-before-start path — no runner, no dispatcher."""
        session.status = RunStatus.CANCELLED
        session.completed_at = datetime.now(timezone.utc)
        for q in list(session.subscribers):
            try:
                q.put_nowait({"type": "error", "message": "cancelled before start"})
                q.put_nowait(_EOF)
            except Exception:
                pass
        session.done_event.set()

    def _evict(self) -> None:
        """Keep the store bounded by ``max_total`` — drop the oldest
        *terminal* sessions first.  Active runs are never evicted."""
        if len(self._sessions) <= self._max_total:
            return
        terminals = [
            s for s in self._sessions.values() if s.status in TERMINAL_STATUSES
        ]
        terminals.sort(key=lambda s: s.completed_at or s.started_at)
        for s in terminals:
            if len(self._sessions) <= self._max_total:
                break
            self._sessions.pop(s.run_id, None)


# ---------------------------------------------------------------------------
# Singleton wiring for the FastAPI process
# ---------------------------------------------------------------------------

_manager: Optional[SessionManager] = None


def init_session_manager(loop: asyncio.AbstractEventLoop) -> SessionManager:
    global _manager
    if _manager is None:
        try:
            max_concurrent = int(os.environ.get("TRADINGCREW_MAX_CONCURRENT_RUNS", "4"))
        except ValueError:
            max_concurrent = 4
        try:
            max_total = int(os.environ.get("TRADINGCREW_MAX_SESSIONS_RETAINED", "20"))
        except ValueError:
            max_total = 20
        _manager = SessionManager(
            loop=loop, max_concurrent=max_concurrent, max_total=max_total
        )
    return _manager


def get_session_manager() -> SessionManager:
    if _manager is None:
        raise RuntimeError(
            "SessionManager not initialised — call init_session_manager(loop) first."
        )
    return _manager


def reset_session_manager_for_tests() -> None:
    """Test-only hook so pytest can start with a fresh manager."""
    global _manager
    _manager = None
