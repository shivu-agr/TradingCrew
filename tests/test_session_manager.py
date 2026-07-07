"""Unit tests for the multi-run session manager (web/backend/session_manager.py).

Uses ``pytest-asyncio`` so each test runs on a proper live event loop
(so ``asyncio.create_task`` and ``asyncio.Semaphore`` behave correctly)
and injects a lightweight fake ``AnalysisRunner`` so we don't need
CrewAI or vLLM alive to exercise the state machine, replay buffer, or
subscriber fan-out.
"""

import asyncio
from typing import Any, Dict, List, Optional

import pytest

from web.backend.session_manager import (
    RunStatus,
    SessionManager,
    _EOF,
)

pytestmark = pytest.mark.asyncio


class FakeRunner:
    """Minimal drop-in for ``AnalysisRunner`` used across tests.

    ``events`` mirrors the real runner's asyncio queue.  ``run(...)``
    pushes whatever the test seeded via ``script`` and returns; with
    ``stall_forever=True`` it blocks until ``cancel()`` is called so
    tests can inspect the RUNNING state and later exercise cancellation.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        script: Optional[List[Dict[str, Any]]] = None,
        stall_forever: bool = False,
    ):
        self._loop = loop
        self.events: asyncio.Queue = asyncio.Queue()
        self._cancelled = asyncio.Event()
        self._script = list(script or [])
        self._stall_forever = stall_forever
        self.run_called_with = None

    def cancel(self) -> None:
        self._cancelled.set()

    async def run(self, ticker: str, cfg: Dict[str, Any]) -> None:
        self.run_called_with = (ticker, cfg)
        for ev in self._script:
            if self._cancelled.is_set():
                await self.events.put({"type": "error", "message": "cancelled"})
                return
            await self.events.put(ev)
            await asyncio.sleep(0)
        if self._stall_forever:
            await self._cancelled.wait()
            await self.events.put({"type": "error", "message": "cancelled"})


def _sm(script=None, max_concurrent=4, max_total=20, **runner_kw):
    """Build a manager wired to a FakeRunner factory."""
    loop = asyncio.get_event_loop()

    def factory(_loop):
        return FakeRunner(_loop, script=script, **runner_kw)

    return SessionManager(
        loop=loop,
        max_concurrent=max_concurrent,
        max_total=max_total,
        runner_factory=factory,
    )


# ---------------------------------------------------------------------------
# Core lifecycle
# ---------------------------------------------------------------------------

async def test_start_records_session_and_transitions_to_running():
    script = [
        {"type": "run_started", "agent_count": 3, "task_count": 5},
        {"type": "final_decision", "decision": {"action": "BUY"}},
        {"type": "run_completed"},
    ]
    sm = _sm(script=script)
    session = await sm.start("NVDA", {"debate_rounds": 2})
    assert session.status in (RunStatus.QUEUED, RunStatus.RUNNING)
    await asyncio.wait_for(session.done_event.wait(), timeout=2.0)
    assert session.status == RunStatus.COMPLETED
    assert session.final_action == "BUY"
    assert session.agent_count == 3
    assert session.task_count == 5
    assert [e["type"] for e in session.events] == [
        "run_started", "final_decision", "run_completed",
    ]


async def test_disconnect_does_not_cancel_run():
    """Subscribing / unsubscribing must never abort the underlying run."""
    script = [
        {"type": "run_started", "agent_count": 1, "task_count": 1},
        {"type": "run_completed"},
    ]
    sm = _sm(script=script)
    session = await sm.start("AAPL", {})
    session_view, replay, q = sm.subscribe(session.run_id)
    assert session_view is session
    sm.unsubscribe(session, q)
    await asyncio.wait_for(session.done_event.wait(), timeout=2.0)
    assert session.status == RunStatus.COMPLETED
    assert any(e["type"] == "run_completed" for e in session.events)


# ---------------------------------------------------------------------------
# Replay + fan-out
# ---------------------------------------------------------------------------

async def test_late_subscriber_replays_buffered_events():
    """A viewer that connects after the run has emitted events must
    replay every prior event before tailing live."""
    script = [
        {"type": "run_started"},
        {"type": "node_started", "node": "Market Analyst"},
        {"type": "node_completed", "node": "Market Analyst"},
        {"type": "run_completed"},
    ]
    sm = _sm(script=script)
    session = await sm.start("AAPL", {})
    await asyncio.wait_for(session.done_event.wait(), timeout=2.0)
    _, replay, q = sm.subscribe(session.run_id)
    assert [e["type"] for e in replay] == [
        "run_started", "node_started", "node_completed", "run_completed",
    ]
    # Terminal sessions push EOF into the queue immediately so a
    # tail loop can exit without hanging.
    assert q.get_nowait() is _EOF


async def test_multiple_subscribers_all_get_live_events():
    """Two viewers subscribed early both receive the live tail."""
    started = asyncio.Event()
    stop = asyncio.Event()

    class SlowRunner:
        def __init__(self):
            self.events: asyncio.Queue = asyncio.Queue()

        def cancel(self):
            pass

        async def run(self, ticker, cfg):
            started.set()
            await self.events.put({"type": "run_started"})
            await stop.wait()
            await self.events.put({"type": "final_decision", "decision": {"action": "HOLD"}})
            await self.events.put({"type": "run_completed"})

    sm = SessionManager(
        loop=asyncio.get_event_loop(),
        max_concurrent=4,
        runner_factory=lambda l: SlowRunner(),
    )
    session = await sm.start("MSFT", {})
    await asyncio.wait_for(started.wait(), timeout=1.0)
    await asyncio.sleep(0.05)  # let dispatcher pick up the first event

    _, replay_a, q_a = sm.subscribe(session.run_id)
    _, replay_b, q_b = sm.subscribe(session.run_id)
    assert any(e["type"] == "run_started" for e in replay_a)
    assert any(e["type"] == "run_started" for e in replay_b)
    stop.set()
    await asyncio.wait_for(session.done_event.wait(), timeout=2.0)

    def drain(q):
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    types_a = [e.get("type") for e in drain(q_a)]
    types_b = [e.get("type") for e in drain(q_b)]
    assert "final_decision" in types_a and "run_completed" in types_a
    assert "final_decision" in types_b and "run_completed" in types_b
    assert _EOF in [e if e is _EOF else None for e in drain(q_a) + drain(q_b) + [_EOF]]  # sentinel present


async def test_subscribe_unknown_run_id_returns_none():
    sm = _sm()
    s, replay, q = sm.subscribe("bogus")
    assert s is None
    assert replay == []
    assert q is None


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------

async def test_cancel_stops_running_session():
    script = [{"type": "run_started"}]
    sm = _sm(script=script, stall_forever=True)
    session = await sm.start("TSLA", {})
    for _ in range(200):
        if any(e.get("type") == "run_started" for e in session.events):
            break
        await asyncio.sleep(0.01)
    assert session.status == RunStatus.RUNNING
    assert await sm.cancel(session.run_id) is True
    await asyncio.wait_for(session.done_event.wait(), timeout=2.0)
    # Cancelled sessions get status=CANCELLED regardless of how the
    # runner shut down; the last event in the buffer is the runner's
    # synthetic {"type": "error", "message": "cancelled"}.
    assert session.status == RunStatus.CANCELLED
    assert any(e.get("type") == "error" for e in session.events)


async def test_cancel_unknown_run_id_returns_false():
    sm = _sm()
    assert (await sm.cancel("nope")) is False


async def test_cancel_terminal_session_is_noop():
    """Once completed a session can't be re-cancelled — helps catch
    UI logic that re-issues cancel on the completed row by accident."""
    sm = _sm(script=[{"type": "run_completed"}])
    session = await sm.start("GOOG", {})
    await asyncio.wait_for(session.done_event.wait(), timeout=2.0)
    assert session.status == RunStatus.COMPLETED
    assert (await sm.cancel(session.run_id)) is False


# ---------------------------------------------------------------------------
# Concurrency + eviction
# ---------------------------------------------------------------------------

async def test_max_concurrent_caps_running_sessions():
    """With ``max_concurrent=2`` a third start must sit in QUEUED until
    one of the first two completes."""
    releases = [asyncio.Event(), asyncio.Event(), asyncio.Event()]
    idx = {"i": 0}

    def factory(_loop):
        i = idx["i"]
        idx["i"] += 1
        release = releases[i]

        class Stalling:
            def __init__(self):
                self.events: asyncio.Queue = asyncio.Queue()

            def cancel(self):
                pass

            async def run(self, t, c):
                await self.events.put({"type": "run_started"})
                await release.wait()
                await self.events.put({"type": "run_completed"})

        return Stalling()

    sm = SessionManager(
        loop=asyncio.get_event_loop(),
        max_concurrent=2,
        runner_factory=factory,
    )
    a = await sm.start("A", {})
    b = await sm.start("B", {})
    c = await sm.start("C", {})
    await asyncio.sleep(0.1)
    running = [s for s in [a, b, c] if s.status == RunStatus.RUNNING]
    queued = [s for s in [a, b, c] if s.status == RunStatus.QUEUED]
    assert len(running) == 2, [s.status for s in (a, b, c)]
    assert len(queued) == 1
    releases[0].set()
    releases[1].set()
    await asyncio.wait_for(a.done_event.wait(), timeout=1.0)
    await asyncio.wait_for(b.done_event.wait(), timeout=1.0)
    await asyncio.sleep(0.05)
    assert c.status == RunStatus.RUNNING
    releases[2].set()
    await asyncio.wait_for(c.done_event.wait(), timeout=1.0)


async def test_evict_drops_oldest_terminal_over_cap():
    """When the retention cap is exceeded the oldest COMPLETED /
    ERROR / CANCELLED session is dropped — never an in-flight one."""
    sm = _sm(script=[{"type": "run_completed"}], max_total=2)
    s1 = await sm.start("A", {})
    await asyncio.wait_for(s1.done_event.wait(), timeout=1.0)
    s2 = await sm.start("B", {})
    await asyncio.wait_for(s2.done_event.wait(), timeout=1.0)
    # 2 terminal sessions.  Starting a 3rd evicts the oldest.
    s3 = await sm.start("C", {})
    listing = [x["run_id"] for x in sm.list()]
    assert s1.run_id not in listing
    assert s2.run_id in listing
    assert s3.run_id in listing


async def test_remove_only_when_terminal():
    """`remove()` refuses to drop a running run — the API layer needs
    this so a naive DELETE call doesn't corrupt in-flight state."""
    sm = _sm(script=[{"type": "run_started"}], stall_forever=True)
    s = await sm.start("A", {})
    await asyncio.sleep(0.05)
    assert s.status == RunStatus.RUNNING
    assert sm.remove(s.run_id) is False
    await sm.cancel(s.run_id)
    await asyncio.wait_for(s.done_event.wait(), timeout=1.0)
    assert sm.remove(s.run_id) is True
    assert sm.get(s.run_id) is None


async def test_list_snapshot_shape():
    """Verify /api/runs/active payload has the fields the frontend needs."""
    sm = _sm(script=[
        {"type": "run_started", "agent_count": 4, "task_count": 6},
        {"type": "final_decision", "decision": {"action": "STRONG_BUY"}},
        {"type": "run_completed"},
    ])
    s = await sm.start("NVDA", {"debate_rounds": 3})
    await asyncio.wait_for(s.done_event.wait(), timeout=2.0)
    listing = sm.list()
    assert len(listing) == 1
    row = listing[0]
    assert row["run_id"] == s.run_id
    assert row["ticker"] == "NVDA"
    assert row["status"] == "completed"
    assert row["final_action"] == "STRONG_BUY"
    assert row["agent_count"] == 4
    assert row["task_count"] == 6
    assert row["event_count"] == 3
    assert row["cfg"] == {"debate_rounds": 3}
