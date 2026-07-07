"""Tool_call / tool_result ordering under CrewAI's threaded event bus.

Reproduction of the bug this test covers
----------------------------------------
CrewAI 1.14.4's ``CrewAIEventsBus.emit`` submits sync handlers to a
``ThreadPoolExecutor`` (see ``crewai/events/event_bus.py``).  When a
tool call is fast (e.g. cache hit → ``elapsed_ms=0``) or when the
experimental agent-executor runs several tool calls in parallel via
its own ``ThreadPoolExecutor``, the ``ToolUsageStartedEvent`` and
``ToolUsageFinishedEvent`` handlers race — and the Finished handler
can win, causing the WebSocket stream to receive ``tool_result``
BEFORE its matching ``tool_call``.  The UI then renders the result
card above its own call card (see screenshots from the ZENTEC.NS +
Qwen3-32B run).

Our fix in ``AnalysisRunner`` serialises the handler pair via a
``threading.Condition``.  This test simulates the race by inverting
the wall-clock call order — firing ``_on_tool_finished`` before
``_on_tool_started`` on separate threads — and verifies:

* The two handlers still complete without deadlock.
* The emitted event queue orders every ``tool_call`` strictly before
  its matching ``tool_result``.
* Each ``tool_result`` is tagged with the correct ``call_id`` of its
  corresponding ``tool_call``.
"""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List


def _fresh_runner():
    from web.backend.runner import AnalysisRunner

    loop = asyncio.new_event_loop()
    r = AnalysisRunner(loop)
    # Skip node announcement plumbing — not under test here.
    r._expected_role_order = []
    r._expected_async_flags = []
    r._running_roles = set()
    return r, loop


def _drain(runner) -> List[dict]:
    out: List[dict] = []
    while True:
        try:
            out.append(runner._sync_q.get_nowait())
        except Exception:
            break
    return out


def _started_event(role: str, tool: str, args: dict):
    """Duck-typed ``ToolUsageStartedEvent`` — only the fields our handler reads."""
    return SimpleNamespace(agent_role=role, tool_name=tool, tool_args=args)


def _finished_event(role: str, tool: str, args: dict, output: str, elapsed_ms: int = 0):
    now = datetime.now(timezone.utc)
    return SimpleNamespace(
        agent_role=role,
        tool_name=tool,
        tool_args=args,
        output=output,
        started_at=now,
        finished_at=now + timedelta(milliseconds=elapsed_ms),
    )


def _pair_indices(events, tool):
    """Return (call_idx, result_idx) for the first matching pair of ``tool``."""
    call_idx = next(
        (
            i for i, e in enumerate(events)
            if e["type"] == "tool_call" and e.get("tool") == tool
        ),
        None,
    )
    result_idx = next(
        (
            i for i, e in enumerate(events)
            if e["type"] == "tool_result" and e.get("tool") == tool
        ),
        None,
    )
    return call_idx, result_idx


def test_finished_before_started_preserves_pair_order():
    """The Finished handler must block until Started emits ``tool_call`` first."""
    runner, loop = _fresh_runner()
    try:
        start_ev = _started_event("Fundamentals Analyst", "get_balance_sheet",
                                  {"ticker": "COCHINSHIP.NS"})
        finish_ev = _finished_event("Fundamentals Analyst", "get_balance_sheet",
                                    {"ticker": "COCHINSHIP.NS"},
                                    output="| Metric | 2024 | 2025 |\n|---|---|---|",
                                    elapsed_ms=0)

        # Race simulation: fire Finished first, then Started 100 ms later.
        # Our fix should make Finished block on the condition variable
        # until Started registers the open call.
        released = threading.Event()

        def run_finish():
            runner._on_tool_finished(None, finish_ev)
            released.set()

        t_finish = threading.Thread(target=run_finish)
        t_finish.start()

        # Confirm Finished is blocked (no events yet).
        assert not released.wait(timeout=0.3), (
            "_on_tool_finished should be waiting on the CV, not emitting yet"
        )
        assert runner._sync_q.qsize() == 0

        # Now fire Started — Finished should unblock immediately.
        runner._on_tool_started(None, start_ev)
        assert released.wait(timeout=2.0), "Finished handler never released"
        t_finish.join(timeout=1.0)

        events = [e for e in _drain(runner)
                  if e["type"] in ("tool_call", "tool_result")]
        types = [(e["type"], e["tool"]) for e in events]
        assert types == [
            ("tool_call", "get_balance_sheet"),
            ("tool_result", "get_balance_sheet"),
        ], f"unexpected event ordering: {types}"

        # Both events must carry the same call_id.
        call_ev, result_ev = events
        assert call_ev["call_id"] and call_ev["call_id"] == result_ev["call_id"]
    finally:
        loop.close()


def test_parallel_tool_calls_orphan_free_and_ordered():
    """Three concurrent tool calls (the ZENTEC.NS fundamentals batch) must
    each produce a ``tool_call`` strictly before its ``tool_result``, with
    call_ids matched pairwise even when the finish threads race."""
    runner, loop = _fresh_runner()
    try:
        tools = ["get_balance_sheet", "get_cashflow", "get_income_statement"]

        # Race: schedule the 3 Finished handlers first, on 3 threads, so
        # they all wait on the CV.  Then fire the 3 Started handlers.
        finish_threads: list[threading.Thread] = []
        for tool in tools:
            fev = _finished_event(
                "Fundamentals Analyst", tool, {"ticker": "ZENTEC.NS"},
                output=f"# {tool} result", elapsed_ms=0,
            )
            t = threading.Thread(target=runner._on_tool_finished, args=(None, fev))
            t.start()
            finish_threads.append(t)

        # Give them time to reach ``self._tool_cv.wait(...)``.
        # 200 ms is plenty on the CI machines this project targets.
        threading.Event().wait(0.2)
        assert runner._sync_q.qsize() == 0, "Finished handlers must not emit yet"

        for tool in tools:
            sev = _started_event("Fundamentals Analyst", tool, {"ticker": "ZENTEC.NS"})
            runner._on_tool_started(None, sev)

        for t in finish_threads:
            t.join(timeout=3.0)
            assert not t.is_alive(), "Finished handler failed to release within 3s"

        events = _drain(runner)
        # For each tool, the tool_call index must be < tool_result index
        # and both call_ids must match.
        for tool in tools:
            call_idx, result_idx = _pair_indices(events, tool)
            assert call_idx is not None, f"missing tool_call for {tool}"
            assert result_idx is not None, f"missing tool_result for {tool}"
            assert call_idx < result_idx, (
                f"{tool}: tool_result (@{result_idx}) came before "
                f"tool_call (@{call_idx})"
            )
            assert events[call_idx]["call_id"] == events[result_idx]["call_id"]
    finally:
        loop.close()


def test_orphan_finished_emits_after_timeout():
    """If no Started ever arrives (regression from a badly-configured event
    bus) the Finished handler must still emit within its bounded window
    so the UI doesn't drop the tool_result silently."""
    runner, loop = _fresh_runner()
    try:
        # Shorten the wait so the test doesn't take 5 seconds.
        original = runner._wait_for_matching_call

        def short_wait(role, tool_name, *, timeout=5.0):  # noqa: ARG001
            return original(role, tool_name, timeout=0.2)

        runner._wait_for_matching_call = short_wait  # type: ignore[method-assign]

        finish_ev = _finished_event(
            "Fundamentals Analyst", "orphan_tool", {"x": 1},
            output="stray output", elapsed_ms=0,
        )
        runner._on_tool_finished(None, finish_ev)

        events = _drain(runner)
        assert len(events) == 1 and events[0]["type"] == "tool_result"
        # Orphan result must still be emitted (call_id can be None).
        assert events[0]["tool"] == "orphan_tool"
        assert events[0]["call_id"] is None
    finally:
        loop.close()
