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


def test_pm_regime_injection_prepends_once_and_skips_unknown():
    """The deterministic regime is handed straight to the PM prompt: prepended
    exactly once, never for UNKNOWN, and never when there's no PM handle."""
    runner, loop = _fresh_runner()
    try:
        base = "You are the Portfolio Manager. Make the final call."
        runner._pm_task = SimpleNamespace(description=base)
        runner._pm_regime_injected = False

        runner._inject_pm_regime("TREND")
        assert runner._pm_regime_injected is True
        assert "MARKET REGIME" in runner._pm_task.description
        assert "TREND" in runner._pm_task.description
        assert runner._pm_task.description.endswith(base)

        # Idempotent — a second call must not stack a second block.
        runner._inject_pm_regime("RANGE")
        assert runner._pm_task.description.count("MARKET REGIME") == 1

        # UNKNOWN / missing regime is not injected (no signal → no noise).
        runner2, loop2 = _fresh_runner()
        try:
            runner2._pm_task = SimpleNamespace(description=base)
            runner2._pm_regime_injected = False
            runner2._inject_pm_regime("UNKNOWN")
            runner2._inject_pm_regime(None)
            assert "MARKET REGIME" not in runner2._pm_task.description
            assert runner2._pm_regime_injected is False
        finally:
            loop2.close()

        # No PM handle → no-op (deep-link / attach-only sessions).
        runner3, loop3 = _fresh_runner()
        try:
            runner3._pm_task = None
            runner3._pm_regime_injected = False
            runner3._inject_pm_regime("TREND")  # must not raise
            assert runner3._pm_regime_injected is False
        finally:
            loop3.close()
    finally:
        loop.close()


def test_parse_desk_bias_classifies_tldr():
    from web.backend.runner import _parse_desk_bias

    assert _parse_desk_bias("TLDR:\n- up [source: x]\n- Bias: bullish\nbody") == "bullish"
    assert _parse_desk_bias("- Bias: bearish / cautious") == "bearish"
    assert _parse_desk_bias("Bias: neutral") == "neutral"
    assert _parse_desk_bias("- Bias: mixed") == "neutral"
    assert _parse_desk_bias("- Bias: positive, long thesis") == "bullish"
    # Markdown-bold leaders the models actually emit (regression: these silently
    # failed to parse at first, rendering an empty board on live runs).
    assert _parse_desk_bias("- **Bias:** Slightly bearish.") == "bearish"
    assert _parse_desk_bias("- **Bias:** Bullish on the chart, options mixed") == "bullish"
    assert _parse_desk_bias("  - **Bias**: mixed signals") == "neutral"
    assert _parse_desk_bias("* Bias - neutral") == "neutral"
    assert _parse_desk_bias("- **Bias:** slight negative due to insider sales.") == "bearish"
    # One safe qualifier word before "bias" is tolerated (Fundamentals writes
    # "Long-term bias:", others "Overall/Net bias:").
    assert _parse_desk_bias("- **Long-term bias:** stable cash-flow, debt reduction") == "neutral"
    assert _parse_desk_bias("- Overall bias: bearish") == "bearish"
    # Must NOT latch onto the reports desk's NET_GUIDANCE_BIAS: line or prose.
    assert _parse_desk_bias("NET_GUIDANCE_BIAS: NEUTRAL (confidence 0.55)") is None
    assert _parse_desk_bias("net positive bias.") is None
    assert _parse_desk_bias("reflecting a market-wide bias toward downside moves") is None
    # Canonical machine tag (the reliable primary source) is honoured and wins
    # over the fuzzy prose line when both are present.
    assert _parse_desk_bias("body...\nDESK_BIAS: BULLISH") == "bullish"
    assert _parse_desk_bias("- DESK_BIAS: NEUTRAL") == "neutral"
    assert _parse_desk_bias("**DESK_BIAS:** BEARISH") == "bearish"
    assert _parse_desk_bias("- **Bias:** bearish on chart\n\nDESK_BIAS: BULLISH") == "bullish"
    # No Bias line → nothing to surface (better than a fabricated lean).
    assert _parse_desk_bias("just some prose, no verdict") is None
    assert _parse_desk_bias("") is None


def test_pm_bias_board_built_from_captured_desks(monkeypatch):
    """Board renders one row per captured desk in canonical lens order, carries
    the verdicts, and is explicitly non-directive."""
    monkeypatch.setenv("TRADINGCREW_PM_DESK_BIAS_BOARD", "1")
    runner, loop = _fresh_runner()
    try:
        runner._desk_biases = {
            "Reports Analyst": "bullish",
            "Market Analyst": "bearish",
            "Macro Analyst": "neutral",
        }
        board = runner._build_bias_board()
        # Canonical order: Market before Reports before Macro.
        assert board.index("Market") < board.index("Reports/Guidance") < board.index("Macro")
        assert "bullish" in board and "bearish" in board and "neutral" in board
        # Non-directive on side, but disagreement is binding (HOLD on split).
        assert "does not prescribe" in board.lower()
        assert "HOLD" in board
        assert "Neutrals do not create a split" in board
    finally:
        loop.close()


def test_pm_bias_board_injects_once_via_pm_announce(monkeypatch):
    """The board is handed to the PM exactly when its node is announced, once,
    and only when the env flag is on."""
    monkeypatch.setenv("TRADINGCREW_PM_DESK_BIAS_BOARD", "1")
    runner, loop = _fresh_runner()
    try:
        base = "You are the Portfolio Manager. Decide."
        runner._pm_task = SimpleNamespace(description=base)
        runner._desk_biases = {"Market Analyst": "bullish", "News Analyst": "neutral"}

        # A non-PM announce must NOT inject.
        runner._announce("Bullish Researcher")
        assert "DESK BIAS BOARD" not in runner._pm_task.description

        # Announcing the PM injects the board, once, ending with the base.
        runner._announce("Portfolio Manager")
        assert runner._pm_bias_board_injected is True
        assert runner._pm_task.description.count("DESK BIAS BOARD") == 1
        assert runner._pm_task.description.endswith(base)

        # …and it surfaces a desk_bias_board event for the dashboard.
        events = _drain(runner)
        board_evs = [e for e in events if e.get("type") == "desk_bias_board"]
        assert len(board_evs) == 1
        desks = {d["desk"]: d["bias"] for d in board_evs[0]["desks"]}
        assert desks == {"Market": "bullish", "News": "neutral"}
    finally:
        loop.close()


def test_pm_bias_board_disabled_by_env(monkeypatch):
    monkeypatch.setenv("TRADINGCREW_PM_DESK_BIAS_BOARD", "0")
    runner, loop = _fresh_runner()
    try:
        runner._pm_task = SimpleNamespace(description="base")
        runner._desk_biases = {"Market Analyst": "bullish"}
        runner._inject_pm_bias_board()
        assert runner._pm_bias_board_injected is False
        assert "DESK BIAS BOARD" not in runner._pm_task.description
    finally:
        loop.close()


def test_pm_bias_board_empty_and_no_pm_are_noops(monkeypatch):
    monkeypatch.setenv("TRADINGCREW_PM_DESK_BIAS_BOARD", "1")
    runner, loop = _fresh_runner()
    try:
        # No desks captured → empty board, no injection.
        runner._pm_task = SimpleNamespace(description="base")
        runner._desk_biases = {}
        runner._inject_pm_bias_board()
        assert runner._pm_bias_board_injected is False
        assert runner._build_bias_board() == ""

        # No PM handle → silent no-op.
        runner._pm_task = None
        runner._desk_biases = {"Market Analyst": "bullish"}
        runner._inject_pm_bias_board()  # must not raise
        assert runner._pm_bias_board_injected is False
    finally:
        loop.close()


def test_pm_bias_board_build_is_cheap():
    """No I/O — building the board 5k times must be well under a second, so the
    feature adds no meaningful wall-clock to a run."""
    import time as _time

    runner, loop = _fresh_runner()
    try:
        runner._desk_biases = {
            "Market Analyst": "bullish", "News Analyst": "neutral",
            "Fundamentals Analyst": "bullish", "Reports Analyst": "bearish",
            "Macro Analyst": "neutral",
        }
        t0 = _time.perf_counter()
        for _ in range(5000):
            runner._build_bias_board()
        assert _time.perf_counter() - t0 < 1.0
    finally:
        loop.close()
