"""Regression test for the analyst "ghost START" bug in ``AnalysisRunner``.

Background
----------
The 8 analyst tasks run with ``async_execution=True``.  CrewAI invokes our
``task_callback`` (``_on_task``) the moment each one finishes, in the
order they complete — which is NOT the order they are listed in
``crew.tasks``.  The old lookahead unconditionally re-emitted
``node_started`` for ``_expected_role_order[_completed_count]`` after
every completion, which during the async batch reliably pointed at a
sibling analyst that had already finished — producing "ghost" duplicate
START events for Macro / Sector / Quant in the UI timeline.

The fix in ``runner.py`` consults a parallel ``_expected_async_flags``
array and skips the lookahead whenever the next slot is async.  Async
slots self-announce via the CrewAI tool-use bus the moment they call
their first tool, so we never lose live-status coverage.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import List

import pytest


def _fresh_runner():
    """Build an ``AnalysisRunner`` bound to a private event loop so the
    test doesn't accidentally bleed into the pytest-asyncio default loop."""
    from web.backend.runner import AnalysisRunner

    loop = asyncio.new_event_loop()
    r = AnalysisRunner(loop)
    # Mirror what _run_blocking would have done at run start.
    r._expected_role_order = [
        # 8 analyst tasks, all async_execution=True (slots 0-7).
        "Market Analyst",
        "Social Analyst",
        "News Analyst",
        "Fundamentals Analyst",
        "Macro Analyst",
        "Geopolitical Analyst",
        "Sector / Peer Analyst",
        "Quant / Options Analyst",
        # First sequential task after the async block (slot 8).
        "Bullish Researcher",
    ]
    r._expected_async_flags = [True] * 8 + [False]
    r._completed_slot_indices = set()
    # Pretend every analyst has been announced via the tool-use bus.
    r._running_roles = set(r._expected_role_order[:8])
    return r, loop


def _drain(runner) -> List[dict]:
    """Pull every event currently buffered on the synchronous queue."""
    out: List[dict] = []
    while True:
        try:
            out.append(runner._sync_q.get_nowait())
        except Exception:
            break
    return out


def _fake_task_output(role: str):
    """Smallest object satisfying ``_agent_role_from_task_output``."""
    return SimpleNamespace(agent=role, raw="dummy")


def test_async_completions_do_not_emit_ghost_node_started():
    """Replay the exact out-of-order analyst completion timeline from
    terminals/267339.txt and assert that NO node_started events are
    emitted for an analyst slot whose role has already completed."""
    runner, loop = _fresh_runner()
    try:
        completion_order = [
            "Quant / Options Analyst",
            "Market Analyst",
            "Macro Analyst",
            "Fundamentals Analyst",
            "Sector / Peer Analyst",
            "News Analyst",
            "Geopolitical Analyst",
            "Social Analyst",
        ]
        for role in completion_order:
            runner._on_task(_fake_task_output(role))

        events = _drain(runner)
        started = [e for e in events if e.get("type") == "node_started"]
        completed = [e for e in events if e.get("type") == "node_completed"]

        # Every analyst must have produced exactly one completion event.
        assert sorted(e["role"] for e in completed) == sorted(completion_order)

        # The ONLY node_started emitted during the async batch should be
        # the lookahead announcement for the first sequential task, which
        # fires when the 8th (final) analyst completes.
        assert [e["role"] for e in started] == ["Bullish Researcher"], (
            f"Unexpected node_started events: {[e['role'] for e in started]}"
        )
    finally:
        loop.close()


def test_lookahead_announces_first_sequential_task_after_async_batch():
    """After all 8 async analysts complete, _completed_count == 8, and
    the lookahead must announce the next slot (Bullish Researcher,
    a sequential / tool-less agent that otherwise stays silent for
    30-60 s until its LLM call returns)."""
    runner, loop = _fresh_runner()
    try:
        for role in runner._expected_role_order[:8]:
            runner._on_task(_fake_task_output(role))

        events = _drain(runner)
        started_roles = [e["role"] for e in events if e.get("type") == "node_started"]

        assert started_roles == ["Bullish Researcher"]
    finally:
        loop.close()


def test_sequential_to_async_batch_fans_out_announcements():
    """Trader → Aggressive/Conservative/Neutral Risk is the canonical
    async-after-sequential pattern: risk tasks are tool-less and
    ``async_execution=True``, so the tool-use bus NEVER fires for them.
    The lookahead must pre-announce ALL three risk analysts the moment
    Trader finishes — otherwise an out-of-order completion (Conservative
    finishing before Aggressive) would emit a DONE without a prior
    START."""
    from web.backend.runner import AnalysisRunner

    loop = asyncio.new_event_loop()
    try:
        runner = AnalysisRunner(loop)
        runner._expected_role_order = [
            "Trader",  # sequential, slot 0
            "Aggressive Risk Analyst",  # async, slot 1
            "Conservative Risk Analyst",  # async, slot 2
            "Neutral Risk Analyst",  # async, slot 3
            "Compliance Officer",  # sequential, slot 4 — must NOT be pre-announced
        ]
        runner._expected_async_flags = [False, True, True, True, False]
        runner._completed_slot_indices = set()
        runner._running_roles = {"Trader"}

        runner._on_task(_fake_task_output("Trader"))
        events = _drain(runner)
        started_roles = [e["role"] for e in events if e.get("type") == "node_started"]
        # All 3 risk analysts get pre-announced; Compliance does NOT
        # (it's the sequential boundary that ends the async batch).
        assert started_roles == [
            "Aggressive Risk Analyst",
            "Conservative Risk Analyst",
            "Neutral Risk Analyst",
        ]

        # An out-of-order completion (Conservative finishes first) must
        # NOT re-announce any of the still-running siblings, and the
        # lookahead must NOT bleed forward into Compliance.
        runner._on_task(_fake_task_output("Conservative Risk Analyst"))
        events = _drain(runner)
        started_roles = [e["role"] for e in events if e.get("type") == "node_started"]
        assert started_roles == []
    finally:
        loop.close()


def test_lookahead_does_announce_sequential_after_sequential():
    """Two adjacent sequential slots (e.g. Research Manager → Quality
    Reviewer) — the second must be announced as soon as the first
    completes so the diagram doesn't sit idle while the next LLM
    spins up."""
    from web.backend.runner import AnalysisRunner

    loop = asyncio.new_event_loop()
    try:
        runner = AnalysisRunner(loop)
        runner._expected_role_order = ["Research Manager", "Quality Reviewer"]
        runner._expected_async_flags = [False, False]
        runner._completed_slot_indices = set()
        runner._running_roles = {"Research Manager"}

        runner._on_task(_fake_task_output("Research Manager"))

        events = _drain(runner)
        started_roles = [e["role"] for e in events if e.get("type") == "node_started"]
        assert started_roles == ["Quality Reviewer"]
    finally:
        loop.close()


def test_repeated_role_in_debate_rounds_announces_each_instance():
    """Debate rounds re-use the same role (e.g. ``Bullish Researcher``
    appears at slot 0 + slot 2 with a sequential Bearish in slot 1).
    Each completion must announce the NEXT instance — not regress to
    an earlier one."""
    from web.backend.runner import AnalysisRunner

    loop = asyncio.new_event_loop()
    try:
        runner = AnalysisRunner(loop)
        runner._expected_role_order = [
            "Bullish Researcher",   # slot 0
            "Bearish Researcher",   # slot 1
            "Bullish Researcher",   # slot 2 (round 2)
            "Bearish Researcher",   # slot 3 (round 2)
            "Research Manager",     # slot 4
        ]
        runner._expected_async_flags = [False] * 5
        runner._completed_slot_indices = set()
        runner._running_roles = {"Bullish Researcher"}

        runner._on_task(_fake_task_output("Bullish Researcher"))  # slot 0 done
        runner._on_task(_fake_task_output("Bearish Researcher"))  # slot 1 done
        runner._on_task(_fake_task_output("Bullish Researcher"))  # slot 2 done
        runner._on_task(_fake_task_output("Bearish Researcher"))  # slot 3 done

        events = _drain(runner)
        started_roles = [e["role"] for e in events if e.get("type") == "node_started"]
        # Each completion should announce the next instance.
        assert started_roles == [
            "Bearish Researcher",   # after slot 0 → next is slot 1
            "Bullish Researcher",   # after slot 1 → next is slot 2
            "Bearish Researcher",   # after slot 2 → next is slot 3
            "Research Manager",     # after slot 3 → next is slot 4
        ]
        assert runner._completed_slot_indices == {0, 1, 2, 3}
    finally:
        loop.close()
