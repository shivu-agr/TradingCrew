"""Integration test for the runner's degraded-output detection.

When the patch in ``trading_crew/_patches.py`` substitutes a tool-call
list with a placeholder prefixed by ``DEGRADED_OUTPUT_MARKER``, the
runner has to:

* Set ``degraded=True`` on the emitted ``node_completed`` WebSocket
  event so the UI can render the amber badge live.
* Track the affected role in ``_degraded_roles`` so the snapshot
  written to ``RunRecord`` for the Recent-Runs panel carries the
  signal too.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import List

import pytest

from trading_crew._patches import DEGRADED_OUTPUT_MARKER


def _fresh_runner():
    from web.backend.runner import AnalysisRunner

    loop = asyncio.new_event_loop()
    r = AnalysisRunner(loop)
    r._expected_role_order = [
        "Market Analyst",
        "News Analyst",
        "Bullish Researcher",
    ]
    r._expected_async_flags = [True, True, False]
    r._completed_slot_indices = set()
    r._running_roles = set(r._expected_role_order[:2])
    return r, loop


def _drain(runner) -> List[dict]:
    out: List[dict] = []
    while True:
        try:
            out.append(runner._sync_q.get_nowait())
        except Exception:
            break
    return out


def _task_output(role: str, raw: str):
    """Smallest object satisfying ``_agent_role_from_task_output`` + ``_safe_text``."""
    return SimpleNamespace(agent=role, raw=raw)


def test_degraded_marker_flips_event_flag_and_tracks_role():
    runner, loop = _fresh_runner()
    try:
        placeholder = f"{DEGRADED_OUTPUT_MARKER}\nAnalyst report missing"
        runner._on_task(_task_output("News Analyst", placeholder))

        events = [e for e in _drain(runner) if e.get("type") == "node_completed"]
        completed = next(e for e in events if e.get("node") == "News Analyst")
        assert completed["degraded"] is True
        assert completed["output"].startswith(DEGRADED_OUTPUT_MARKER)
        assert "News Analyst" in runner._degraded_roles
    finally:
        loop.close()


def test_clean_string_output_is_not_degraded():
    runner, loop = _fresh_runner()
    try:
        runner._on_task(
            _task_output("Market Analyst", "**MARKET REPORT** — Bullish")
        )

        events = [e for e in _drain(runner) if e.get("type") == "node_completed"]
        completed = next(e for e in events if e.get("node") == "Market Analyst")
        assert completed["degraded"] is False
        assert "Market Analyst" not in runner._degraded_roles
    finally:
        loop.close()


def test_degraded_role_persists_into_run_record():
    """The snapshotter on ``_emit`` must copy the degraded list into the
    saved ``RunRecord`` so the Recent-Runs panel shows the badge after
    a reload."""
    from trading_crew.agentic.runs import RunRecord

    runner, loop = _fresh_runner()
    try:
        runner._run_record = RunRecord(
            run_id="test",
            ticker="LT.NS",
            started_at="2026-01-01T00:00:00+00:00",
        )
        placeholder = f"{DEGRADED_OUTPUT_MARKER}\nAnalyst report missing"
        runner._on_task(_task_output("News Analyst", placeholder))
        # Emit happens synchronously inside _on_task → the snapshotter
        # has already been called by now.
        assert "News Analyst" in runner._run_record.degraded_roles
        assert runner._run_record.reports["News Analyst"].startswith(
            DEGRADED_OUTPUT_MARKER
        )
    finally:
        loop.close()
