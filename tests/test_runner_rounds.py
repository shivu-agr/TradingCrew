"""Regression tests for the multi-round persistence fix in ``AnalysisRunner``.

Background
----------
Before this fix every round of a multi-turn debate or risk-team
discussion (Bullish / Bearish Researcher, the three Risk personas)
was persisted under the SAME role key in ``RunRecord.reports`` — so
round 2 silently overwrote round 1 on disk and in the UI.  The user
saw "Bullish Researcher: <last round>" with no way to compare how
the argument evolved between rounds.

The fix in ``runner.py``:

* Pre-computes ``_role_round_per_slot`` (1-based round number for each
  slot in ``_expected_role_order``) at run start.
* Emits ``round`` + ``total_rounds`` on every ``node_started`` and
  ``node_completed`` event.
* Persists every round into ``RunRecord.reports_by_round[role]`` as a
  chronological list (the single ``reports`` dict still tracks the
  last round for backward-compatibility with downstream consumers).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import List

import pytest


def _make_runner_for_full_flow():
    """Build a runner whose ``_expected_role_order`` mirrors a real
    kickoff with ``debate_rounds=2`` and ``risk_rounds=2`` so the
    multi-round persistence is actually exercised."""
    from web.backend.runner import AnalysisRunner
    from trading_crew.agentic.runs import RunRecord

    loop = asyncio.new_event_loop()
    runner = AnalysisRunner(loop)

    runner._expected_role_order = [
        "Market Analyst", "Social Analyst", "News Analyst",
        "Fundamentals Analyst", "Macro Analyst", "Geopolitical Analyst",
        "Sector / Peer Analyst", "Quant / Options Analyst",
        # 2 debate rounds → bull/bear pair × 2.
        "Bullish Researcher", "Bearish Researcher",
        "Bullish Researcher", "Bearish Researcher",
        "Research Manager", "Quality Reviewer", "Trader",
        # 2 risk rounds → aggressive/conservative/neutral × 2.
        "Aggressive Risk Analyst", "Conservative Risk Analyst", "Neutral Risk Analyst",
        "Aggressive Risk Analyst", "Conservative Risk Analyst", "Neutral Risk Analyst",
        "Compliance Officer", "Portfolio Manager",
    ]
    runner._expected_async_flags = [False] * len(runner._expected_role_order)
    runner._completed_slot_indices = set()
    # Replicate the round / total-rounds pre-computation that
    # ``_run_blocking`` does at run start.
    role_seen: dict[str, int] = {}
    runner._role_round_per_slot = []
    for r in runner._expected_role_order:
        role_seen[r] = role_seen.get(r, 0) + 1
        runner._role_round_per_slot.append(role_seen[r])
    runner._role_total_rounds = {
        r: runner._expected_role_order.count(r)
        for r in set(runner._expected_role_order)
    }

    # Attach a fresh RunRecord so _snapshot_event can mirror events.
    runner._run_record = RunRecord(
        run_id="test", ticker="TEST", started_at="2026-06-30T00:00:00Z",
    )
    return runner, loop


def _fake_task_output(role: str, raw: str = "dummy"):
    return SimpleNamespace(agent=role, raw=raw)


def _drain(runner) -> List[dict]:
    out: List[dict] = []
    while True:
        try:
            out.append(runner._sync_q.get_nowait())
        except Exception:
            break
    return out


def test_role_round_per_slot_is_one_for_single_round_roles():
    """Analysts / manager / trader / compliance / PM all fire once
    per run so ``round_num`` must equal 1 for every one of their
    slots, with ``total_rounds == 1`` in the lookup table."""
    runner, loop = _make_runner_for_full_flow()
    try:
        for slot, role in enumerate(runner._expected_role_order):
            if runner._expected_role_order.count(role) == 1:
                assert runner._role_round_per_slot[slot] == 1, (
                    f"slot {slot} role {role!r} should be round 1, "
                    f"got {runner._role_round_per_slot[slot]}"
                )
                assert runner._role_total_rounds[role] == 1
    finally:
        loop.close()


def test_role_round_per_slot_increments_for_multi_round_roles():
    """The two Bullish / Bearish researcher slots and the three Risk
    persona slots each appear TWICE in expected_role_order — the
    second occurrence must come back as round 2."""
    runner, loop = _make_runner_for_full_flow()
    try:
        slots = list(enumerate(runner._expected_role_order))
        bull_rounds = [
            runner._role_round_per_slot[i]
            for i, r in slots if r == "Bullish Researcher"
        ]
        bear_rounds = [
            runner._role_round_per_slot[i]
            for i, r in slots if r == "Bearish Researcher"
        ]
        agg_rounds = [
            runner._role_round_per_slot[i]
            for i, r in slots if r == "Aggressive Risk Analyst"
        ]
        cons_rounds = [
            runner._role_round_per_slot[i]
            for i, r in slots if r == "Conservative Risk Analyst"
        ]
        neut_rounds = [
            runner._role_round_per_slot[i]
            for i, r in slots if r == "Neutral Risk Analyst"
        ]
        assert bull_rounds == [1, 2]
        assert bear_rounds == [1, 2]
        assert agg_rounds == [1, 2]
        assert cons_rounds == [1, 2]
        assert neut_rounds == [1, 2]
        for role in [
            "Bullish Researcher", "Bearish Researcher",
            "Aggressive Risk Analyst", "Conservative Risk Analyst",
            "Neutral Risk Analyst",
        ]:
            assert runner._role_total_rounds[role] == 2
    finally:
        loop.close()


def test_node_completed_event_carries_round_and_total_rounds():
    """Replay every analyst + 2 debate rounds and confirm the
    ``node_completed`` events emitted by ``_on_task`` carry the right
    ``round`` / ``total_rounds`` fields.  Without this the frontend
    can't tell round 1 from round 2 and the Reports tab collapses
    them back into a single card."""
    runner, loop = _make_runner_for_full_flow()
    try:
        # Fire every task in the configured order.
        for role in runner._expected_role_order:
            runner._on_task(_fake_task_output(role, raw=f"output of {role}"))

        events = _drain(runner)
        completed = [e for e in events if e.get("type") == "node_completed"]

        # Two bull entries: one round 1, one round 2.  Same for bear,
        # aggressive, conservative, neutral.
        for role in [
            "Bullish Researcher", "Bearish Researcher",
            "Aggressive Risk Analyst", "Conservative Risk Analyst",
            "Neutral Risk Analyst",
        ]:
            role_events = [e for e in completed if e["role"] == role]
            assert len(role_events) == 2
            assert role_events[0]["round"] == 1
            assert role_events[1]["round"] == 2
            assert role_events[0]["total_rounds"] == 2
            assert role_events[1]["total_rounds"] == 2

        # Single-round roles must report round=1 total=1.
        for role in ["Market Analyst", "Research Manager", "Trader",
                     "Portfolio Manager", "Compliance Officer"]:
            role_events = [e for e in completed if e["role"] == role]
            assert len(role_events) == 1
            assert role_events[0]["round"] == 1
            assert role_events[0]["total_rounds"] == 1
    finally:
        loop.close()


def test_reports_by_round_preserves_every_round_in_chronological_order():
    """Direct check on the persisted ``RunRecord.reports_by_round``:
    round 1 + round 2 must BOTH survive instead of round 2 silently
    overwriting round 1 (the original bug)."""
    runner, loop = _make_runner_for_full_flow()
    try:
        # Distinct raw text per round so the assertion has signal.
        round_label = {1: "first take", 2: "second take after rebuttal"}
        slot_round: List[int] = []
        for slot, role in enumerate(runner._expected_role_order):
            r = runner._role_round_per_slot[slot]
            slot_round.append(r)
            label = round_label.get(r, "single round")
            runner._on_task(_fake_task_output(role, raw=f"{role} :: {label}"))

        rec = runner._run_record
        # Bullish Researcher must hold 2 entries, with the first one
        # carrying "first take" text and the second "second take ...".
        bull_entries = rec.reports_by_round["Bullish Researcher"]
        assert len(bull_entries) == 2
        assert bull_entries[0]["round"] == 1
        assert bull_entries[0]["total_rounds"] == 2
        assert "first take" in bull_entries[0]["output"]
        assert bull_entries[1]["round"] == 2
        assert "second take" in bull_entries[1]["output"]

        # rec.reports[role] should still hold the LAST round's text
        # (back-compat for older consumers that only know the dict).
        assert "second take" in rec.reports["Bullish Researcher"]

        # Single-round roles must show exactly one entry.
        assert len(rec.reports_by_round["Market Analyst"]) == 1
        assert rec.reports_by_round["Market Analyst"][0]["round"] == 1
        assert rec.reports_by_round["Market Analyst"][0]["total_rounds"] == 1
    finally:
        loop.close()


def test_announce_emits_round_aware_node_started():
    """When ``_announce`` fires (manual or via the lookahead), the
    emitted ``node_started`` event must carry the round for the FIRST
    not-yet-completed slot whose role matches.  Without that the
    diagram subtitle can't render "round 2 of 2"."""
    runner, loop = _make_runner_for_full_flow()
    try:
        runner._running_roles = set()
        # Mark round 1 of Bullish Researcher complete (slot 8).
        runner._completed_slot_indices = {8}
        runner._announce("Bullish Researcher")
        events = _drain(runner)
        started = [e for e in events if e.get("type") == "node_started"]
        assert started, "expected one node_started event"
        assert started[0]["round"] == 2  # round 1 already completed → next is round 2
        assert started[0]["total_rounds"] == 2
    finally:
        loop.close()
