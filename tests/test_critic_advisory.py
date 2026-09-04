"""The Reflective Critic is advisory by default.

The critic is a one-directional operator (it can only hold / downgrade / cut
size / ABSTAIN, never upgrade), so letting it override the PM both flips the
decision at the last moment and biases the book toward HOLD/SELL.  In advisory
mode the audit still runs and is surfaced in the UI, but the PM's action/size
is KEPT — the critic only overrides on a hard integrity failure (compliance
BLOCKED or a fabricated source).
"""

from __future__ import annotations

from types import SimpleNamespace

from web.backend.runner import _critic_hard_fail_reason, _resolve_critic_outcome


def _decision(action="BUY", size=2.0, compliance="CLEAR", rationale="", sources=None):
    return SimpleNamespace(
        action=action,
        size_pct_of_book=size,
        compliance_status=compliance,
        rationale=rationale,
        sources=sources or [],
    )


def test_advisory_keeps_pm_decision_when_critic_downgrades_to_hold():
    pm = _decision(action="STRONG_BUY", size=2.5, rationale="Order book strong [source: screener.in ...]")
    critiqued = _decision(action="HOLD", size=0.0)  # critic collapsed it to abstain

    final, applied, reason = _resolve_critic_outcome(pm, critiqued, advisory=True)

    assert applied is False
    assert reason is None
    assert final is pm
    assert final.action == "STRONG_BUY"
    assert final.size_pct_of_book == 2.5


def test_advisory_overrides_on_blocked_compliance():
    pm = _decision(action="BUY", size=2.0, compliance="BLOCKED")
    critiqued = _decision(action="HOLD", size=0.0, compliance="BLOCKED")

    final, applied, reason = _resolve_critic_outcome(pm, critiqued, advisory=True)

    assert applied is True
    assert reason == "compliance BLOCKED"
    assert final is critiqued
    assert final.action == "HOLD"


def test_advisory_overrides_on_fabricated_source():
    pm = _decision(
        action="BUY",
        size=2.0,
        rationale="Revenue up 18% [source: https://example.com/made-up-report].",
    )
    critiqued = _decision(action="HOLD", size=0.0)

    reason = _critic_hard_fail_reason(pm)
    assert reason == "fabricated source citation"

    final, applied, _ = _resolve_critic_outcome(pm, critiqued, advisory=True)
    assert applied is True
    assert final is critiqued


def test_qualitative_judgement_is_not_a_hard_failure():
    # A well-labelled qualitative claim must NOT trip the fabricated-source
    # override — advisory mode should keep the PM's directional call.
    pm = _decision(
        action="BUY",
        size=2.0,
        rationale="The domestic capex cycle favours the sector [analyst judgement].",
    )
    critiqued = _decision(action="HOLD", size=0.0)

    assert _critic_hard_fail_reason(pm) is None
    final, applied, _ = _resolve_critic_outcome(pm, critiqued, advisory=True)
    assert applied is False
    assert final is pm


def test_non_advisory_restores_legacy_override():
    pm = _decision(action="STRONG_BUY", size=2.5)
    critiqued = _decision(action="HOLD", size=0.0)

    final, applied, reason = _resolve_critic_outcome(pm, critiqued, advisory=False)

    assert applied is True
    assert reason is None  # legacy override doesn't need a hard-fail reason
    assert final is critiqued
