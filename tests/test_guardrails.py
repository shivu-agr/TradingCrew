"""Unit tests for the Portfolio Manager confidence guardrail.

The guardrail rejects PM outputs that are self-inconsistent or
implausibly confident.  These tests focus on the 5-tier action alphabet
introduced alongside the retail-friendly PortfolioDecision refactor —
specifically the "HOLD ⇔ size==0" and "directional tier ⇒ size>0"
invariants that the guardrail is the last line of defence for.
"""

from __future__ import annotations

from types import SimpleNamespace

from trading_crew.guardrails import confidence_guardrail
from trading_crew.schemas import PortfolioDecision


def _decision(**overrides) -> PortfolioDecision:
    base = dict(
        action="BUY",
        confidence=0.65,
        size_pct_of_book=1.5,
        entry_price=100.0,
        stop_loss=92.0,
        target_price=118.0,
        horizon_days=90,
        expected_return_pct=0.18,
        rationale="Mid-term thesis [source: r].",
        key_drivers=["a"],
        key_risks=["b"],
        falsifiers=["c"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
    )
    base.update(overrides)
    return PortfolioDecision(**base)


def _run(decision):
    return confidence_guardrail(SimpleNamespace(pydantic=decision))


def test_valid_buy_passes():
    ok, out = _run(_decision(action="BUY", size_pct_of_book=1.5))
    assert ok is True
    assert out is not None


def test_valid_strong_buy_passes():
    ok, out = _run(_decision(action="STRONG_BUY", size_pct_of_book=2.5, confidence=0.75))
    assert ok is True


def test_valid_hold_with_zero_size_passes():
    ok, out = _run(_decision(action="HOLD", size_pct_of_book=0.0))
    assert ok is True


def test_hold_with_nonzero_size_rejected():
    """A HOLD with positive size is contradictory — the guardrail must
    catch it because both the M1 bridge and the UI badge would render
    incoherent state."""
    ok, msg = _run(_decision(action="HOLD", size_pct_of_book=1.0))
    assert ok is False
    assert "HOLD" in msg
    assert "size_pct_of_book" in msg


def test_directional_tier_with_zero_size_rejected():
    """A BUY with size=0 is a HOLD in disguise; force the PM to pick
    the right tier so downstream code can trust the label."""
    ok, msg = _run(_decision(action="BUY", size_pct_of_book=0.0))
    assert ok is False
    assert "BUY" in msg


def test_blocked_compliance_requires_hold_tier():
    """BLOCKED compliance ⇒ action MUST be HOLD (formerly NEUTRAL)."""
    ok, msg = _run(_decision(
        action="BUY", size_pct_of_book=1.0, compliance_status="BLOCKED",
    ))
    assert ok is False
    assert "BLOCKED" in msg
    assert "HOLD" in msg


def test_blocked_compliance_with_hold_and_zero_size_passes():
    ok, _ = _run(_decision(
        action="HOLD", size_pct_of_book=0.0, compliance_status="BLOCKED",
    ))
    assert ok is True


def test_high_confidence_directional_rejected():
    """Overconfident single-name trades are rejected regardless of tier."""
    ok, msg = _run(_decision(
        action="STRONG_BUY", size_pct_of_book=2.5, confidence=0.95,
    ))
    assert ok is False
    assert "confidence" in msg


def test_high_confidence_hold_is_allowed():
    """A high-confidence HOLD (e.g. "the evidence is genuinely mixed")
    is a legitimate call — the overconfidence check only applies to
    directional tiers."""
    ok, _ = _run(_decision(action="HOLD", size_pct_of_book=0.0, confidence=0.9))
    assert ok is True


def test_oversized_low_confidence_rejected():
    """A large size with low confidence should get bounced back for
    resizing regardless of the exact action tier."""
    ok, msg = _run(_decision(
        action="BUY", size_pct_of_book=6.0, confidence=0.5,
    ))
    assert ok is False
    assert "size_pct_of_book" in msg


def test_legacy_neutral_input_coerces_and_passes_as_hold():
    """A caller who still emits the legacy ``NEUTRAL`` action string
    (e.g. an older cached PM output re-fed into the guardrail) should
    have it silently normalised to ``HOLD`` by the schema validator,
    which the guardrail then accepts.  Guards against a regression
    where the guardrail special-cased the exact 5-tier strings and
    rejected everything else."""
    ok, out = _run(_decision(action="NEUTRAL", size_pct_of_book=0.0))
    assert ok is True
    assert out.action == "HOLD"


def test_guardrail_message_lists_valid_tiers_when_output_missing():
    """When ``task_output.pydantic`` is ``None`` the guardrail's retry
    message must enumerate the 5-tier alphabet so the LLM knows what
    to emit."""
    ok, msg = confidence_guardrail(SimpleNamespace(pydantic=None))
    assert ok is False
    for tier in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"):
        assert tier in msg, f"retry message must mention {tier}"
