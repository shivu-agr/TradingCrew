"""Unit tests for the ``PortfolioDecision`` action tier and legacy coercion.

The action alphabet moved from a 3-tier institutional label
(``OVERWEIGHT`` / ``NEUTRAL`` / ``UNDERWEIGHT``) to a retail-friendly
5-tier label (``STRONG_BUY`` / ``BUY`` / ``HOLD`` / ``SELL`` /
``STRONG_SELL``).  These tests pin down:

1. The new 5-tier is accepted end-to-end.
2. The legacy 3-tier still loads (older ``RunRecord`` JSONs on disk are
   expected to keep working) and gets normalised to the closest new
   tier.
3. Case / whitespace inputs are normalised.
4. Junk strings surface as a clean Pydantic ``ValidationError`` on the
   ``Literal`` — not a silent fallback.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trading_crew.horizons import Horizon
from trading_crew.schemas import (
    LEGACY_ACTION_MAPPING,
    PortfolioDecision,
    normalise_action,
)


def _base(**overrides) -> dict:
    """Minimal set of PortfolioDecision fields; tests override ``action``."""
    base = dict(
        action="BUY",
        confidence=0.6,
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
    return base


@pytest.mark.parametrize("tier", ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"])
def test_five_tier_actions_accepted(tier):
    """Every tier in the new 5-tier alphabet must validate cleanly."""
    d = PortfolioDecision(**_base(action=tier))
    assert d.action == tier


@pytest.mark.parametrize(
    "legacy,expected",
    [
        ("OVERWEIGHT", "BUY"),
        ("UNDERWEIGHT", "SELL"),
        ("NEUTRAL", "HOLD"),
    ],
)
def test_legacy_actions_are_coerced(legacy, expected):
    """Old RunRecord JSONs on disk must still load: the legacy triplet
    is silently rewritten to the closest 5-tier equivalent."""
    d = PortfolioDecision(**_base(action=legacy))
    assert d.action == expected
    assert LEGACY_ACTION_MAPPING[legacy] == expected


def test_case_and_whitespace_are_normalised():
    """Lower-case / stray-whitespace inputs (e.g. from a hand-typed
    scenario file) get normalised before Literal validation."""
    d = PortfolioDecision(**_base(action="  buy \n"))
    assert d.action == "BUY"


def test_unknown_action_still_raises_validation_error():
    """Junk strings must NOT silently fall back to a default — the
    caller needs a loud failure so they fix the input."""
    with pytest.raises(ValidationError):
        PortfolioDecision(**_base(action="MOON_BUY"))


def test_normalise_action_handles_non_string_defensively():
    """The module-level normaliser is used by other bridges (commodity
    crew, RL policy).  It must not crash on ``None`` / non-strings —
    the Pydantic layer will produce the actual ``ValidationError`` for
    those cases."""
    assert normalise_action(None) is None  # type: ignore[arg-type]
    assert normalise_action(123) == 123  # type: ignore[arg-type]


def test_hold_with_zero_size_is_valid():
    """HOLD ⇒ size_pct_of_book == 0 is the canonical sit-out shape."""
    d = PortfolioDecision(**_base(action="HOLD", size_pct_of_book=0.0))
    assert d.action == "HOLD"
    assert d.size_pct_of_book == 0.0


def test_strong_buy_high_conviction_shape_is_valid():
    """STRONG_BUY typical shape: size >= 2, confidence high — the schema
    itself doesn't enforce that (the guardrail does), but the tier must
    at least serialise cleanly at these values."""
    d = PortfolioDecision(**_base(
        action="STRONG_BUY", size_pct_of_book=3.0, confidence=0.82,
    ))
    dumped = d.model_dump()
    assert dumped["action"] == "STRONG_BUY"
    assert dumped["size_pct_of_book"] == 3.0


# ---------------------------------------------------------------------------
# Horizon field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "horizon,expected",
    [
        ("SHORT", Horizon.SHORT),
        ("MEDIUM", Horizon.MEDIUM),
        ("LONG", Horizon.LONG),
        ("short-term", Horizon.SHORT),
        ("long hold", Horizon.LONG),
        (Horizon.SHORT, Horizon.SHORT),
    ],
)
def test_horizon_field_accepts_and_coerces(horizon, expected):
    d = PortfolioDecision(**_base(horizon=horizon))
    assert d.horizon == expected


def test_horizon_defaults_to_medium_when_omitted():
    """Legacy RunRecord JSONs pre-date the ``horizon`` field entirely.
    They should still load and — when ``horizon_days`` sits in the
    MEDIUM band (61-252d) — the auto-derived horizon is MEDIUM.
    """
    d = PortfolioDecision(**_base(horizon_days=90))
    assert d.horizon == Horizon.MEDIUM


def test_legacy_horizon_days_backfills_short_tier():
    """A legacy record with horizon_days=21 must load and report
    ``horizon == SHORT`` even though the JSON didn't carry the field.
    """
    d = PortfolioDecision(**_base(horizon_days=21))
    assert d.horizon == Horizon.SHORT


def test_legacy_horizon_days_backfills_long_tier():
    """A legacy record with horizon_days=1260 must load and report
    ``horizon == LONG`` — the 5-year outcome window on a compounder
    thesis is exactly the LONG tier's shape.
    """
    d = PortfolioDecision(**_base(horizon_days=1260, expected_return_pct=1.8))
    assert d.horizon == Horizon.LONG


def test_long_horizon_carries_view_field():
    """LONG-tier decisions must populate the ``long_term_view`` field
    with a multi-year narrative.  For SHORT/MEDIUM decisions the field
    is still expected to be populated (with a shorter "thesis intact /
    broken / N/A" one-liner) so the desk's final report always shows
    the multi-year context — but the schema defaults to empty so
    legacy records still round-trip cleanly.
    """
    d_long = PortfolioDecision(**_base(
        action="STRONG_BUY",
        horizon="LONG",
        horizon_days=1260,
        long_term_view="10y revenue CAGR of 18%, ROCE compounding.",
    ))
    assert d_long.long_term_view.startswith("10y")

    d_medium = PortfolioDecision(**_base(
        horizon="MEDIUM",
        long_term_view="Multi-year thesis intact: 5y expectancy +2.1%.",
    ))
    assert d_medium.long_term_view.startswith("Multi-year thesis intact")


def test_long_term_thesis_alias_is_accepted_for_backwards_compat():
    """Persisted RunRecord JSONs written before the rename used
    ``long_term_thesis``.  The schema's ``_alias_long_term_thesis``
    validator maps the old key onto the new ``long_term_view`` field
    so those records still load without an on-disk migration."""
    d = PortfolioDecision(**_base(
        horizon="LONG",
        horizon_days=1260,
        long_term_thesis="Legacy multi-year narrative here.",
    ))
    assert d.long_term_view == "Legacy multi-year narrative here."


def test_long_term_view_wins_over_alias_when_both_provided():
    """If a caller (or a mixed pipeline) provides *both* keys, the new
    ``long_term_view`` value takes precedence — the alias validator only
    fills in when the new field is truly absent."""
    d = PortfolioDecision(**_base(
        horizon="LONG",
        horizon_days=1260,
        long_term_view="New narrative wins.",
        long_term_thesis="Legacy narrative that must be ignored.",
    ))
    assert d.long_term_view == "New narrative wins."


def test_long_term_view_defaults_to_empty_for_legacy_records():
    """A pre-rename JSON that carries neither field must still load —
    the field's default is the empty string, which the UI renders as
    'No long-term view was captured for this decision.'"""
    d = PortfolioDecision(**_base(horizon="MEDIUM"))
    assert d.long_term_view == ""
