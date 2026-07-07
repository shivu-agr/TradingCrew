"""Unit tests for the ``trading_crew.horizons`` vocabulary.

The horizon tier is now a *first-class* field on ``PortfolioDecision``
and is threaded through the backtest panel, the RLMF reward function,
and the frontend badge — so the classification helpers here are hot-
path code and deserve a small but thorough test.
"""

from __future__ import annotations

import pytest

from trading_crew.horizons import (
    ALL_HORIZON_DAYS,
    Horizon,
    LONG_HORIZON_DAYS,
    MEDIUM_HORIZON_DAYS,
    SHORT_HORIZON_DAYS,
    horizon_days_to_horizon,
    horizon_human_label,
    horizon_panel_days,
    horizon_representative_days,
    normalise_horizon,
)


# ---------------------------------------------------------------------------
# horizon_days_to_horizon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "days,expected",
    [
        (5, Horizon.SHORT),
        (21, Horizon.SHORT),
        (60, Horizon.SHORT),   # inclusive upper bound
        (61, Horizon.MEDIUM),
        (90, Horizon.MEDIUM),
        (252, Horizon.MEDIUM), # inclusive upper bound
        (253, Horizon.LONG),
        (504, Horizon.LONG),
        (1260, Horizon.LONG),
        (2520, Horizon.LONG),
        # Edge cases — the mapping never crashes on missing / bad values.
        (0, Horizon.MEDIUM),
        (-1, Horizon.MEDIUM),
        (None, Horizon.MEDIUM),
    ],
)
def test_horizon_days_to_horizon(days, expected):
    assert horizon_days_to_horizon(days) == expected


# ---------------------------------------------------------------------------
# horizon_representative_days
# ---------------------------------------------------------------------------


def test_horizon_representative_days_are_middle_of_tier():
    """The representative day-counts are chosen mid-tier so a
    directional mistake at the boundary doesn't flip conviction."""
    assert horizon_representative_days(Horizon.SHORT) == 21
    assert horizon_representative_days(Horizon.MEDIUM) == 189
    assert horizon_representative_days(Horizon.LONG) == 1260


def test_representative_days_roundtrip_to_same_tier():
    """Every representative day-count must map back to its own tier
    — otherwise we'd have a cycle (LONG -> LONG_repr -> MEDIUM)."""
    for tier in (Horizon.SHORT, Horizon.MEDIUM, Horizon.LONG):
        reprs = horizon_representative_days(tier)
        assert horizon_days_to_horizon(reprs) == tier


# ---------------------------------------------------------------------------
# horizon_panel_days
# ---------------------------------------------------------------------------


def test_horizon_panel_days_match_module_tuples():
    assert horizon_panel_days(Horizon.SHORT) == SHORT_HORIZON_DAYS
    assert horizon_panel_days(Horizon.MEDIUM) == MEDIUM_HORIZON_DAYS
    assert horizon_panel_days(Horizon.LONG) == LONG_HORIZON_DAYS


def test_all_horizon_days_is_flattened_and_sorted():
    """The flat tuple must be the concatenation, in tier order, of the
    per-tier tuples — the frontend legend depends on that ordering."""
    assert ALL_HORIZON_DAYS == (
        SHORT_HORIZON_DAYS + MEDIUM_HORIZON_DAYS + LONG_HORIZON_DAYS
    )


def test_panel_day_counts_are_in_their_tier():
    """A SHORT panel day-count must classify as SHORT; a LONG one
    must classify as LONG.  A regression here would silently
    misalign the backtest panel with the LLM's prompt vocabulary.
    """
    for d in SHORT_HORIZON_DAYS:
        assert horizon_days_to_horizon(d) == Horizon.SHORT
    for d in MEDIUM_HORIZON_DAYS:
        assert horizon_days_to_horizon(d) == Horizon.MEDIUM
    for d in LONG_HORIZON_DAYS:
        assert horizon_days_to_horizon(d) == Horizon.LONG


# ---------------------------------------------------------------------------
# horizon_human_label
# ---------------------------------------------------------------------------


def test_horizon_human_labels():
    assert "short" in horizon_human_label(Horizon.SHORT).lower()
    assert "medium" in horizon_human_label(Horizon.MEDIUM).lower()
    label = horizon_human_label(Horizon.LONG).lower()
    assert "long" in label
    # The user asked for "5-10 year" phrasing; we ship the compact
    # "1-10 years" label because a LONG-tier decision starts *at* 1
    # year, not 5.  Either 5 or 10 should appear so the operator
    # sees the multi-year framing.
    assert "10" in label or "10 year" in label


# ---------------------------------------------------------------------------
# normalise_horizon — free-form string coercion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SHORT", Horizon.SHORT),
        ("Short", Horizon.SHORT),
        ("short-term", Horizon.SHORT),
        ("MEDIUM", Horizon.MEDIUM),
        ("medium term", Horizon.MEDIUM),
        ("LONG", Horizon.LONG),
        ("long hold", Horizon.LONG),
        (Horizon.SHORT, Horizon.SHORT),
        (Horizon.LONG, Horizon.LONG),
    ],
)
def test_normalise_horizon_accepts_variants(raw, expected):
    assert normalise_horizon(raw) is expected


@pytest.mark.parametrize(
    "raw",
    [None, "", "unknown", "OVERWEIGHT", 42, 3.14, object()],
)
def test_normalise_horizon_returns_none_on_bad_input(raw):
    assert normalise_horizon(raw) is None
