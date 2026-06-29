"""M2 — CostModel correctness and sensitivity sweep."""

from __future__ import annotations

import math

import pytest

from trading_crew.agentic.execution.cost import (
    CostModel,
    COST_MODEL_LIBRARY,
    cost_sweep,
    get_cost_model,
)


# ---------------------------------------------------------------------------
# CostModel components
# ---------------------------------------------------------------------------


def test_fees_is_bps_of_notional_plus_flat():
    m = CostModel(fee_bps=2.0, flat_fee=5.0, half_spread_bps=0.0, impact_k=0.0)
    # 10,000 * 2 bps = 2.0 + 5 flat = 7.0
    assert m.fees(10_000) == pytest.approx(7.0)
    # Negative notional treated as abs — sell side pays the same fees
    assert m.fees(-10_000) == pytest.approx(7.0)


def test_spread_cost_is_half_spread_bps_of_notional():
    m = CostModel(fee_bps=0.0, flat_fee=0.0, half_spread_bps=4.0, impact_k=0.0)
    # 10,000 * 4 bps = 4.0
    assert m.spread_cost(10_000) == pytest.approx(4.0)


def test_impact_is_sqrt_of_participation():
    m = CostModel(fee_bps=0.0, flat_fee=0.0, half_spread_bps=0.0, impact_k=10.0)
    # k * sqrt(0.01) = 10 * 0.1 = 1 bps
    assert m.impact_bps(0.01) == pytest.approx(1.0)
    # k * sqrt(0.04) = 10 * 0.2 = 2 bps
    assert m.impact_bps(0.04) == pytest.approx(2.0)
    # k * sqrt(0.25) = 10 * 0.5 = 5 bps
    assert m.impact_bps(0.25) == pytest.approx(5.0)


def test_impact_at_zero_participation_is_zero():
    m = CostModel(fee_bps=0.0, flat_fee=0.0, half_spread_bps=0.0, impact_k=10.0)
    assert m.impact_bps(0.0) == 0.0
    assert m.impact_bps(-0.01) == 0.0  # defensive against caller bugs


def test_impact_is_monotonic_increasing_in_participation():
    m = CostModel(fee_bps=0.0, flat_fee=0.0, half_spread_bps=0.0, impact_k=10.0)
    prev = 0.0
    for p in [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]:
        cur = m.impact_bps(p)
        assert cur > prev, f"impact decreased at participation={p}"
        prev = cur


def test_total_cost_includes_all_three_components():
    m = CostModel(fee_bps=1.0, flat_fee=0.0, half_spread_bps=2.0, impact_k=10.0)
    br = m.total_cost(10_000, 0.01)
    assert br["fees"] == pytest.approx(1.0)
    assert br["spread"] == pytest.approx(2.0)
    # impact = 10 * sqrt(0.01) = 1 bps -> 1.0 on 10,000 notional
    assert br["impact"] == pytest.approx(1.0)
    assert br["total"] == pytest.approx(4.0)
    assert br["total_bps"] == pytest.approx(4.0)


def test_total_cost_with_zero_notional_returns_zero_total_bps():
    m = CostModel(fee_bps=1.0, flat_fee=0.0, half_spread_bps=2.0, impact_k=10.0)
    br = m.total_cost(0, 0.0)
    assert br["total"] == 0.0
    assert br["total_bps"] == 0.0


# ---------------------------------------------------------------------------
# Preset library
# ---------------------------------------------------------------------------


def test_preset_library_has_equity_and_futures_presets():
    # Equity presets are the original three; futures presets were added
    # alongside the commodity dashboard.
    equity = {"low", "standard", "high"}
    futures = {"futures_low", "futures_standard", "futures_high"}
    assert equity.issubset(set(COST_MODEL_LIBRARY))
    assert futures.issubset(set(COST_MODEL_LIBRARY))


def test_get_cost_model_raises_on_unknown_name():
    with pytest.raises(KeyError, match="Unknown cost model"):
        get_cost_model("non_existent_preset")


def test_high_cost_preset_is_strictly_higher_than_low():
    low = get_cost_model("low").total_cost(10_000, 0.01)
    high = get_cost_model("high").total_cost(10_000, 0.01)
    assert high["total"] > low["total"]


# ---------------------------------------------------------------------------
# Sensitivity sweep
# ---------------------------------------------------------------------------


def test_cost_sweep_produces_scenario_for_each_preset():
    scenarios = cost_sweep(10_000, 0.01)
    labels = [s.label for s in scenarios]
    assert labels == ["low", "standard", "high"]
    for s in scenarios:
        assert s.cost_breakdown["total"] >= 0
        assert s.notional == 10_000
        assert s.participation == 0.01


def test_cost_sweep_total_increases_monotonically_with_friction():
    scenarios = cost_sweep(10_000, 0.05)
    totals = [s.cost_breakdown["total"] for s in scenarios]
    assert totals == sorted(totals), f"Cost sweep should be monotonic; got {totals}"


def test_cost_sweep_total_bps_matches_total_per_notional():
    scenarios = cost_sweep(10_000, 0.01)
    for s in scenarios:
        expected_bps = s.cost_breakdown["total"] / 10_000 * 1e4
        assert s.cost_breakdown["total_bps"] == pytest.approx(expected_bps)
