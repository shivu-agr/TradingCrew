"""Tests for the commodity_crew package + futures cost model + bridge."""

from __future__ import annotations

import pytest

from commodity_crew.bridge import futures_decision_to_portfolio_decision
from commodity_crew.schemas import FuturesDecision
from commodity_crew.tools import (
    COMMODITY_META,
    _futures_chain_symbols,
    _meta_for,
    _root_of,
    ALL_TOOLS,
    DEFAULT_AGENT_TOOLS,
    get_tool_catalog,
)
from commodity_crew.crew import get_agent_catalog
from trading_crew.agentic.execution.cost import (
    COST_MODEL_LIBRARY,
    get_cost_model,
    roll_yield_carry_cost,
)


# ---------------------------------------------------------------------------
# Root extraction + meta lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_sym,expected_root", [
    ("CL=F", "CL"),
    ("CL", "CL"),
    ("CLN26.NYM", "CL"),
    ("GC=F", "GC"),
    ("ZC=F", "ZC"),
    ("ZSN26.CBT", "ZS"),
    ("HG=F", "HG"),
])
def test_root_of_strips_yfinance_suffixes(input_sym, expected_root):
    assert _root_of(input_sym) == expected_root


def test_meta_for_known_symbol_returns_full_record():
    meta = _meta_for("CL=F")
    assert meta["name"] == "WTI Crude Oil"
    assert meta["contract_size"] == 1000.0
    assert "cftc_name" in meta


def test_meta_for_unknown_symbol_returns_defaults():
    meta = _meta_for("XYZ=F")
    assert meta["name"] == "XYZ=F"
    assert meta["contract_size"] == 1.0


def test_all_commodities_have_cftc_mapping():
    """Every preset commodity should have a CFTC name so the COT tool
    works out of the box.  Failing this means a UI ticker preset would
    produce a 'no CFTC mapping' error."""
    for root, meta in COMMODITY_META.items():
        assert meta.get("cftc_name"), f"{root} missing cftc_name"
        assert meta.get("name")
        assert meta.get("contract_size", 0) > 0


# ---------------------------------------------------------------------------
# Futures chain symbol generator
# ---------------------------------------------------------------------------


def test_futures_chain_returns_n_months():
    chain = _futures_chain_symbols("CL", n_months=6)
    assert len(chain) == 6
    # Each symbol should start with the root and end with the exchange.
    for sym in chain:
        assert sym.startswith("CL")
        assert sym.endswith(".NYM")


def test_futures_chain_uses_correct_exchange_for_grains():
    chain = _futures_chain_symbols("ZC", n_months=3)
    for sym in chain:
        assert sym.endswith(".CBT")


def test_futures_chain_uses_comex_for_metals():
    chain = _futures_chain_symbols("GC", n_months=3)
    for sym in chain:
        assert sym.endswith(".CMX")


# ---------------------------------------------------------------------------
# Catalogs
# ---------------------------------------------------------------------------


def test_get_agent_catalog_has_full_debate_lineup():
    """Catalog must expose every persona the dashboard sidebar / diagram
    expects: 7 analysts + 3 researchers + QA + trader + 3 risk + compliance
    + PM = 17 agents."""
    catalog = get_agent_catalog()
    assert len(catalog) == 17
    roles = {a["role"] for a in catalog}
    # analyst tier
    assert "Market Analyst" in roles
    # researcher tier
    assert "Bullish Researcher" in roles
    assert "Bearish Researcher" in roles
    assert "Research Manager" in roles
    # QA + trader
    assert "Quality Reviewer" in roles
    assert "Futures Trader" in roles
    # risk + compliance + PM
    assert "Aggressive Risk Analyst" in roles
    assert "Neutral Risk Analyst" in roles
    assert "Conservative Risk Analyst" in roles
    assert "Compliance Officer" in roles
    assert "Portfolio Manager" in roles


def test_get_tool_catalog_lists_all_commodity_tools():
    cat = get_tool_catalog()
    expected = {
        "get_commodity_ohlcv", "get_commodity_indicators",
        "get_futures_curve", "get_seasonality", "get_cot_report",
        "get_commodity_news", "get_commodity_geopolitical",
        "retrieve_past_episodes_commodity",
        # L4 RL advisor — same shape as the equity tool, just namespaced.
        "rl_policy_recommendation_commodity",
    }
    assert set(cat.keys()) == expected


def test_default_agent_tools_only_reference_known_tools():
    """Every tool name appearing in DEFAULT_AGENT_TOOLS must exist in
    ALL_TOOLS — otherwise CrewAI silently drops the tool at runtime
    and the agent goes tool-less.
    """
    for agent_key, tools in DEFAULT_AGENT_TOOLS.items():
        for t in tools:
            assert t in ALL_TOOLS, f"{agent_key} references unknown tool {t}"


# ---------------------------------------------------------------------------
# FuturesDecision -> PortfolioDecision bridge
# ---------------------------------------------------------------------------


def _futures_decision(**overrides) -> FuturesDecision:
    base = dict(
        action="LONG", confidence=0.7, size_pct_of_book=2.0,
        entry_price=70.0, stop_loss=68.0, target_price=80.0,
        horizon_days=30, expected_return_pct=0.14,
        contract_month="2026-09", contract_size=1000.0,
        curve_view="BACKWARDATION", roll_yield_pct_annualised=2.5,
        rationale="A" * 50, key_drivers=["a", "b"],
        key_risks=["risk1"], falsifiers=["f1"],
    )
    base.update(overrides)
    return FuturesDecision(**base)


def test_bridge_maps_long_to_overweight():
    pd = futures_decision_to_portfolio_decision(_futures_decision(action="LONG"))
    assert pd.action == "OVERWEIGHT"


def test_bridge_maps_short_to_underweight():
    pd = futures_decision_to_portfolio_decision(_futures_decision(action="SHORT"))
    assert pd.action == "UNDERWEIGHT"


def test_bridge_maps_neutral_to_neutral():
    pd = futures_decision_to_portfolio_decision(_futures_decision(action="NEUTRAL"))
    assert pd.action == "NEUTRAL"


def test_bridge_preserves_futures_context_in_rationale():
    fd = _futures_decision(curve_view="CONTANGO", roll_yield_pct_annualised=-3.5)
    pd = futures_decision_to_portfolio_decision(fd)
    assert "2026-09" in pd.rationale
    assert "CONTANGO" in pd.rationale
    assert "-3.50%" in pd.rationale


def test_bridge_adds_futures_drivers_as_tags():
    fd = _futures_decision()
    pd = futures_decision_to_portfolio_decision(fd)
    driver_set = set(pd.key_drivers)
    assert any("contract_month:2026-09" in d for d in driver_set)
    assert any("curve_view:BACKWARDATION" in d for d in driver_set)


def test_bridge_omits_tiny_roll_yield_tag():
    """Roll yields under 0.1%/y are noise — they shouldn't bloat key_drivers."""
    fd = _futures_decision(roll_yield_pct_annualised=0.05)
    pd = futures_decision_to_portfolio_decision(fd)
    assert not any(d.startswith("roll_yield:") for d in pd.key_drivers)


def test_bridge_forwards_sources_unchanged():
    """The bridge must carry the FuturesDecision.sources list onto the
    PortfolioDecision so the equity-side Reflective Critic sees the same
    provenance trail."""
    fd = _futures_decision(
        sources=[
            "yfinance futures OHLCV CL=F (6mo, indicators)",
            "yfinance futures curve CL (next 12 contracts)",
            "CFTC COT https://www.cftc.gov/files/dea/history/deacot2026.zip (WTI FINANCIAL CRUDE OIL)",
        ],
    )
    pd = futures_decision_to_portfolio_decision(fd)
    assert pd.sources == fd.sources


def test_bridge_handles_missing_sources_field():
    """A FuturesDecision built without ``sources`` (legacy / abstain
    paths) must still bridge cleanly — sources collapses to an empty
    list rather than raising."""
    fd = _futures_decision()
    assert fd.sources == []
    pd = futures_decision_to_portfolio_decision(fd)
    assert pd.sources == []


# ---------------------------------------------------------------------------
# Tool provenance footers — every commodity tool must end its body with a
# ``Source: …`` line so analysts can copy the identifier into an inline
# ``[source: <identifier>]`` tag.
# ---------------------------------------------------------------------------


def test_source_line_helper_includes_identifier_and_utc_timestamp():
    """The shared helper writes a single-line, parseable footer."""
    import re
    from commodity_crew.tools import _source_line

    line = _source_line("yfinance futures curve CL (next 12 contracts)")
    assert line.startswith("\nSource: ")
    assert "yfinance futures curve CL (next 12 contracts)" in line
    # ISO-8601 minute-resolution UTC tag, e.g. 2026-06-12T14:32Z
    assert re.search(r"retrieved \d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z", line)


def test_source_line_strips_square_brackets_to_keep_tags_parseable():
    """If an identifier contained square brackets the [source: …] tag
    would be ambiguous, so the helper rewrites them to round parens."""
    from commodity_crew.tools import _source_line

    line = _source_line("vendor [internal] feed")
    assert "[" not in line
    assert "(internal)" in line


# ---------------------------------------------------------------------------
# Futures cost-model presets + roll-yield carry helper
# ---------------------------------------------------------------------------


def test_futures_cost_presets_registered():
    assert "futures_low" in COST_MODEL_LIBRARY
    assert "futures_standard" in COST_MODEL_LIBRARY
    assert "futures_high" in COST_MODEL_LIBRARY


def test_futures_low_cheaper_than_equity_standard():
    """Index futures should be measurably cheaper than the equity standard
    preset — otherwise the presets don't reflect futures economics."""
    f_low = get_cost_model("futures_low")
    eq_std = get_cost_model("standard")
    cost_f = f_low.total_cost(notional=100_000, participation=0.01)
    cost_e = eq_std.total_cost(notional=100_000, participation=0.01)
    assert cost_f["total"] < cost_e["total"]


def test_roll_yield_contango_costs_long():
    """Long position in a contango (negative annualised roll) market pays carry."""
    carry = roll_yield_carry_cost(
        notional=100_000.0,
        annualised_roll_yield_pct=-6.0,   # 6% annualised contango drag
        holding_days=30,
    )
    # ~ -100000 * (-0.06/365) * 30 = +492.6
    assert carry > 0.0
    assert 400 < carry < 600


def test_roll_yield_backwardation_credits_long():
    """Long position in backwardation (positive roll yield) receives a credit."""
    carry = roll_yield_carry_cost(
        notional=100_000.0,
        annualised_roll_yield_pct=4.0,
        holding_days=60,
    )
    assert carry < 0.0  # negative cost = credit


def test_roll_yield_zero_when_no_holding():
    assert roll_yield_carry_cost(100_000.0, 5.0, 0) == 0.0
