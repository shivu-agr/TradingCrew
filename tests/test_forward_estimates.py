"""Tests for the forward-looking consensus tool (hermetic — yfinance mocked)."""

import pandas as pd
import pytest

import trading_crew.tools as tools


class _FakeTicker:
    def __init__(self, populated: bool):
        if populated:
            self.info = {
                "currentPrice": 54.79,
                "forwardPE": 25.0,
                "trailingPE": 57.7,
                "forwardEps": 2.19,
                "trailingEps": 0.95,
                "targetMeanPrice": 57.01,
                "targetHighPrice": 65.0,
                "targetLowPrice": 40.17,
                "numberOfAnalystOpinions": 15,
            }
            self.earnings_estimate = pd.DataFrame(
                {"avg": [1.92, 2.19], "growth": [0.188, 0.138]}, index=["0y", "+1y"]
            )
            self.revenue_estimate = pd.DataFrame(
                {"avg": [2.83e9, 3.20e9], "growth": [0.117, 0.128]}, index=["0y", "+1y"]
            )
            self.eps_revisions = pd.DataFrame(
                {"upLast30days": [16, 9], "downLast30days": [0, 6]}, index=["0y", "+1y"]
            )
            self.growth_estimates = pd.DataFrame(
                {"stockTrend": [0.12]}, index=["LTG"]
            )
            self.recommendations_summary = pd.DataFrame(
                [{"strongBuy": 3, "buy": 6, "hold": 8, "sell": 0, "strongSell": 0}]
            )
        else:
            self.info = {}
            self.earnings_estimate = pd.DataFrame()
            self.revenue_estimate = pd.DataFrame()
            self.eps_revisions = pd.DataFrame()
            self.growth_estimates = pd.DataFrame()
            self.recommendations_summary = pd.DataFrame()


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(tools, "resolve_ticker", lambda t: t.upper())


def _call(ticker: str) -> str:
    # Distinct tickers per test avoid memoize-cache collisions.
    return tools.get_forward_estimates.run(ticker)


def test_populated_output_has_all_sections(monkeypatch):
    monkeypatch.setattr(tools.yf, "Ticker", lambda t: _FakeTicker(True))
    out = _call("NTNX")
    assert "+4.1% implied" in out  # 57.01/54.79 - 1
    assert "forward P/E 25.0 vs trailing 57.7" in out
    assert "compressing (earnings expected to rise)" in out
    assert "forward EPS 2.19 vs trailing 0.95 (+130.5%)" in out
    assert "FY0 1.92 (growth +18.8%)" in out
    assert "FY0 2.83B" in out
    assert "16up/0down" in out
    assert "strongBuy 3" in out and "sell 0" in out
    # provenance tags present + estimate-specific
    assert "[source: yfinance analyst_price_targets NTNX]" in out
    assert "[source: yfinance earnings_estimate NTNX]" in out
    assert "[source: yfinance revenue_estimate NTNX]" in out


def test_sparse_coverage_is_honest_not_fabricated(monkeypatch):
    monkeypatch.setattr(tools.yf, "Ticker", lambda t: _FakeTicker(False))
    out = _call("XYZ")
    assert "No forward-looking consensus available" in out
    assert "Do NOT fabricate estimates" in out
    # must NOT invent any numbers
    assert "implied" not in out
    assert "forward P/E" not in out


def test_registered_and_wired():
    assert "get_forward_estimates" in tools.ALL_TOOLS
    assert "get_forward_estimates" in tools.DEFAULT_AGENT_TOOLS["fundamentals_analyst"]
