"""M1 — PortfolioState invariants and atomic persistence."""

from __future__ import annotations

import json
import os

import pytest

from trading_crew.agentic.portfolio.state import (
    Position,
    PortfolioState,
    PortfolioStateStore,
    load_portfolio_state,
    save_portfolio_state,
)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


def test_position_market_value_and_pnl_for_a_long():
    p = Position(
        symbol="AAPL", qty=100, avg_cost=150.0, last_price=160.0,
        last_mark_ts="2026-01-15T20:00:00+00:00",
        opened_ts="2026-01-15T15:00:00+00:00",
    )
    assert p.market_value == 16000.0
    assert p.cost_basis == 15000.0
    assert p.unrealized_pnl == 1000.0


def test_position_unrealized_pnl_for_a_short_inverts_sign():
    """A short benefits when the price drops — verify the sign."""
    p = Position(
        symbol="TSLA", qty=-50, avg_cost=200.0, last_price=180.0,
        last_mark_ts="2026-01-15T20:00:00+00:00",
        opened_ts="2026-01-15T15:00:00+00:00",
    )
    assert p.market_value == -9000.0
    assert p.unrealized_pnl == 1000.0


# ---------------------------------------------------------------------------
# PortfolioState — single-fill invariants
# ---------------------------------------------------------------------------


def _fresh_state(starting=100_000.0) -> PortfolioState:
    return PortfolioState(
        portfolio_id="test", base_currency="USD",
        starting_cash=starting, cash=starting, peak_nav=starting,
    )


def test_apply_fill_opens_new_long_and_debits_cash():
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 150.0, fees=5.0, ts="2026-01-15T15:00:00+00:00")
    assert s.cash == pytest.approx(100_000.0 - 100 * 150.0 - 5.0)
    assert "AAPL" in s.positions
    assert s.positions["AAPL"].qty == 100
    assert s.positions["AAPL"].avg_cost == 150.0


def test_apply_fill_with_zero_qty_raises():
    s = _fresh_state()
    with pytest.raises(ValueError, match="qty_delta=0"):
        s.apply_fill("AAPL", 0, 150.0, fees=0, ts="t")


def test_apply_fill_adding_to_long_rolls_weighted_average_cost():
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 100.0, fees=0, ts="t1")
    s.apply_fill("AAPL", 100, 200.0, fees=0, ts="t2")
    # Weighted average = (100*100 + 100*200) / 200 = 150
    assert s.positions["AAPL"].qty == 200
    assert s.positions["AAPL"].avg_cost == 150.0


def test_apply_fill_partial_close_realises_pnl_and_keeps_avg_cost():
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 100.0, fees=0, ts="t1")
    s.apply_fill("AAPL", -40, 120.0, fees=0, ts="t2")
    # Realised P&L on the 40 closed shares = 40 * (120 - 100) = 800
    assert s.realized_pnl == 800.0
    # Avg cost on the remaining 60 shares stays at the original 100
    assert s.positions["AAPL"].qty == 60
    assert s.positions["AAPL"].avg_cost == 100.0


def test_apply_fill_full_close_removes_position():
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 100.0, fees=0, ts="t1")
    s.apply_fill("AAPL", -100, 110.0, fees=0, ts="t2")
    assert "AAPL" not in s.positions
    assert s.realized_pnl == 1000.0


def test_apply_fill_crossing_long_to_short_realises_then_opens_new_short():
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 100.0, fees=0, ts="t1")
    s.apply_fill("AAPL", -150, 120.0, fees=0, ts="t2")
    # 100 closed at +20 each = +2000 realised
    assert s.realized_pnl == 2000.0
    # 50 short opened at 120
    assert s.positions["AAPL"].qty == -50
    assert s.positions["AAPL"].avg_cost == 120.0


def test_fees_are_always_debited_from_cash_regardless_of_side():
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 100.0, fees=10.0, ts="t1")
    cash_after_buy = s.cash  # 100000 - 100*100 - 10 = 89990
    s.apply_fill("AAPL", -100, 110.0, fees=10.0, ts="t2")
    # Round-trip P&L = 100 * (110 - 100) = +1000, less 20 fees = +980.
    # Ending cash = starting_cash + net P&L = 100000 + 980 = 100980.
    assert s.cash == pytest.approx(100_980.0)
    assert s.cash > cash_after_buy
    # And the realised P&L on the books should be the gross P&L (fees go to cash,
    # not realised P&L) — that matches paper §6.2 cost-vs-pnl separation.
    assert s.realized_pnl == pytest.approx(1_000.0)


# ---------------------------------------------------------------------------
# PortfolioState — NAV, drawdown, exposure
# ---------------------------------------------------------------------------


def test_mark_to_market_only_updates_positions_with_a_supplied_price():
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 100.0, fees=0, ts="t1")
    s.apply_fill("MSFT", 50, 200.0, fees=0, ts="t1")
    s.mark_to_market({"AAPL": 110.0}, ts="t2")  # MSFT NOT in prices
    assert s.positions["AAPL"].last_price == 110.0
    # MSFT keeps its previous mark price (its opened_ts price)
    assert s.positions["MSFT"].last_price == 200.0


def test_peak_nav_and_max_drawdown_are_monotonic():
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 100.0, fees=0, ts="t1")
    s.mark_to_market({"AAPL": 120.0}, ts="t2")  # NAV up
    peak1 = s.peak_nav
    s.mark_to_market({"AAPL": 90.0}, ts="t3")  # NAV down
    assert s.peak_nav == peak1  # peak doesn't decrease
    expected_dd = (peak1 - s.nav) / peak1
    assert s.max_drawdown == pytest.approx(expected_dd)
    s.mark_to_market({"AAPL": 80.0}, ts="t4")  # deeper drawdown
    assert s.max_drawdown > expected_dd


def test_gross_exposure_counts_short_position_abs_value():
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 100.0, fees=0, ts="t1")
    s.apply_fill("TSLA", -50, 200.0, fees=0, ts="t1")
    # +10000 long + |-10000| short = 20000 gross
    assert s.gross_exposure == 20_000.0
    # signed net = 10000 - 10000 = 0
    assert s.net_exposure == 0.0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_then_load_roundtrips_through_disk(tmp_path):
    path = tmp_path / "book.json"
    s = _fresh_state()
    s.apply_fill("AAPL", 100, 150.0, fees=5.0, ts="t1")
    s.mark_to_market({"AAPL": 160.0}, ts="t2")

    store = PortfolioStateStore(path)
    store.save(s)
    assert path.is_file()

    loaded = store.load()
    assert loaded.cash == pytest.approx(s.cash)
    assert loaded.realized_pnl == s.realized_pnl
    assert loaded.positions["AAPL"].avg_cost == 150.0
    assert loaded.positions["AAPL"].last_price == 160.0


def test_atomic_write_leaves_no_tmp_file_on_success(tmp_path):
    path = tmp_path / "book.json"
    PortfolioStateStore(path).save(_fresh_state())
    tmp = path.with_suffix(path.suffix + ".tmp")
    assert not tmp.exists()


def test_load_portfolio_state_initialises_when_no_file_exists(tmp_path):
    path = tmp_path / "fresh.json"
    s = load_portfolio_state("u123", starting_cash=50_000.0, path=path)
    assert s.cash == 50_000.0
    assert s.starting_cash == 50_000.0
    assert s.peak_nav == 50_000.0
    # File should now exist
    assert path.is_file()


def test_load_portfolio_state_reuses_existing_cash_over_config_default(tmp_path):
    path = tmp_path / "book.json"
    load_portfolio_state("u123", starting_cash=50_000.0, path=path)
    # Caller passes a different starting_cash — but the existing file wins
    s2 = load_portfolio_state("u123", starting_cash=99_999.0, path=path)
    assert s2.starting_cash == 50_000.0
    assert s2.cash == 50_000.0


def test_schema_version_mismatch_raises_rather_than_silently_loading(tmp_path):
    path = tmp_path / "book.json"
    bad = {
        "schema_version": 999,
        "portfolio_id": "x", "base_currency": "USD",
        "starting_cash": 100, "cash": 100, "positions": {},
        "realized_pnl": 0, "peak_nav": 100, "max_drawdown": 0,
        "created_ts": "t", "last_update_ts": "t",
    }
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="schema_version mismatch"):
        PortfolioStateStore(path).load()


def test_save_portfolio_state_helper_writes_through_default_path(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_CACHE_DIR", str(tmp_path))
    s = _fresh_state()
    save_portfolio_state(s)
    expected = tmp_path / "portfolios" / "test.json"
    assert expected.is_file()
