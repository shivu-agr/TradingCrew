"""Deterministic signal-alignment guard — desks/computed trends must be used."""

from __future__ import annotations

from trading_crew.agentic.signals.alignment import (
    apply_signal_alignment,
    computed_trend_lean,
)
from trading_crew.schemas import PortfolioDecision


def _dec(action="BUY", size=2.0, rationale="thesis [source: x]"):
    return PortfolioDecision(
        action=action,
        confidence=0.6,
        size_pct_of_book=size,
        entry_price=100.0,
        stop_loss=92.0,
        target_price=118.0,
        horizon_days=90,
        expected_return_pct=0.18,
        rationale=rationale,
        key_drivers=["a"],
        key_risks=["b"],
        falsifiers=["c"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
    )


def _board(*pairs):
    return [{"desk": d, "bias": b} for d, b in pairs]


def test_majority_oppose_holds_a_buy():
    aligned, notes = apply_signal_alignment(
        _dec("BUY"),
        board=_board(
            ("Market", "bearish"),
            ("News", "bearish"),
            ("Fundamentals", "bullish"),
            ("Reports/Guidance", "bearish"),
            ("Macro/Geopolitical", "neutral"),
        ),
    )
    assert aligned.action == "HOLD"
    assert aligned.size_pct_of_book == 0.0
    assert any("majority" in n for n in notes)
    assert "[signal alignment]" in aligned.rationale


def test_split_board_holds_rather_than_forcing_a_side():
    aligned, notes = apply_signal_alignment(
        _dec("BUY"),
        board=_board(
            ("Market", "bearish"),
            ("News", "bullish"),
            ("Fundamentals", "bullish"),
            ("Reports/Guidance", "neutral"),
            ("Macro/Geopolitical", "bearish"),
        ),
    )
    assert aligned.action == "HOLD"
    assert aligned.size_pct_of_book == 0.0
    assert any("split" in n for n in notes)


def test_split_is_symmetric_for_sell():
    aligned, _ = apply_signal_alignment(
        _dec("SELL", size=1.5),
        board=_board(
            ("Market", "bearish"),
            ("News", "bullish"),
            ("Fundamentals", "bullish"),
            ("Macro/Geopolitical", "bearish"),
        ),
    )
    assert aligned.action == "HOLD"
    assert aligned.size_pct_of_book == 0.0


def test_neutrals_do_not_create_a_split():
    """3 bullish / 2 neutral is a bullish majority — the NTNX HOLD
    pattern of treating neutrals as a split must not fire here."""
    original = _dec("BUY")
    aligned, notes = apply_signal_alignment(
        original,
        board=_board(
            ("Market", "bullish"),
            ("News", "bullish"),
            ("Fundamentals", "neutral"),
            ("Reports/Guidance", "bullish"),
            ("Macro/Geopolitical", "neutral"),
        ),
    )
    assert aligned is original
    assert notes == []


def test_three_one_majority_is_not_a_split():
    """3 bull / 1 bear is a majority, not a tie — BHARATFORG latest."""
    original = _dec("BUY")
    aligned, notes = apply_signal_alignment(
        original,
        board=_board(
            ("Market", "bullish"),
            ("News", "neutral"),
            ("Fundamentals", "bearish"),
            ("Reports/Guidance", "bullish"),
            ("Macro/Geopolitical", "bullish"),
        ),
    )
    assert aligned is original
    assert notes == []


def test_one_sided_board_leaves_buy_alone():
    original = _dec("BUY")
    aligned, notes = apply_signal_alignment(
        original,
        board=_board(
            ("Market", "bullish"),
            ("News", "bullish"),
            ("Fundamentals", "bullish"),
            ("Reports/Guidance", "bullish"),
            ("Macro/Geopolitical", "neutral"),
        ),
    )
    assert aligned is original
    assert notes == []


def test_strong_buy_capped_when_any_desk_opposes():
    aligned, notes = apply_signal_alignment(
        _dec("STRONG_BUY", size=2.5),
        board=_board(
            ("Market", "bullish"),
            ("News", "bullish"),
            ("Fundamentals", "bullish"),
            ("Reports/Guidance", "bullish"),
            ("Macro/Geopolitical", "bearish"),
        ),
    )
    assert aligned.action == "BUY"
    assert aligned.size_pct_of_book == 2.5  # size kept; only the tier drops
    assert any("STRONG" in n for n in notes)


def test_empty_or_thin_board_does_not_hold():
    original = _dec("BUY")
    aligned, notes = apply_signal_alignment(original, board=[])
    assert aligned is original and notes == []
    # Two desks is below the coverage floor — a parse miss must not sit out.
    aligned, notes = apply_signal_alignment(
        original,
        board=_board(("Market", "bearish"), ("News", "bearish")),
    )
    assert aligned is original and notes == []


def test_hold_passthrough():
    original = _dec("HOLD", size=0.0)
    aligned, notes = apply_signal_alignment(
        original,
        board=_board(("Market", "bullish"), ("News", "bullish"), ("Macro", "bullish")),
    )
    assert aligned is original and notes == []


def test_computed_trends_count_as_a_virtual_desk():
    # 2-2 split would HOLD; both computed trends accelerating tips it bullish.
    report = {
        "signals": [
            {"name": "REVENUE_TREND", "direction": "ACCELERATING"},
            {"name": "EARNINGS_TREND", "direction": "ACCELERATING"},
        ]
    }
    original = _dec("BUY")
    aligned, notes = apply_signal_alignment(
        original,
        board=_board(
            ("Market", "bearish"),
            ("News", "bullish"),
            ("Fundamentals", "bullish"),
            ("Macro/Geopolitical", "bearish"),
        ),
        report_signals=report,
    )
    assert aligned is original
    assert notes == []
    assert computed_trend_lean(report) == "bullish"


def test_computed_lean_mixed_is_not_a_vote():
    assert computed_trend_lean({
        "signals": [
            {"name": "REVENUE_TREND", "direction": "DECELERATING"},
            {"name": "EARNINGS_TREND", "direction": "ACCELERATING"},
        ]
    }) is None


def test_price_structure_ledger_does_not_cite_absent_dip_buy():
    from trading_crew.price_structure import PriceStructure
    from trading_crew.report_signals import ReportSignal

    ps = PriceStructure(
        ann_vol=0.2, vol_pct=0.3, atr_pct=0.01, pos_in_range=0.5,
        drawdown_from_high=-0.1, regime_age_days=100, hurst=0.5, ret20=0.0,
        dip_buy=False,
        signals=[ReportSignal(
            name="VOL_STATE", direction="NORMAL", confidence=0.7, computed=True,
        )],
    )
    ledger = ps.render_pm_ledger()
    assert "DIP_BUY setup present" not in ledger
    assert "cite the DIP_BUY" not in ledger
    assert "does not prescribe a side" in ledger.lower()


def test_computed_both_decelerating_can_hold_a_buy_on_a_close_board():
    aligned, notes = apply_signal_alignment(
        _dec("BUY"),
        board=_board(
            ("Market", "bullish"),
            ("News", "bearish"),
            ("Fundamentals", "bullish"),
            ("Macro/Geopolitical", "bearish"),
        ),
        report_signals={
            "signals": [
                {"name": "REVENUE_TREND", "direction": "DECELERATING"},
                {"name": "EARNINGS_TREND", "direction": "DECLINING"},
            ]
        },
    )
    assert aligned.action == "HOLD"
    assert any("majority" in n or "split" in n for n in notes)
