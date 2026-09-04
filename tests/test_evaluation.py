"""Tests for the reusable end-to-end evaluation service.

These exercise the "compact" glue that ties memory -> backtest ->
attribution -> fusion together, using an injected synthetic OHLCV provider
so they are deterministic and offline (no yfinance, no episode store).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from trading_crew.agentic import evaluation
from trading_crew.agentic.execution.contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    ValidityCheck,
)


def _valid_check() -> ValidityCheck:
    return ValidityCheck(
        data_timestamps_valid=True,
        fits_risk_budget=True,
        survives_transaction_costs=True,
        liquidity_sufficient=True,
    )


def _proposal(ts: str, side: ActionSide, weight: float) -> ActionProposal:
    return ActionProposal(
        symbol="TICK",
        decision_ts=ts,
        side=side,
        target_weight=weight,
        horizon_days=30,
        conviction_score=0.7,
        conviction_tier=ConvictionTier.from_score(0.7),
        expected_return_pct=0.05 if weight >= 0 else -0.05,
        rationale="test",
        validity_check=_valid_check(),
    )


def _episode(proposal: ActionProposal) -> SimpleNamespace:
    """Minimal duck-typed episode: evaluation only reads .action_proposal + .symbol."""
    return SimpleNamespace(
        symbol=proposal.symbol,
        action_proposal=proposal.model_dump(mode="json"),
    )


def _synthetic_ohlcv(start="2022-06-01", periods=500, drift=0.001):
    """A gently trending daily OHLCV frame (Date/Open/High/Low/Close/Volume)."""
    dates = pd.date_range(start=start, periods=periods, freq="D")
    close = [100.0]
    for _ in range(1, periods):
        close.append(close[-1] * (1 + drift))
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": [c * 1.01 for c in close],
            "Low": [c * 0.99 for c in close],
            "Close": close,
            "Volume": [1_000_000] * periods,
        }
    )
    return df


def test_proposals_from_episodes_sorts_and_skips_malformed():
    p_late = _proposal("2023-03-01T00:00:00+00:00", ActionSide.BUY, 0.05)
    p_early = _proposal("2023-01-01T00:00:00+00:00", ActionSide.BUY, 0.05)
    good = [_episode(p_late), _episode(p_early)]
    bad = SimpleNamespace(symbol="TICK", action_proposal={"garbage": True})

    proposals = evaluation.proposals_from_episodes([*good, bad])

    assert [p.decision_ts for p in proposals] == [
        "2023-01-01T00:00:00+00:00",
        "2023-03-01T00:00:00+00:00",
    ]


def test_evaluate_ticker_end_to_end_with_injected_provider(monkeypatch):
    # Six monthly BUY decisions on a rising tape.
    base = datetime(2023, 1, 2, tzinfo=timezone.utc)
    episodes = [
        _episode(_proposal((base + timedelta(days=30 * i)).isoformat(), ActionSide.BUY, 0.05))
        for i in range(6)
    ]
    monkeypatch.setattr(evaluation, "load_cohort", lambda *a, **k: episodes)

    report = evaluation.evaluate_ticker(
        "TICK",
        ohlcv_provider=lambda *a, **k: _synthetic_ohlcv(),
        train_size=3,
        embargo_size=1,
        test_size=1,
    )

    assert report.error is None
    assert report.n_proposals == 6
    assert report.n_folds > 0
    # Fractional shares are on by default for backtests, so small weights fill.
    assert report.n_filled > 0
    assert report.metrics is not None
    assert report.attribution is not None
    assert report.attribution["n_directional"] > 0


def test_evaluate_ticker_reports_error_without_episodes(monkeypatch):
    monkeypatch.setattr(evaluation, "load_cohort", lambda *a, **k: [])
    report = evaluation.evaluate_ticker("TICK", ohlcv_provider=lambda *a, **k: _synthetic_ohlcv())
    assert report.n_proposals == 0
    assert report.error is not None


def test_latest_fused_signal_shapes(monkeypatch):
    episodes = [
        _episode(_proposal("2023-01-02T00:00:00+00:00", ActionSide.BUY, 0.05)),
        _episode(_proposal("2023-02-02T00:00:00+00:00", ActionSide.BUY, 0.05)),
    ]
    monkeypatch.setattr(evaluation, "load_cohort", lambda *a, **k: episodes)

    out = evaluation.latest_fused_signal(
        "TICK", ohlcv_provider=lambda *a, **k: _synthetic_ohlcv()
    )
    assert out is not None
    assert out["ticker"] == "TICK"
    assert "fused" in out and "side" in out["fused"]
    assert isinstance(out["fused"]["guardrails"], list)
