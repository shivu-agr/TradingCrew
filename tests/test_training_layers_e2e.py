"""End-to-end training layers: Hermes apply + L1 retrieval + L3+Hermes sizing.

Isolated cache only — never writes the live journal. L4 is not on the
live prediction path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trading_crew.agentic.harness import seed_and_step, seed_harness_book, step_harness
from trading_crew.agentic.memory.episodic import Episode, EpisodicMemory, OutcomeStatus, Regime
from trading_crew.agentic.self_learn import active_experiment, journal_path
from trading_crew.agentic.strategy import load_desk_strategy
from trading_crew.graph.routers import debate_rounds_from_action, resolve_playbook_action
from trading_crew.tools import ALL_TOOLS, DEFAULT_AGENT_TOOLS


def _llm_env(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_LLM_MODEL", "dummy-model")
    monkeypatch.setenv("VLLM_LLM_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("VLLM_LLM_API_KEY", "dummy")
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://example.com")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "abc")
    monkeypatch.setenv("TRADINGCREW_DECISION_PLANNING", "0")
    monkeypatch.setenv("TRADINGCREW_MEMORY_EMBEDDER", "tfidf")


def _ohlcv(days: int = 260, start: str = "2024-01-02") -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="B")
    close = [100.0 + i * 0.15 for i in range(days)]
    return pd.DataFrame({
        "Date": dates,
        "Open": [c - 0.4 for c in close],
        "High": [c + 0.8 for c in close],
        "Low": [c - 0.8 for c in close],
        "Close": close,
        "Volume": [1_000_000] * days,
    })


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    _llm_env(monkeypatch)
    return tmp_path


def _seed_step(isolated, *, pattern: str, fail: bool = True):
    mem = EpisodicMemory(isolated / "memory" / "episodes.jsonl")
    s = load_desk_strategy()
    seeded = seed_and_step(
        count=s.min_reviews_to_change,
        memory=mem,
        journal_file=journal_path(),
        fail=fail,
        pattern=pattern,
    )
    return s, mem, seeded, active_experiment(journal_path())


def test_hermes_side_applies_strategy_and_topology_on_real_crew(isolated):
    s, _, seeded, exp = _seed_step(isolated, pattern="side")
    assert seeded["harness"]["action"] == "open"
    assert exp["knob"] == "debate_rounds_2"
    assert resolve_playbook_action("US") == "debate_rounds_2"
    assert debate_rounds_from_action("debate_rounds_2", ui_rounds=1) == 2

    from trading_crew.crew import TradingCrew

    agents = TradingCrew(ticker="AAPL", memory=False).materialize_agents()
    trader = agents["trader"].backstory
    rm = agents["research_manager"].backstory
    pm = agents["portfolio_manager"].backstory
    fund = agents["fundamentals_analyst"].backstory
    market = agents["market_analyst"].backstory

    for block in (trader, rm, pm):
        assert "## DESK STRATEGY" in block
        assert s.mandate.split()[0] in block or "cash-market book" in block
        assert "cashflow_positioning_book" in block
        assert "30-day SUCCESS" in block
        assert "ACTIVE LEARNING EXPERIMENT" in block
        assert "debate_rounds_2" in block
        assert "Prefer cash-flow quality" in block
        assert "Do not open a second experiment" in block
    assert "DESK STRATEGY" not in market
    assert "ACTIVE LEARNING EXPERIMENT" not in fund
    assert "HARNESS DESK ADDENDUM" not in fund
    assert "retrieve_past_episodes" not in DEFAULT_AGENT_TOOLS["market_analyst"]
    assert "retrieve_past_episodes" not in ALL_TOOLS
    assert "rl_policy_recommendation" not in DEFAULT_AGENT_TOOLS["trader"]
    assert "rl_policy_recommendation" not in DEFAULT_AGENT_TOOLS["market_analyst"]
    assert "rl_policy_recommendation" not in ALL_TOOLS


def test_hermes_desk_prompt_only_on_hurt_desk(isolated):
    _, _, _, exp = _seed_step(isolated, pattern="desk")
    assert exp["knob"] == "desk:fundamentals_analyst"
    assert exp["apply"] == "desk_prompt"
    assert "NET_GUIDANCE_BIAS" in (exp.get("desk_prompts") or {})["fundamentals_analyst"]

    from trading_crew.crew import TradingCrew

    agents = TradingCrew(ticker="AAPL", memory=False).materialize_agents()
    assert "HARNESS DESK ADDENDUM" in agents["fundamentals_analyst"].backstory
    assert "NET_GUIDANCE_BIAS" in agents["fundamentals_analyst"].backstory
    assert "HARNESS DESK ADDENDUM" not in agents["macro_analyst"].backstory
    assert "desk:fundamentals_analyst" in agents["trader"].backstory
    assert "ACTIVE LEARNING EXPERIMENT" in agents["portfolio_manager"].backstory
    assert resolve_playbook_action("US") == "default"


def test_hermes_alpha_cuts_live_sizer(isolated):
    _, _, _, exp = _seed_step(isolated, pattern="alpha", fail=False)
    assert exp["knob"] == "size_scale_down"
    assert exp["apply"] == "prompt"
    assert exp["size_scale"] == 0.75

    from trading_crew.crew import TradingCrew
    from trading_crew.agentic.risk import SizingConfig
    from trading_crew.agentic.tuning import resolve_live_sizing

    agents = TradingCrew(ticker="AAPL", memory=False).materialize_agents()
    assert "size_scale_down" in agents["trader"].backstory
    assert "Cut directional size" in agents["trader"].backstory
    assert resolve_playbook_action("US") == "default"

    sizing, gate, meta = resolve_live_sizing("AAPL", cache_dir=str(isolated))
    assert meta["hermes_size_scale"] == 0.75
    assert meta["l3_applied"] is False
    default = SizingConfig()
    assert sizing is not None and gate is not None
    assert sizing.kelly_fraction == pytest.approx(default.kelly_fraction * 0.75)
    assert sizing.max_position_weight == pytest.approx(default.max_position_weight * 0.75)
    assert gate.max_position_weight == pytest.approx(0.20 * 0.75)


def test_hermes_split_and_commodity_strategy_rules(isolated):
    _, _, _, exp = _seed_step(isolated, pattern="split")
    assert exp["knob"] == "extra_debate_on_split"
    assert resolve_playbook_action("US") == "extra_debate_on_split"

    from commodity_crew.crew import CommodityCrew

    crew = CommodityCrew(ticker="CL=F", memory=False, debate_rounds=1, risk_rounds=1).crew()
    by_role = {a.role: a for a in crew.agents}
    trader = by_role["Senior Futures Trader"].backstory
    assert "## DESK STRATEGY" in trader
    assert "commodity" in trader.lower() or "Curve first" in trader
    assert "extra_debate_on_split" in trader
    assert "ACTIVE LEARNING EXPERIMENT" in trader
    market = by_role["Commodity Market Analyst"].backstory
    assert "ACTIVE LEARNING EXPERIMENT" not in market


def test_l3_grid_apply_changes_live_sizer(isolated):
    from trading_crew.agentic.backtest import WalkForwardConfig
    from trading_crew.agentic.execution.contracts import (
        ActionProposal,
        ActionSide,
        ConvictionTier,
        OrderTimeInForce,
        SizingBasis,
        ValidityCheck,
    )
    from trading_crew.agentic.execution.pipeline import run_pipeline
    from trading_crew.agentic.grid_search import GridAxis, run_grid_search
    from trading_crew.agentic.risk import SizingConfig
    from trading_crew.agentic.tuning import load_tuned_config, save_tuned_config

    def _prop(ts: str) -> ActionProposal:
        return ActionProposal(
            symbol="L3E2E",
            decision_ts=ts,
            side=ActionSide.BUY,
            target_weight=0.08,
            horizon_days=10,
            tif=OrderTimeInForce.DAY,
            conviction_score=0.7,
            conviction_tier=ConvictionTier.HIGH,
            sizing_basis=SizingBasis.TARGET_WEIGHT,
            expected_return_pct=0.06,
            rationale="L3 e2e proposal " + "x" * 80,
            validity_check=ValidityCheck(
                data_timestamps_valid=True,
                fits_risk_budget=True,
                survives_transaction_costs=True,
                liquidity_sufficient=True,
                notes="ok",
            ),
        )

    proposals = [
        _prop("2024-02-01T00:00:00"),
        _prop("2024-02-15T00:00:00"),
        _prop("2024-03-01T00:00:00"),
        _prop("2024-03-15T00:00:00"),
    ]
    result = run_grid_search(
        proposals=proposals,
        ohlcv_by_symbol={"L3E2E": _ohlcv()},
        walk_forward_config=WalkForwardConfig(train_size=2, embargo_size=0, test_size=1),
        grid=[
            GridAxis("kelly_fraction", (0.10, 0.50)),
            GridAxis("vol_target", (0.10,)),
            GridAxis("max_position_weight", (0.12,)),
        ],
    )
    best = result.best()
    assert best is not None
    assert result.n_points == 2
    save_tuned_config(
        "L3E2E",
        sizing_config=best.sizing_config,
        gate_config=best.gate_config,
        meta={"rank_metric": best.rank_metric, "n_proposals": result.n_proposals},
        cache_dir=str(isolated),
    )
    loaded = load_tuned_config("L3E2E", cache_dir=str(isolated))
    assert loaded is not None
    sizing, gate = loaded
    assert isinstance(sizing, SizingConfig)
    assert sizing.kelly_fraction in (0.10, 0.50)

    default = run_pipeline(
        proposals[0], ohlcv=_ohlcv(), persist=False, portfolio_id="l3-default",
    )
    tuned = run_pipeline(
        proposals[0],
        ohlcv=_ohlcv(),
        persist=False,
        portfolio_id="l3-tuned",
        sizing_config=sizing,
        gate_config=gate,
    )
    assert default.sizing is not None and tuned.sizing is not None
    assert default.sizing.kelly_cap != tuned.sizing.kelly_cap or (
        default.sizing.final_weight != tuned.sizing.final_weight
    )


def test_l4_not_on_live_prediction_path(isolated):
    from commodity_crew.tools import ALL_TOOLS as COMM_ALL
    from commodity_crew.tools import DEFAULT_AGENT_TOOLS as COMM_DEFAULT

    assert "rl_policy_recommendation" not in ALL_TOOLS
    assert "rl_policy_recommendation" not in DEFAULT_AGENT_TOOLS["trader"]
    assert "rl_policy_recommendation" not in DEFAULT_AGENT_TOOLS["market_analyst"]
    assert "rl_policy_recommendation_commodity" not in COMM_ALL
    for names in COMM_DEFAULT.values():
        assert "rl_policy_recommendation_commodity" not in names
