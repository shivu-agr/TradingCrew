"""Harness workflow: config fail-closed, safety gates, loop diversity."""

from __future__ import annotations

import pytest

from trading_crew.harness.config import HarnessConfigError, load_harness_config
from trading_crew.harness.safety import (
    HarnessGateError,
    check_spawn,
    check_tool,
    coverage_gaps,
    diversity_gaps,
)
from trading_crew.horizons import Horizon
from trading_crew.schemas import (
    HorizonForecast,
    HorizonForecasts,
    PortfolioDecision,
)


def _forecast(stance: str, days: int, *drivers: str) -> HorizonForecast:
    return HorizonForecast(
        stance=stance,
        horizon_days=days,
        expected_return_pct=0.02 if stance == "bullish" else (-0.02 if stance == "bearish" else 0.0),
        confidence=0.5,
        drivers=list(drivers),
        risks=["gap"],
        falsifiers=["print misses"],
        sources=["test"],
    )


def test_conductor_turn_coerces_messy_live_json():
    from trading_crew.harness.conductor import ConductorTurn

    spawn = ConductorTurn.model_validate({
        "action": "spawn_parallel",
        "tool": None,
        "arguments": None,
        "primary_horizon": "",
        "agents": [
            {"kind": "bear_researcher"},
            {"kind": "market_analyst"},
        ],
    })
    assert spawn.agents[0].kind == "desk"
    assert spawn.agents[0].desk == "bear_researcher"
    assert spawn.primary_horizon is None

    inferred = ConductorTurn.model_validate({
        "tool": "get_analyst_recommendations",
        "arguments": {"ticker": "NTNX"},
    })
    assert inferred.action == "call_tool"


def test_spawn_instruction_leads_standing_remit():
    from trading_crew.graph.nodes import merge_spawn_instruction

    text = merge_spawn_instruction(
        "Investigate NTNX through your assigned lens.",
        "You are working the reports_analyst lens on NTNX (stock harness).\n\n"
        "## User request (do this spawn thoroughly)\n\n"
        "1. Call get_report_insights.",
    )
    assert text.startswith("You are working the reports_analyst lens")
    assert "## Standing desk remit" in text
    assert text.index("User request") < text.index("Investigate NTNX")


def test_planner_binds_debate_and_risk_tools():
    from trading_crew.harness.conductor import SpawnSpec
    from trading_crew.harness.planner import plan_spawn
    from trading_crew.harness.state import HarnessState

    cfg = load_harness_config()
    captured = {}

    def writer(**kwargs):
        captured[kwargs["desk"]] = list(kwargs["bound"])
        return "1. Re-check the gap with tools."

    plan_spawn(
        spec=SpawnSpec(kind="desk", desk="debate", brief="need both sides"),
        state=HarnessState(ticker="NTNX"),
        cfg=cfg,
        writer=writer,
    )
    plan_spawn(
        spec=SpawnSpec(kind="desk", desk="risk", brief="size the book"),
        state=HarnessState(ticker="NTNX"),
        cfg=cfg,
        writer=writer,
    )
    assert "research_web" in captured["debate"]
    assert "get_stock_data" in captured["risk"]


def test_conductor_prompt_mentions_spawn_planner():
    from trading_crew.harness.conductor import build_system_prompt

    text = build_system_prompt(
        ticker="NTNX",
        standing="rules",
        allowlist=["get_stock_data"],
        catalog=["market_analyst"],
        spawn_left=3,
        steps_left=10,
        coverage="Missing required lenses: market_analyst",
    )
    assert "spawn planner" in text
    assert "short gap brief" in text


def test_conductor_turn_drops_null_forecasts_on_tool_call():
    from trading_crew.harness.conductor import ConductorTurn

    turn = ConductorTurn.model_validate({
        "action": "call_tool",
        "tool": "get_stock_data",
        "arguments": {"ticker": "NTNX"},
        "horizon_forecasts": {"SHORT": None, "MEDIUM": None, "LONG": None},
    })
    assert turn.horizon_forecasts is None
    assert turn.tool == "get_stock_data"


def test_packaged_harness_config_loads():
    cfg = load_harness_config()
    assert cfg["max_steps"] >= 1
    assert cfg["exhaust_steps"] >= 1
    assert "few steps remain" in cfg["standing_rules"]
    assert "get_market_context" in cfg["tool_allowlist"]
    assert "risk" in cfg["catalog_desks"]
    assert cfg["require_bundled_risk"] is True
    assert cfg["require_news_lens"] is True
    assert cfg["require_fundamentals_lens"] is True
    assert cfg["require_macro_lens"] is True
    assert cfg["corpus_dir"].is_dir()
    assert "policy-driven" in cfg["standing_rules"]
    # Info desks are required in gates/board, not named as a mandate in standing rules.
    assert "news_analyst" not in cfg["standing_rules"]
    assert "fundamentals_analyst" not in cfg["standing_rules"]
    assert "macro_analyst" not in cfg["standing_rules"]
    plugins = cfg["plugins"]
    assert plugins["planner"] == "spawn_v1"
    assert "get_report_insights" in plugins["skills"]["reports_analyst"]["required_tools"]
    for desk in cfg["catalog_desks"]:
        assert desk in plugins["skills"]
    assert "generic_investigator" in plugins["skills"]
    assert cfg["plugin_ctx"].get("planner", "spawn_v1") is not None


def test_planner_writes_full_spawn_contract():
    from trading_crew.harness.conductor import SpawnSpec
    from trading_crew.harness.planner import plan_spawn
    from trading_crew.harness.state import HarnessState

    cfg = load_harness_config()
    state = HarnessState(ticker="NTNX")
    state.add_observation("thought", "step 1", "need filings before debate")
    state.reports["market_analyst"] = "Tape firm [source: yfinance]"

    def writer(**kwargs):
        assert kwargs["ticker"] == "NTNX"
        assert kwargs["desk"] == "reports_analyst"
        assert "get_report_insights" in kwargs["skill"]["required_tools"]
        return (
            "1. Call get_report_insights for NTNX and re-check the tape "
            "excerpt — do not assume it is current.\n"
            "2. Extract guidance vs actual and name what you cannot verify."
        )

    text = plan_spawn(
        spec=SpawnSpec(kind="desk", desk="reports_analyst", brief="filings gap"),
        state=state,
        cfg=cfg,
        writer=writer,
    )
    assert "## User request (do this spawn thoroughly)" in text
    assert "## Constraints" in text
    assert "## Prior findings (re-verify; may be stale)" in text
    assert "## Verify" in text
    assert "## Return to parent" in text
    assert "get_report_insights" in text
    assert "NTNX" in text
    assert "Tape firm" in text


def test_planner_empty_assignment_is_an_error():
    from trading_crew.harness.conductor import SpawnSpec
    from trading_crew.harness.planner import plan_spawn
    from trading_crew.harness.state import HarnessState

    cfg = load_harness_config()
    with pytest.raises(HarnessConfigError, match="empty user_request"):
        plan_spawn(
            spec=SpawnSpec(kind="desk", desk="market_analyst", brief="tape"),
            state=HarnessState(ticker="NTNX"),
            cfg=cfg,
            writer=lambda **k: "   ",
        )


def test_missing_skill_plugin_is_an_error(monkeypatch, tmp_path):
    from trading_crew.harness.plugins import skill_for

    cfg = load_harness_config()
    cfg = dict(cfg)
    cfg["plugins"] = dict(cfg["plugins"])
    cfg["plugins"]["skills"] = dict(cfg["plugins"]["skills"])
    del cfg["plugins"]["skills"]["trader"]
    with pytest.raises(HarnessConfigError, match="no skill plugin"):
        skill_for(cfg, "trader")


def test_missing_harness_key_is_an_error(monkeypatch, tmp_path):
    path = tmp_path / "harness.yaml"
    path.write_text("max_steps: 3\n", encoding="utf-8")
    monkeypatch.setenv("TRADINGCREW_HARNESS_CONFIG", str(path))
    with pytest.raises(HarnessConfigError, match="missing keys"):
        load_harness_config()


def test_memory_overlay_missing_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGCREW_HARNESS_MEMORY", str(tmp_path / "nope.yaml"))
    with pytest.raises(HarnessConfigError, match="HARNESS_MEMORY"):
        load_harness_config()


def test_query_required_tools_reject_empty_query():
    cfg = load_harness_config()
    with pytest.raises(HarnessGateError, match="agent-written query"):
        check_tool("get_geopolitical_news", {"ticker": "NTNX", "query": ""}, cfg)
    check_tool("get_geopolitical_news", {"ticker": "NTNX", "query": "export control on hypervisor sales"}, cfg)


def test_deny_and_unknown_desk():
    cfg = load_harness_config()
    with pytest.raises(HarnessGateError, match="denied"):
        check_tool("backtest_setup", {"ticker": "NTNX"}, cfg)
    with pytest.raises(HarnessGateError, match="cannot be spawned"):
        check_spawn(
            "portfolio_manager",
            depth=0,
            cfg=cfg,
            spawn_used=0,
        )
    with pytest.raises(HarnessGateError, match="unknown catalog"):
        check_spawn(
            "not_a_desk",
            depth=0,
            cfg=cfg,
            spawn_used=0,
        )
    with pytest.raises(HarnessGateError, match="depth"):
        check_spawn(
            "macro_analyst",
            depth=1,
            cfg=cfg,
            spawn_used=0,
        )
    check_spawn("bear_researcher", depth=0, cfg=cfg, spawn_used=0)
    check_spawn("bear_researcher", depth=0, cfg=cfg, spawn_used=int(cfg["max_spawn"]) - 1)
    with pytest.raises(HarnessGateError, match="max_spawn"):
        check_spawn("bear_researcher", depth=0, cfg=cfg, spawn_used=int(cfg["max_spawn"]))


def test_diversity_gaps():
    cfg = load_harness_config()
    assert diversity_gaps(families={"policy"}, skeptical=False, cfg=cfg)
    assert not diversity_gaps(
        families={"price", "fundamental", "policy"},
        skeptical=True,
        cfg=cfg,
    )


def test_coverage_gaps_require_debate_reports_market_risk():
    cfg = load_harness_config()
    assert cfg["require_debate_pair"] is True
    assert cfg["require_bundled_debate"] is True
    assert cfg["require_bundled_risk"] is True
    assert cfg["require_reports_lens"] is True
    assert cfg["require_news_lens"] is True
    assert cfg["require_fundamentals_lens"] is True
    assert cfg["require_macro_lens"] is True
    families = {"price", "fundamental", "policy"}
    gaps = coverage_gaps(
        families=families,
        skeptical=True,
        spawned=["fundamentals_analyst", "bear_researcher"],
        reports={"bear_researcher": "bear case"},
        tool_names=["get_stock_data", "get_fundamentals", "get_news"],
        steps=3,
        cfg=cfg,
    )
    joined = " ".join(gaps)
    assert "market_analyst" in joined
    assert "news_analyst" in joined
    assert "macro_analyst" in joined
    assert "reports_analyst" in joined or "get_report_insights" in joined
    assert "bull_researcher" in joined
    assert "bundled debate" in joined
    assert "risk" in joined
    assert "long-horizon" in joined
    sides_only = coverage_gaps(
        families=families,
        skeptical=True,
        spawned=[
            "market_analyst",
            "reports_analyst",
            "bull_researcher",
            "bear_researcher",
            "risk_conservative",
        ],
        reports={
            "bull_researcher": "bull",
            "bear_researcher": "bear",
            "risk_conservative": "size down",
        },
        tool_names=["get_stock_data", "get_report_insights", "get_news"],
        steps=int(cfg["min_steps_before_finish"]),
        cfg=cfg,
    )
    assert any("bundled debate" in g for g in sides_only)
    assert any("bundled risk" in g for g in sides_only)
    assert any("news_analyst" in g for g in sides_only)
    assert any("fundamentals_analyst" in g for g in sides_only)
    assert any("macro_analyst" in g for g in sides_only)
    filled = coverage_gaps(
        families=families,
        skeptical=True,
        spawned=[
            "market_analyst",
            "news_analyst",
            "fundamentals_analyst",
            "reports_analyst",
            "macro_analyst",
            "debate",
            "risk",
        ],
        reports={
            "debate": "bull open / bear rebut / bull close",
            "bull_researcher": "bull",
            "bear_researcher": "bear",
            "risk": "aggr open / cons open / neut open / rebuts",
            "risk_aggressive": "size up",
            "risk_conservative": "size down",
            "risk_neutral": "mid size",
        },
        tool_names=["get_stock_data", "get_report_insights", "get_news"],
        steps=int(cfg["min_steps_before_finish"]),
        cfg=cfg,
    )
    assert filled == []


def test_state_compact_in_place_and_overflow_detect():
    from trading_crew.harness.state import HarnessState, is_context_overflow

    state = HarnessState(ticker="SNDK")
    state.add_observation("tool", "get_news", "x" * 2000)
    state.add_observation("tool", "get_fundamentals", "y" * 2000)
    state.add_observation("spawn", "bear_researcher", "z" * 2000)
    state.add_observation("thought", "step 4", "keep going")
    saved = state.compact_in_place(keep_recent=1)
    assert saved > 0
    assert len(state.observations[0]["body"]) < 2000
    assert state.observations[-1]["body"] == "keep going"
    assert is_context_overflow(RuntimeError("maximum context length is 8192 tokens"))
    assert not is_context_overflow(RuntimeError("connection reset"))


def test_coverage_board_lists_info_desks_when_missing():
    from trading_crew.harness.state import HarnessState

    empty = HarnessState(ticker="NTNX")
    board = empty.coverage_board()
    assert "news_analyst" in board
    assert "fundamentals_analyst" in board
    assert "macro_analyst" in board
    filled = HarnessState(ticker="NTNX")
    filled.spawned = [
        "market_analyst",
        "news_analyst",
        "fundamentals_analyst",
        "reports_analyst",
        "macro_analyst",
        "debate",
        "risk",
    ]
    filled.reports = {
        "bull_researcher": "bull",
        "bear_researcher": "bear",
        "risk_aggressive": "up",
        "risk_conservative": "down",
        "risk_neutral": "mid",
        "debate": "exchange",
        "risk": "wings",
    }
    assert "Missing required lenses: (none — coverage complete)" in filled.coverage_board()


def test_spawn_many_queues_leftover_specs(monkeypatch):
    from trading_crew.harness.conductor import SpawnSpec
    from trading_crew.harness.spawn import spawn_many
    from trading_crew.harness.state import HarnessState

    cfg = load_harness_config()
    cfg = dict(cfg)
    cfg["max_parallel"] = 2
    state = HarnessState(ticker="NTNX")
    seen: list[str] = []

    def fake_one(spec, **kwargs):
        seen.append(spec.desk)
        state.spawn_used += 1
        state.spawned.append(spec.desk)
        state.reports[spec.desk] = f"{spec.desk} ok"
        return spec.desk, f"{spec.desk} ok"

    monkeypatch.setattr("trading_crew.harness.spawn.spawn_one", fake_one)
    specs = [
        SpawnSpec(kind="desk", desk="market_analyst"),
        SpawnSpec(kind="desk", desk="news_analyst"),
        SpawnSpec(kind="desk", desk="reports_analyst"),
    ]
    out = spawn_many(
        specs,
        runtime=object(),
        state=state,
        cfg=cfg,
        depth=0,
        prior="",
    )
    assert {k for k, _ in out} == {
        "market_analyst",
        "news_analyst",
        "reports_analyst",
    }
    assert set(seen) == {"market_analyst", "news_analyst", "reports_analyst"}
    assert seen[-1] == "reports_analyst"


def test_horizon_forecast_coerces_hold_and_label_confidence():
    row = HorizonForecast.model_validate({
        "stance": "HOLD",
        "horizon_days": "21d",
        "expected_return_pct": "5%",
        "confidence": "Medium-High",
        "drivers": "event crowding",
        "risks": None,
        "falsifiers": [],
        "sources": ["test"],
    })
    assert row.stance == "neutral"
    assert row.horizon_days == 21
    assert abs(row.expected_return_pct - 0.05) < 1e-9
    assert row.confidence == 0.65
    assert HorizonForecast.model_validate({
        "stance": "mixed",
        "horizon_days": 90,
        "expected_return_pct": 0.0,
        "confidence": "moderate",
        "drivers": ["x"],
        "risks": ["y"],
        "falsifiers": ["z"],
        "sources": ["t"],
    }).confidence == 0.5
    assert row.drivers == ["event crowding"]


def test_horizon_forecasts_optional_on_plain_decision():
    d = PortfolioDecision(
        action="HOLD",
        confidence=0.4,
        size_pct_of_book=0.0,
        entry_price=10.0,
        stop_loss=9.0,
        target_price=11.0,
        horizon_days=90,
        expected_return_pct=0.0,
        rationale="sit out [source: test]",
        key_drivers=["mixed"],
        key_risks=["event"],
        falsifiers=["breakout"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
        long_term_view="Long-term data unavailable for this ticker",
    )
    assert d.horizon_forecasts is None
    assert d.horizon == Horizon.MEDIUM


def test_ingest_and_query_knowledge(tmp_path, monkeypatch):
    from trading_crew.harness.ingest import ingest_corpus
    from trading_crew.harness.knowledge import query_knowledge

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "policy.txt").write_text(
        "Policy-driven tape maps a rule onto an issuer's cash, not a headline.\n\n"
        "Related-party circular capital can inflate revenue without new demand.",
        encoding="utf-8",
    )
    assert ingest_corpus(root) >= 1
    cfg = load_harness_config()
    cfg = dict(cfg)
    cfg["corpus_dir"] = root
    out = query_knowledge("circular capital related-party", cfg=cfg)
    assert "circular" in out.lower() or "related" in out.lower()
    assert "policy.txt" in out


def test_parallel_ingest_does_not_race_tmp_file(tmp_path):
    """Five harness kickoffs share one corpus; a shared .tmp used to ENOENT."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from trading_crew.harness.ingest import ingest_corpus

    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text("Order book and backlog are operating facts.\n", encoding="utf-8")
    errors = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = [pool.submit(ingest_corpus, root) for _ in range(5)]
        for fut in as_completed(futs):
            try:
                assert fut.result() >= 1
            except Exception as exc:  # noqa: BLE001 — collect every racer
                errors.append(exc)
    assert errors == []
    assert (root / ".chunks.jsonl").is_file()


def test_harness_loop_finish_requires_diverse_lenses(monkeypatch, tmp_path):
    from trading_crew.harness import loop as loop_mod
    from trading_crew.harness.conductor import ConductorTurn
    from trading_crew.harness.loop import run_harness

    cfg = load_harness_config()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("Horizon evidence standards differ by tier.", encoding="utf-8")
    cfg = dict(cfg)
    cfg["corpus_dir"] = corpus
    cfg["max_steps"] = 8
    cfg["require_debate_pair"] = False
    cfg["require_bundled_debate"] = False
    cfg["require_reports_lens"] = False
    cfg["require_market_lens"] = False
    cfg["require_news_lens"] = False
    cfg["require_fundamentals_lens"] = False
    cfg["require_macro_lens"] = False
    cfg["require_risk_lens"] = False
    cfg["require_bundled_risk"] = False
    cfg["min_steps_before_finish"] = 0

    hf = HorizonForecasts(
        SHORT=_forecast("bearish", 21, "event crowding"),
        MEDIUM=_forecast("mixed", 126, "guidance lag"),
        LONG=_forecast("bullish", 504, "reinvestment returns"),
    )
    turns = [
        ConductorTurn(action="call_tool", tool="get_stock_data", arguments={"ticker": "NTNX"}),
        ConductorTurn(action="call_tool", tool="get_fundamentals", arguments={"ticker": "NTNX"}),
        ConductorTurn(
            action="call_tool",
            tool="get_geopolitical_news",
            arguments={"ticker": "NTNX", "query": "software export control customer mix"},
        ),
        ConductorTurn(
            action="finish",
            horizon_forecasts=hf,
            primary_horizon=Horizon.MEDIUM,
            notes="three lenses in",
        ),
    ]

    class ScriptLLM:
        def call(self, messages=None, response_format=None, **kwargs):
            turn = turns.pop(0)
            return turn if response_format is None else turn

    def fake_tool(name, args, *, cfg):
        return f"{name} ok for {args.get('ticker')} q={args.get('query', '')}"

    class FakeRuntime:
        def __init__(self, *a, **k):
            self.last_pydantic = None

        def run_pm(self, **kwargs):
            self.last_pydantic = PortfolioDecision(
                action="HOLD",
                confidence=0.45,
                size_pct_of_book=0.0,
                entry_price=70.0,
                stop_loss=65.0,
                target_price=78.0,
                horizon_days=126,
                expected_return_pct=0.0,
                rationale="mixed medium [source: test]",
                key_drivers=["guidance lag"],
                key_risks=["policy"],
                falsifiers=["beat and raise"],
                geopolitical_flags=["export control"],
                compliance_status="CLEAR",
                long_term_view="compounder not proven",
            )
            return self.last_pydantic.rationale

    monkeypatch.setattr(loop_mod, "invoke_tool", fake_tool)
    monkeypatch.setattr(loop_mod, "GraphRuntime", FakeRuntime)
    monkeypatch.setattr(loop_mod, "ingest_corpus", lambda path: 1)

    class Crew:
        ticker = "NTNX"

    result = run_harness(Crew(), {}, llm=ScriptLLM(), cfg=cfg)
    assert result.pydantic.action == "HOLD"
    assert result.pydantic.horizon_forecasts.SHORT.stance == "bearish"
    assert result.pydantic.horizon_forecasts.LONG.stance == "bullish"
    assert result.state.skeptical is True


def test_spawn_debate_fills_both_sides(monkeypatch):
    from trading_crew.harness.conductor import SpawnSpec
    from trading_crew.harness.spawn import spawn_one
    from trading_crew.harness.state import HarnessState

    cfg = load_harness_config()
    cfg = dict(cfg)
    captured = {}

    def writer(**kwargs):
        captured.update(kwargs)
        return (
            "1. Open a cited bull case for BYND.\n"
            "2. Rebut with tools if a claim lacks a source."
        )

    cfg["planner_writer"] = writer
    state = HarnessState(ticker="BYND")

    class FakeRuntime:
        last_raw = "## Bundled debate — Bull open\nbull\n\n## Bundled debate — Bear rebuttal\nbear"
        prior = ""

        def run_bundled_debate(self, prior=""):
            self.prior = prior
            return "bull case [source: https://sec.gov/a]", "bear case [source: https://sec.gov/b]"

    runtime = FakeRuntime()
    key, text = spawn_one(
        SpawnSpec(kind="desk", desk="debate", brief="argue the name"),
        runtime=runtime,
        state=state,
        cfg=cfg,
        depth=0,
        prior="",
    )
    assert key == "debate"
    assert "bull" in text.lower()
    assert state.reports["bull_researcher"].startswith("bull")
    assert state.reports["bear_researcher"].startswith("bear")
    assert "debate" in state.spawned
    assert state.debate_used == 1
    assert state.skeptical is True
    assert "## User request (do this spawn thoroughly)" in runtime.prior
    assert captured["desk"] == "debate"


def test_conductor_coerces_debate_kind():
    from trading_crew.harness.conductor import ConductorTurn

    turn = ConductorTurn.model_validate({
        "action": "spawn",
        "agents": [{"kind": "debate", "brief": "live exchange"}],
    })
    assert turn.agents[0].kind == "desk"
    assert turn.agents[0].desk == "debate"


def test_spawn_risk_fills_three_wings(monkeypatch):
    from trading_crew.harness.conductor import SpawnSpec
    from trading_crew.harness.spawn import spawn_one
    from trading_crew.harness.state import HarnessState

    cfg = load_harness_config()
    cfg = dict(cfg)
    cfg["planner_writer"] = lambda **k: (
        "1. Each risk wing writes an independent size opinion on BYND.\n"
        "2. Debate those opinions with cited tape."
    )
    state = HarnessState(ticker="BYND")

    class FakeRuntime:
        last_raw = "## Bundled risk — Aggressive open\nsize up"
        prior = ""

        def run_bundled_risk(self, prior=""):
            self.prior = prior
            return "size up", "size down", "mid size"

    runtime = FakeRuntime()
    key, text = spawn_one(
        SpawnSpec(kind="desk", desk="risk", brief="two-round risk debate"),
        runtime=runtime,
        state=state,
        cfg=cfg,
        depth=0,
        prior="",
    )
    assert key == "risk"
    assert "size" in text.lower()
    assert state.reports["risk_aggressive"] == "size up"
    assert state.reports["risk_conservative"] == "size down"
    assert state.reports["risk_neutral"] == "mid size"
    assert "risk" in state.spawned
    assert state.risk_used == 1
    assert state.skeptical is True
    assert "## User request (do this spawn thoroughly)" in runtime.prior


def test_conductor_coerces_risk_kind():
    from trading_crew.harness.conductor import ConductorTurn

    turn = ConductorTurn.model_validate({
        "action": "spawn",
        "agents": [{"kind": "risk", "brief": "two-round debate"}],
    })
    assert turn.agents[0].kind == "desk"
    assert turn.agents[0].desk == "risk"


def test_harness_exhaust_returns_last_pm(monkeypatch, tmp_path):
    """BYND-style: stance mismatch must still return a ticket at max_steps."""
    from trading_crew.harness import loop as loop_mod
    from trading_crew.harness.conductor import ConductorTurn
    from trading_crew.harness.loop import run_harness

    cfg = load_harness_config()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("Horizon evidence standards differ by tier.", encoding="utf-8")
    cfg = dict(cfg)
    cfg["corpus_dir"] = corpus
    cfg["max_steps"] = 1
    cfg["exhaust_steps"] = 0
    cfg["require_debate_pair"] = False
    cfg["require_bundled_debate"] = False
    cfg["require_reports_lens"] = False
    cfg["require_market_lens"] = False
    cfg["require_news_lens"] = False
    cfg["require_fundamentals_lens"] = False
    cfg["require_macro_lens"] = False
    cfg["require_risk_lens"] = False
    cfg["require_bundled_risk"] = False
    cfg["min_steps_before_finish"] = 0
    cfg["min_tool_families"] = 0
    cfg["require_skeptical_lens"] = False

    hf = HorizonForecasts(
        SHORT=_forecast("bearish", 21, "overbought rsi"),
        MEDIUM=_forecast("bearish", 126, "leverage"),
        LONG=_forecast("bearish", 504, "shrinking revenue", "second long driver"),
    )
    turns = [
        ConductorTurn(
            action="finish",
            horizon_forecasts=hf,
            primary_horizon=Horizon.MEDIUM,
            notes="bearish tape vs hold pm",
        ),
    ]

    class ScriptLLM:
        def call(self, messages=None, response_format=None, **kwargs):
            return turns.pop(0)

    def fake_tool(name, args, *, cfg):
        return f"{name} ok"

    class FakeRuntime:
        def __init__(self, *a, **k):
            self.last_pydantic = None

        def run_pm(self, **kwargs):
            self.last_pydantic = PortfolioDecision(
                action="HOLD",
                confidence=0.6,
                size_pct_of_book=0.0,
                entry_price=13.86,
                stop_loss=12.0,
                target_price=13.86,
                horizon_days=21,
                expected_return_pct=0.0,
                rationale="split evidence [source: test]",
                key_drivers=["guide raise"],
                key_risks=["leverage"],
                falsifiers=["margin > 35%"],
                geopolitical_flags=[],
                compliance_status="CLEAR",
                long_term_view="compounder thesis is broken",
            )
            return self.last_pydantic.rationale

    monkeypatch.setattr(loop_mod, "invoke_tool", fake_tool)
    monkeypatch.setattr(loop_mod, "GraphRuntime", FakeRuntime)
    monkeypatch.setattr(loop_mod, "ingest_corpus", lambda path: 1)

    class Crew:
        ticker = "BYND"

    result = run_harness(Crew(), {}, llm=ScriptLLM(), cfg=cfg)
    assert result.pydantic.action == "HOLD"
    assert result.pydantic.entry_price == 13.86
    assert "without a gated finish" in result.pydantic.rationale
    assert result.state.observations[-1]["title"] == "budget exhausted"


def test_harness_exhaust_runs_pm_when_finish_never_cleared(monkeypatch, tmp_path):
    from trading_crew.harness import loop as loop_mod
    from trading_crew.harness.conductor import ConductorTurn
    from trading_crew.harness.loop import run_harness

    cfg = load_harness_config()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("Horizon evidence standards differ by tier.", encoding="utf-8")
    cfg = dict(cfg)
    cfg["corpus_dir"] = corpus
    cfg["max_steps"] = 1
    cfg["exhaust_steps"] = 0
    cfg["min_steps_before_finish"] = 8

    turns = [ConductorTurn(action="finish", notes="too early")]

    class ScriptLLM:
        def call(self, messages=None, response_format=None, **kwargs):
            return turns.pop(0)

    def fake_tool(name, args, *, cfg):
        return f"{name} ok"

    class FakeRuntime:
        def __init__(self, *a, **k):
            self.last_pydantic = None

        def run_pm(self, **kwargs):
            self.last_pydantic = PortfolioDecision(
                action="HOLD",
                confidence=0.5,
                size_pct_of_book=0.0,
                entry_price=1.0,
                stop_loss=1.0,
                target_price=1.0,
                horizon_days=21,
                expected_return_pct=0.0,
                rationale="budget end [source: test]",
                key_drivers=["tape"],
                key_risks=["gap"],
                falsifiers=["print"],
                geopolitical_flags=[],
                compliance_status="CLEAR",
                long_term_view="not locked",
            )
            return self.last_pydantic.rationale

    monkeypatch.setattr(loop_mod, "invoke_tool", fake_tool)
    monkeypatch.setattr(loop_mod, "GraphRuntime", FakeRuntime)
    monkeypatch.setattr(loop_mod, "ingest_corpus", lambda path: 1)

    class Crew:
        ticker = "PLUG"

    result = run_harness(Crew(), {}, llm=ScriptLLM(), cfg=cfg)
    assert result.pydantic.action == "HOLD"
    assert "without a gated finish" in result.pydantic.rationale


def test_harness_exhaust_hold_when_pm_fails(monkeypatch, tmp_path):
    from trading_crew.harness import loop as loop_mod
    from trading_crew.harness.conductor import ConductorTurn
    from trading_crew.harness.loop import run_harness

    cfg = load_harness_config()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("Horizon evidence standards evenly differ.", encoding="utf-8")
    cfg = dict(cfg)
    cfg["corpus_dir"] = corpus
    cfg["max_steps"] = 1
    cfg["exhaust_steps"] = 0
    cfg["min_steps_before_finish"] = 8

    turns = [ConductorTurn(action="finish", notes="too early")]

    class ScriptLLM:
        def call(self, messages=None, response_format=None, **kwargs):
            return turns.pop(0)

    def fake_tool(name, args, *, cfg):
        return f"{name} ok"

    class FakeRuntime:
        def __init__(self, *a, **k):
            self.last_pydantic = None

        def run_pm(self, **kwargs):
            raise RuntimeError("pm down")

    monkeypatch.setattr(loop_mod, "invoke_tool", fake_tool)
    monkeypatch.setattr(loop_mod, "GraphRuntime", FakeRuntime)
    monkeypatch.setattr(loop_mod, "ingest_corpus", lambda path: 1)

    class Crew:
        ticker = "IDEA.NS"

    result = run_harness(Crew(), {}, llm=ScriptLLM(), cfg=cfg)
    assert result.pydantic.action == "HOLD"
    assert result.pydantic.size_pct_of_book == 0.0
    assert "did not emit a valid ticket" in result.pydantic.rationale


def test_conductor_prompt_urges_finish_when_few_steps_remain():
    from trading_crew.harness.conductor import build_closing_prompt, build_system_prompt

    late = build_system_prompt(
        ticker="BEL.NS",
        standing="rules",
        allowlist=["get_stock_data"],
        catalog=["market_analyst"],
        spawn_left=2,
        steps_left=3,
        coverage="Missing required lenses: news_analyst",
    )
    assert "few steps remain" in late
    assert "Do not open another bundled debate" in late
    early = build_system_prompt(
        ticker="BEL.NS",
        standing="rules",
        allowlist=["get_stock_data"],
        catalog=["market_analyst"],
        spawn_left=2,
        steps_left=20,
        coverage="",
    )
    assert "few steps remain" not in early
    closing = build_closing_prompt(
        ticker="BEL.NS",
        standing="rules",
        coverage="Missing required lenses: news_analyst",
        steps_left=3,
    )
    assert "CLOSING PHASE" in closing
    assert "Do not spawn" in closing
    assert "step budget failed" in closing


def test_hold_from_reports_cites_desk_text():
    from trading_crew.harness.loop import _hold_from_reports
    from trading_crew.harness.state import HarnessState

    state = HarnessState(ticker="BEL.NS")
    state.reports["bull_researcher"] = "Order book supports 12% upside [source: ar]"
    state.reports["bear_researcher"] = "Valuation stretched vs peers [source: tape]"
    pm = _hold_from_reports("BEL.NS", 48, state)
    assert pm.action == "HOLD"
    assert pm.size_pct_of_book == 0.0
    assert "Order book supports 12% upside" in pm.rationale
    assert "Valuation stretched vs peers" in pm.rationale
    assert "did not emit a valid ticket" not in pm.rationale


def test_harness_exhaust_hold_cites_desk_reports(monkeypatch, tmp_path):
    from trading_crew.harness import loop as loop_mod
    from trading_crew.harness.conductor import ConductorTurn
    from trading_crew.harness.loop import run_harness
    from trading_crew.harness.state import HarnessState

    cfg = load_harness_config()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("Horizon evidence standards evenly differ.", encoding="utf-8")
    cfg = dict(cfg)
    cfg["corpus_dir"] = corpus
    cfg["max_steps"] = 1
    cfg["exhaust_steps"] = 0
    cfg["min_steps_before_finish"] = 8

    turns = [ConductorTurn(action="finish", notes="too early")]
    real_init = HarnessState.__init__

    def seeded_init(self, ticker):
        real_init(self, ticker)
        self.reports["bull_researcher"] = "Order book supports 12% upside [source: ar]"
        self.reports["bear_researcher"] = "Valuation stretched vs peers [source: tape]"

    class ScriptLLM:
        def call(self, messages=None, response_format=None, **kwargs):
            return turns.pop(0)

    def fake_tool(name, args, *, cfg):
        return f"{name} ok"

    class FakeRuntime:
        def __init__(self, *a, **k):
            self.last_pydantic = None

        def run_pm(self, **kwargs):
            raise RuntimeError("pm down")

    monkeypatch.setattr(loop_mod, "HarnessState", HarnessState)
    monkeypatch.setattr(HarnessState, "__init__", seeded_init)
    monkeypatch.setattr(loop_mod, "invoke_tool", fake_tool)
    monkeypatch.setattr(loop_mod, "GraphRuntime", FakeRuntime)
    monkeypatch.setattr(loop_mod, "ingest_corpus", lambda path: 1)

    class Crew:
        ticker = "BEL.NS"

    result = run_harness(Crew(), {}, llm=ScriptLLM(), cfg=cfg)
    assert result.pydantic.action == "HOLD"
    assert "Order book supports 12% upside" in result.pydantic.rationale
    assert "Valuation stretched vs peers" in result.pydantic.rationale
    assert "did not emit a valid ticket" not in result.pydantic.rationale


def test_harness_closing_finish_writes_reasoned_hold(monkeypatch, tmp_path):
    from trading_crew.harness import loop as loop_mod
    from trading_crew.harness.conductor import ConductorTurn
    from trading_crew.harness.loop import run_harness

    cfg = load_harness_config()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.txt").write_text("Horizon evidence standards differ by tier.", encoding="utf-8")
    cfg = dict(cfg)
    cfg["corpus_dir"] = corpus
    cfg["max_steps"] = 1
    cfg["exhaust_steps"] = 2
    cfg["min_steps_before_finish"] = 8
    cfg["require_debate_pair"] = False
    cfg["require_bundled_debate"] = False
    cfg["require_reports_lens"] = False
    cfg["require_market_lens"] = False
    cfg["require_news_lens"] = False
    cfg["require_fundamentals_lens"] = False
    cfg["require_macro_lens"] = False
    cfg["require_risk_lens"] = False
    cfg["require_bundled_risk"] = False
    cfg["min_tool_families"] = 0
    cfg["require_skeptical_lens"] = False

    hf = HorizonForecasts(
        SHORT=_forecast("bullish", 21, "order book"),
        MEDIUM=_forecast("mixed", 126, "valuation stretch"),
        LONG=_forecast("bearish", 504, "defence cycle peak"),
    )
    turns = [
        ConductorTurn(action="finish", notes="too early"),
        ConductorTurn(
            action="spawn",
            agents=[{"kind": "desk", "desk": "debate", "brief": "another round"}],
        ),
        ConductorTurn(
            action="finish",
            horizon_forecasts=hf,
            primary_horizon=Horizon.MEDIUM,
            thought="bull order-book vs bear valuation — HOLD",
        ),
    ]
    captured = {}

    class ScriptLLM:
        def call(self, messages=None, response_format=None, **kwargs):
            return turns.pop(0)

    def fake_tool(name, args, *, cfg):
        return f"{name} ok"

    class FakeRuntime:
        def __init__(self, *a, **k):
            self.last_pydantic = None

        def run_pm(self, **kwargs):
            captured.update(kwargs)
            self.last_pydantic = PortfolioDecision(
                action="HOLD",
                confidence=0.55,
                size_pct_of_book=0.0,
                entry_price=411.9,
                stop_loss=381.48,
                target_price=457.53,
                horizon_days=21,
                expected_return_pct=0.11,
                rationale=(
                    "Bull cites a defence order book; bear cites a stretched "
                    "multiple versus peers. Conflicted — HOLD [source: test]"
                ),
                key_drivers=["order book vs multiple"],
                key_risks=["cycle peak"],
                falsifiers=["guidance cut"],
                geopolitical_flags=[],
                compliance_status="CLEAR",
                long_term_view="defence compounder not proven this cycle",
            )
            return self.last_pydantic.rationale

    monkeypatch.setattr(loop_mod, "invoke_tool", fake_tool)
    monkeypatch.setattr(loop_mod, "GraphRuntime", FakeRuntime)
    monkeypatch.setattr(loop_mod, "ingest_corpus", lambda path: 1)

    class Crew:
        ticker = "BEL.NS"

    result = run_harness(Crew(), {}, llm=ScriptLLM(), cfg=cfg)
    assert result.pydantic.action == "HOLD"
    assert "order book" in result.pydantic.rationale.lower()
    assert "did not emit a valid ticket" not in result.pydantic.rationale
    assert "without a gated finish" not in result.pydantic.rationale
    assert "CLOSING" in (captured.get("thesis") or "")
    assert any(o["title"] == "closing finish" for o in result.state.observations)
