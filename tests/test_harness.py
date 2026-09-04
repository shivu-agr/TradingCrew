"""Hermes diagnosis harness — isolated cache, synthetic 30-day books."""

from __future__ import annotations

from trading_crew.agentic.harness import (
    apply_harness,
    clear_sample_and_restore,
    seed_and_step,
    seed_harness_book,
    step_harness,
)
from trading_crew.agentic.memory.episodic import EpisodicMemory
from trading_crew.agentic.self_learn import active_experiment, journal_path, load_journal, save_journal
from trading_crew.agentic.strategy import load_desk_strategy
from trading_crew.graph.routers import resolve_playbook_action


class _Agent:
    def __init__(self, role: str, backstory: str = "base backstory") -> None:
        self.role = role
        self.backstory = backstory


def _agents() -> dict:
    return {
        "trader": _Agent("Trader"),
        "research_manager": _Agent("Research Manager"),
        "portfolio_manager": _Agent("Portfolio Manager"),
        "fundamentals_analyst": _Agent("Fundamentals Analyst"),
        "macro_analyst": _Agent("Macro Analyst"),
        "market_analyst": _Agent("Market Analyst"),
    }


def _step(tmp_path, monkeypatch, *, pattern: str, fail: bool = True):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path / pattern))
    mem = EpisodicMemory(tmp_path / pattern / "memory" / "episodes.jsonl")
    journal = journal_path()
    s = load_desk_strategy()
    n = s.min_reviews_to_change
    seeded = seed_harness_book(count=n, memory=mem, fail=fail, pattern=pattern)
    out = step_harness(mem.all_episodes(), journal_file=journal)
    return s, mem, journal, seeded, out


def test_side_opens_debate_rounds(tmp_path, monkeypatch):
    s, mem, journal, seeded, out = _step(tmp_path, monkeypatch, pattern="side")
    assert seeded["seeded"] == s.min_reviews_to_change
    assert out["action"] == "open"
    assert out["score"]["verdict"] == "FAILURE"
    assert (out.get("diagnosis") or {}).get("rule") == "wrong_side"
    exp = active_experiment(journal)
    assert exp is not None
    assert exp["knob"] == "debate_rounds_2"
    assert exp["apply"] == "playbook"
    assert exp["playbook_action"] == "debate_rounds_2"
    assert not (exp.get("desk_prompts") or {})
    assert "ingested" not in out

    agents = _agents()
    apply_harness(agents, journal_file=journal)
    assert "ACTIVE LEARNING EXPERIMENT" in agents["trader"].backstory
    assert "debate_rounds_2" in agents["trader"].backstory
    assert "DESK STRATEGY" in agents["research_manager"].backstory
    assert "HARNESS DESK ADDENDUM" not in agents["fundamentals_analyst"].backstory
    assert "HARNESS DESK ADDENDUM" not in agents["market_analyst"].backstory
    assert resolve_playbook_action("US") == "debate_rounds_2"

    from trading_crew.agentic.harness import dashboard_payload

    board = dashboard_payload(mem.all_episodes(), journal_file=journal)
    layer_ids = [layer["id"] for layer in board["live_layers"]]
    assert "synthesis_prompt" in layer_ids
    assert "topology" in layer_ids
    assert "size" not in layer_ids
    assert board["hermes_size_scale"] is None
    assert board["min_reviews_to_change"] == load_desk_strategy().min_reviews_to_change
    assert board["one_variable_at_a_time"] is True
    assert "debate_rounds_2" in board["used_knobs"]


def test_alpha_opens_size_scale_down(tmp_path, monkeypatch):
    _, _, journal, _, out = _step(tmp_path, monkeypatch, pattern="alpha", fail=False)
    assert out["action"] == "open"
    assert (out.get("diagnosis") or {}).get("rule") == "size_not_side"
    exp = active_experiment(journal)
    assert exp is not None
    assert exp["knob"] == "size_scale_down"
    assert exp["apply"] == "prompt"
    assert exp["size_scale"] == 0.75
    agents = _agents()
    apply_harness(agents, journal_file=journal)
    assert "size_scale_down" in agents["trader"].backstory
    assert "HARNESS DESK ADDENDUM" not in agents["fundamentals_analyst"].backstory

    from trading_crew.agentic.harness import dashboard_payload

    board = dashboard_payload(
        EpisodicMemory(tmp_path / "alpha" / "memory" / "episodes.jsonl").all_episodes(),
        journal_file=journal,
    )
    layer_ids = [layer["id"] for layer in board["live_layers"]]
    assert "synthesis_prompt" in layer_ids
    assert "size" in layer_ids
    assert "topology" not in layer_ids
    assert board["hermes_size_scale"] == 0.75


def test_desk_opens_desk_prompt(tmp_path, monkeypatch):
    _, _, journal, _, out = _step(tmp_path, monkeypatch, pattern="desk")
    assert out["action"] == "open"
    assert (out.get("diagnosis") or {}).get("rule") == "dominant_hurt_desk"
    exp = active_experiment(journal)
    assert exp is not None
    assert exp["knob"] == "desk:fundamentals_analyst"
    assert exp["apply"] == "desk_prompt"
    desks = exp.get("desk_prompts") or {}
    assert list(desks) == ["fundamentals_analyst"]
    assert "NET_GUIDANCE_BIAS" in desks["fundamentals_analyst"]

    agents = _agents()
    applied = apply_harness(agents, journal_file=journal)
    assert "ACTIVE LEARNING EXPERIMENT" in agents["trader"].backstory
    assert "desk:fundamentals_analyst" in agents["trader"].backstory
    assert "HARNESS DESK ADDENDUM" in agents["fundamentals_analyst"].backstory
    assert "HARNESS DESK ADDENDUM" not in agents["macro_analyst"].backstory
    assert "HARNESS DESK ADDENDUM" not in agents["market_analyst"].backstory
    assert "fundamentals_analyst" in applied["applied_desks"]

    from trading_crew.agentic.harness import dashboard_payload

    board = dashboard_payload(
        EpisodicMemory(tmp_path / "desk" / "memory" / "episodes.jsonl").all_episodes(),
        journal_file=journal,
    )
    layer_ids = [layer["id"] for layer in board["live_layers"]]
    assert "desk_prompt" in layer_ids
    assert "topology" not in layer_ids


def test_reports_unhelpful_opens_skip_reports(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from trading_crew.agentic.harness import collect_evidence, propose_experiment
    from trading_crew.agentic.objectives import ObjectiveScore
    from trading_crew.agentic.objectives import HorizonMetrics

    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path / "reports"))
    s = load_desk_strategy()
    eps = []
    for i in range(s.min_reviews_to_change):
        eps.append(SimpleNamespace(
            episode_id=f"R{i}",
            outcome_status="RESOLVED",
            outcome_label="LOSS",
            review_status="RESOLVED",
            review={
                "outcome_label": "LOSS",
                "reflection_meta": {
                    "desk_lessons": [{
                        "desk": "reports_analyst",
                        "verdict": "hurt",
                        "what_they_got_wrong": "Filings print was stale.",
                        "recommended_change": "",
                    }],
                    "unhelpful_tools": ["get_report_insights"],
                    "overemphasized": ["filings"],
                    "suggested_playbook": "skip_reports",
                },
            },
            action_proposal={"side": "BUY"},
            desk_leans={"reports_analyst": "bullish"},
            reflection_meta=None,
        ))
    ev = collect_evidence(eps)
    assert ev["dominant_desk"] == "reports_analyst"
    score = ObjectiveScore(
        tier="short",
        verdict="FAILURE",
        metrics=HorizonMetrics(
            n=len(eps), wins=0, losses=len(eps),
            hit_rate=0.3, mean_alpha=-0.02, mean_drawdown=-0.05,
            expectancy=-0.01, profit_factor=0.8, mean_realised=-0.01,
        ),
        reasons=["hit-rate"],
    )
    first = propose_experiment(eps, score, s, [])
    assert first is not None
    assert first["knob"] == "desk:reports_analyst"
    exp = propose_experiment(eps, score, s, ["desk:reports_analyst"])
    assert exp is not None
    assert exp["knob"] == "skip_reports"
    assert (exp.get("diagnosis") or {}).get("rule") == "reports_unhelpful"


def test_split_opens_extra_debate(tmp_path, monkeypatch):
    _, _, journal, _, out = _step(tmp_path, monkeypatch, pattern="split")
    assert out["action"] == "open"
    assert (out.get("diagnosis") or {}).get("rule") == "split_boards"
    exp = active_experiment(journal)
    assert exp is not None
    assert exp["knob"] == "extra_debate_on_split"
    assert exp["playbook_action"] == "extra_debate_on_split"
    assert resolve_playbook_action("US") == "extra_debate_on_split"


def test_second_failure_walks_catalog_after_used_evidence(tmp_path, monkeypatch):
    _, mem, journal, _, first = _step(tmp_path, monkeypatch, pattern="side")
    assert first["experiment"]["knob"] == "debate_rounds_2"
    data = load_journal(journal)
    data["experiment"] = None
    save_journal(data, journal)
    out = step_harness(mem.all_episodes(), journal_file=journal)
    exp = active_experiment(journal)
    assert exp is not None
    assert exp["knob"] == "size_scale_down"
    assert exp["size_scale"] == 0.75
    assert out["action"] == "open"
    assert (out.get("diagnosis") or {}).get("rule") == "yaml_catalog"


def test_seed_and_clear_restores_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    mem = EpisodicMemory(tmp_path / "memory" / "episodes.jsonl")
    journal = journal_path()
    journal.parent.mkdir(parents=True, exist_ok=True)
    save_journal(
        {
            "experiment": {
                "knob": "guidance_weight",
                "apply": "prompt",
                "status": "open",
                "hypothesis": "live book experiment",
            },
            "used_knobs": ["guidance_weight"],
            "history": [],
            "last_score": None,
        },
        journal,
    )
    s = load_desk_strategy()
    seeded = seed_and_step(
        count=s.min_reviews_to_change, memory=mem, journal_file=journal, fail=True,
    )
    assert seeded["seeded"] == s.min_reviews_to_change
    assert seeded["journal_snapshotted"] is True
    live = active_experiment(journal)
    assert live is not None
    assert live["knob"] == "debate_rounds_2"
    cleared = clear_sample_and_restore(memory=mem, journal_file=journal)
    assert cleared["removed"] == s.min_reviews_to_change
    assert cleared["journal_restored"] is True
    restored = active_experiment(journal)
    assert restored is not None
    assert restored["knob"] == "guidance_weight"
    assert not any(
        str(e.episode_id).startswith("SAMPLE-") for e in mem.all_episodes()
    )


def test_sequential_crew_list_gets_strategy_block(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    mem = EpisodicMemory(tmp_path / "memory" / "episodes.jsonl")
    journal = journal_path()
    s = load_desk_strategy()
    seed_harness_book(count=s.min_reviews_to_change, memory=mem, fail=True, pattern="side")
    step_harness(mem.all_episodes(), journal_file=journal)
    agents = list(_agents().values())
    apply_harness(agents, journal_file=journal)
    trader = next(a for a in agents if a.role == "Trader")
    market = next(a for a in agents if a.role == "Market Analyst")
    assert "ACTIVE LEARNING EXPERIMENT" in trader.backstory
    assert "ACTIVE LEARNING EXPERIMENT" not in market.backstory


def test_ui_open_close_and_clear_used(tmp_path, monkeypatch):
    from trading_crew.agentic.self_learn import (
        clear_used_knobs_ui,
        close_experiment_ui,
        open_knob_ui,
    )

    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    mem = EpisodicMemory(tmp_path / "memory" / "episodes.jsonl")
    journal = journal_path()
    opened = open_knob_ui("debate_rounds_1", mem.all_episodes(), journal_file=journal)
    assert opened["experiment"]["knob"] == "debate_rounds_1"
    assert active_experiment(journal)["playbook_action"] == "debate_rounds_1"
    try:
        open_knob_ui("skip_reports", mem.all_episodes(), journal_file=journal)
        raise AssertionError("second open should fail")
    except ValueError as exc:
        assert "already open" in str(exc)
    closed = close_experiment_ui(journal_file=journal)
    assert closed["experiment"] is None
    assert active_experiment(journal) is None
    assert load_journal(journal)["used_knobs"] == ["debate_rounds_1"]
    cleared = clear_used_knobs_ui(journal_file=journal)
    assert cleared["used_knobs"] == []
    assert load_journal(journal)["used_knobs"] == []


def test_disabled_skip_reports_not_proposed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from trading_crew.agentic.harness import propose_experiment
    from trading_crew.agentic.objectives import HorizonMetrics, ObjectiveScore
    from trading_crew.agentic.strategy import load_desk_strategy, parse_desk_strategy, strategy_to_raw

    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path / "off"))
    raw = strategy_to_raw(load_desk_strategy())
    for k in raw["knobs"]:
        if k["id"] == "skip_reports":
            k["enabled"] = False
    s = parse_desk_strategy(raw)
    n = s.min_reviews_to_change
    eps = []
    for i in range(n):
        eps.append(SimpleNamespace(
            episode_id=f"R{i}",
            outcome_status="RESOLVED",
            outcome_label="LOSS",
            review_status="RESOLVED",
            review={
                "outcome_label": "LOSS",
                "reflection_meta": {
                    "desk_lessons": [],
                    "unhelpful_tools": ["get_report_insights"],
                    "overemphasized": ["filings"],
                },
            },
            action_proposal={"side": "BUY"},
            desk_leans={},
            reflection_meta=None,
        ))
    score = ObjectiveScore(
        tier="short",
        verdict="FAILURE",
        metrics=HorizonMetrics(
            n=n, wins=0, losses=n,
            hit_rate=0.3, mean_alpha=-0.02, mean_drawdown=-0.05,
            expectancy=-0.01, profit_factor=0.8, mean_realised=-0.01,
        ),
        reasons=["hit-rate"],
    )
    exp = propose_experiment(eps, score, s, ["desk:reports_analyst"])
    assert exp is None or exp["knob"] != "skip_reports"
