"""Mass 1PO/2TO promote gate + resolve-day tape reason (training only)."""

from __future__ import annotations

from pathlib import Path

from trading_crew.agentic.mass import (
    evaluate_cohort,
    fetch_tape_reason,
    on_episode_resolved,
    promote,
)
from trading_crew.agentic.memory.episodic import Episode, OutcomeStatus, Regime
from trading_crew.agentic.outcome import OutcomeLabel
from trading_crew.graph.routers import resolve_playbook_action


def _ep(**kw):
    base = dict(
        episode_id="NTNX-1",
        symbol="NTNX",
        decision_ts="2026-05-01T00:00:00+00:00",
        state_summary="breakout",
        regime=Regime.TREND,
        action_proposal={"side": "BUY", "target_weight": 0.1},
        outcome_ts="2026-06-01T00:00:00+00:00",
        outcome_status=OutcomeStatus.RESOLVED,
        realised_return=0.08,
        alpha_return=0.03,
        outcome_label=OutcomeLabel.WIN.value,
        reflection_meta={
            "desk_lessons": [
                {
                    "desk": "news_analyst",
                    "lean_at_decision": "bullish",
                    "verdict": "helped",
                    "what_they_got_wrong": "",
                    "recommended_change": "",
                }
            ]
        },
    )
    base.update(kw)
    return Episode(**base)


def test_promote_refuses_when_win_rate_or_alpha_lose(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    ok = promote(
        candidate_action="debate_rounds_2",
        candidate_metrics={"n": 10, "win_rate": 0.40, "mean_alpha": 0.02},
        current_metrics={"n": 10, "win_rate": 0.50, "mean_alpha": 0.01},
        playbook_key_name="US",
    )
    assert ok is False
    ok = promote(
        candidate_action="debate_rounds_2",
        candidate_metrics={"n": 10, "win_rate": 0.60, "mean_alpha": 0.00},
        current_metrics={"n": 10, "win_rate": 0.50, "mean_alpha": 0.02},
        playbook_key_name="US",
    )
    assert ok is False


def test_promote_writes_action_when_both_metrics_beat(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    ok = promote(
        candidate_action="debate_rounds_2",
        desk_instructions={"news_analyst": "Weight catalysts, not headlines."},
        candidate_metrics={"n": 8, "win_rate": 0.62, "mean_alpha": 0.03},
        current_metrics={"n": 8, "win_rate": 0.50, "mean_alpha": 0.01},
        playbook_key_name="US",
    )
    assert ok is True
    assert resolve_playbook_action("US") == "debate_rounds_2"


def test_promote_refuses_small_n(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    assert promote(
        candidate_metrics={"n": 2, "win_rate": 1.0, "mean_alpha": 0.5},
        current_metrics={"n": 10, "win_rate": 0.1, "mean_alpha": 0.0},
    ) is False


def test_evaluate_cohort_win_rate():
    wins = [_ep(episode_id=f"W{i}", outcome_label="WIN", alpha_return=0.02) for i in range(3)]
    loss = [_ep(episode_id="L1", outcome_label="LOSS", alpha_return=-0.01)]
    m = evaluate_cohort(wins + loss)
    assert m["n"] == 4
    assert m["win_rate"] == 0.75


def test_1po_writes_candidate_on_resolve(tmp_path, monkeypatch):
    monkeypatch.setenv("TRADINGCREW_CACHE_DIR", str(tmp_path))
    ep = _ep(
        outcome_label="LOSS",
        alpha_return=-0.04,
        market_reason="Missed guidance [source: https://sec.gov/x]",
        reflection_meta={
            "desk_lessons": [
                {
                    "desk": "news_analyst",
                    "lean_at_decision": "bullish",
                    "verdict": "hurt",
                    "what_they_got_wrong": "ignored the print",
                    "recommended_change": "re-read guidance",
                }
            ]
        },
    )
    on_episode_resolved(ep, skip_search=True)
    files = list((tmp_path / "v3" / "candidates" / "news_analyst").glob("*.json"))
    assert files
    text = files[0].read_text(encoding="utf-8")
    assert "Missed guidance" in text
    assert "NTNX" in text


def test_tape_reason_skipped_in_pit(monkeypatch):
    from trading_crew.data_sources import pit

    monkeypatch.setattr(pit, "is_backfill", lambda: True)
    assert fetch_tape_reason("NTNX", "2026-08-01", -0.04, "BUY") == ""


def test_live_graph_and_runner_do_not_read_market_reason():
    root = Path(__file__).resolve().parents[1]
    for rel in (
        "trading_crew/graph/app.py",
        "trading_crew/graph/nodes.py",
        "trading_crew/graph/routers.py",
        "web/backend/runner.py",
        "trading_crew/crew.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert "market_reason" not in text
        assert "fetch_tape_reason" not in text
