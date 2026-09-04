"""One-knob scientific loop — never promote from a single LOSS."""

from __future__ import annotations

from trading_crew.agentic.memory.episodic import Episode, OutcomeStatus, Regime
from trading_crew.agentic.self_learn import active_experiment, load_journal, step_self_learn
from trading_crew.agentic.strategy import load_desk_strategy


def _ep(i: int, *, label: str, alpha: float = -0.03, realised: float = -0.04, dd: float = -0.05) -> Episode:
    return Episode(
        episode_id=f"X-{i}",
        symbol="X",
        decision_ts="2026-01-01T00:00:00+00:00",
        state_summary="test",
        regime=Regime.TREND,
        action_proposal={"side": "BUY", "horizon_days": 180},
        outcome_ts="2026-07-01T00:00:00+00:00",
        outcome_status=OutcomeStatus.PENDING,
        review_status="RESOLVED",
        review={
            "outcome_label": label,
            "alpha_return": alpha,
            "realised_return": realised,
            "max_drawdown": dd,
        },
    )


def test_single_loss_does_not_open_a_knob(tmp_path):
    journal = tmp_path / "journal.json"
    out = step_self_learn([_ep(1, label="LOSS")], journal_file=journal)
    assert out["action"] == "wait"
    assert active_experiment(journal) is None


def test_failure_opens_exactly_one_knob(tmp_path):
    s = load_desk_strategy()
    n = s.min_reviews_to_change
    eps = [_ep(i, label="LOSS") for i in range(n)]
    journal = tmp_path / "journal.json"
    out = step_self_learn(eps, journal_file=journal, strategy=s)
    assert out["action"] == "open"
    exp = active_experiment(journal)
    assert exp is not None
    assert exp["knob"] == s.knobs[0].id
    # Second FAILURE with the same open experiment closes it — no second knob.
    out2 = step_self_learn(eps, journal_file=journal, strategy=s)
    assert out2["action"] == "reject"
    assert active_experiment(journal) is None


def test_lookahead_does_not_install(tmp_path):
    s = load_desk_strategy()
    trust = int(s.objectives["short"].look_ahead["min_n_for_trust"])
    n = max(3, trust - 5)
    eps = [
        _ep(i, label="WIN", alpha=0.40, realised=0.45, dd=-0.01)
        for i in range(n)
    ]
    journal = tmp_path / "journal.json"
    out = step_self_learn(eps, journal_file=journal, strategy=s)
    assert out["score"]["verdict"] == "LOOKAHEAD_SUSPECT"
    assert out["action"] in ("discard_lookahead", "reject_lookahead")
    data = load_journal(journal)
    assert data.get("experiment") in (None, {})
