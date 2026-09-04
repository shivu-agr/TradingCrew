"""Book-level success / failure / look-ahead scoring against desk_strategy.yaml."""

from __future__ import annotations

from trading_crew.agentic.memory.episodic import Episode, OutcomeStatus, Regime
from trading_crew.agentic.objectives import BookVerdict, score_episodes
from trading_crew.agentic.strategy import load_desk_strategy


def _ep(i: int, *, label: str, alpha: float, realised: float, dd: float) -> Episode:
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


def test_insufficient_below_min_n():
    s = load_desk_strategy()
    eps = [_ep(1, label="WIN", alpha=0.02, realised=0.03, dd=-0.02)]
    score = score_episodes(eps, tier="short", strategy=s)
    assert score.verdict == BookVerdict.INSUFFICIENT
    assert score.metrics.n == 1


def test_success_clears_all_short_bars():
    s = load_desk_strategy()
    n = s.objectives["short"].min_n
    wins = max(int(n * 0.7), n - 2)
    eps = []
    for i in range(n):
        if i < wins:
            eps.append(_ep(i, label="WIN", alpha=0.02, realised=0.03, dd=-0.03))
        else:
            eps.append(_ep(i, label="LOSS", alpha=-0.005, realised=-0.01, dd=-0.04))
    score = score_episodes(eps, tier="short", strategy=s)
    assert score.verdict == BookVerdict.SUCCESS


def test_failure_on_hit_rate():
    s = load_desk_strategy()
    n = s.objectives["short"].min_n
    eps = [_ep(i, label="LOSS", alpha=-0.03, realised=-0.04, dd=-0.05) for i in range(n)]
    score = score_episodes(eps, tier="short", strategy=s)
    assert score.verdict == BookVerdict.FAILURE


def test_lookahead_suspect_on_tiny_perfect_book():
    s = load_desk_strategy()
    # n below min_n_for_trust with absurd hit-rate + alpha.
    trust = int(s.objectives["short"].look_ahead["min_n_for_trust"])
    n = max(3, trust - 5)
    eps = [_ep(i, label="WIN", alpha=0.40, realised=0.45, dd=-0.01) for i in range(n)]
    score = score_episodes(eps, tier="short", strategy=s)
    assert score.verdict == BookVerdict.LOOKAHEAD_SUSPECT
