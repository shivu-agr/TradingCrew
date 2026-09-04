"""Data-driven crowd-weight learner (crowd_fit)."""

from __future__ import annotations

import random

from trading_crew.agentic.memory.episodic import Episode, OutcomeStatus, Regime
from trading_crew.agentic.swarm.crowd import CrowdConfig
from trading_crew.agentic.swarm import crowd_fit


def _episode(idx: int, components, alpha, *, resolved=True) -> Episode:
    return Episode(
        episode_id=f"AAA-{idx}",
        symbol="AAA",
        decision_ts="2024-01-01T00:00:00+00:00",
        state_summary="s",
        regime=Regime.UNKNOWN,
        action_proposal={"side": "BUY", "target_weight": 0.1, "horizon_days": 30},
        outcome_ts="2024-02-01T00:00:00+00:00",
        outcome_status=OutcomeStatus.RESOLVED if resolved else OutcomeStatus.PENDING,
        alpha_return=alpha,
        crowd_components=components,
    )


def test_episode_roundtrip_preserves_crowd_components():
    ep = _episode(1, {"put_call": 0.3, "analyst": -0.2}, 0.05)
    restored = Episode.from_dict(ep.to_dict())
    assert restored.crowd_components == {"put_call": 0.3, "analyst": -0.2}


def test_fit_returns_none_below_min_samples():
    eps = [_episode(i, {"put_call": 0.1, "analyst": 0.1}, 0.01) for i in range(5)]
    assert crowd_fit.fit_crowd_weights(eps, min_samples=30) is None


def test_fit_ignores_pending_and_unlabelled():
    eps = [_episode(i, {"put_call": 0.1}, None, resolved=False) for i in range(50)]
    assert crowd_fit.fit_crowd_weights(eps, min_samples=10) is None


def test_predictive_component_gets_more_weight():
    # analyst strongly (positively) predicts alpha; put_call is pure noise.
    rng = random.Random(0)
    eps = []
    for i in range(200):
        analyst = rng.uniform(-1, 1)
        noise = rng.uniform(-1, 1)
        alpha = 0.5 * analyst + 0.01 * rng.gauss(0, 1)
        eps.append(_episode(i, {"analyst": analyst, "put_call": noise}, alpha))

    result = crowd_fit.fit_crowd_weights(eps, min_samples=30, min_component_obs=10)
    assert result is not None
    assert result["n_samples"] == 200
    w = result["weights"]
    # The predictive component should carry more fusion weight than the noise one.
    assert w["w_analyst"] > w["w_put_call"]
    # analyst is momentum-positive here → positive coefficient.
    assert result["coefficients"]["analyst"] > 0


def test_load_crowd_config_roundtrip(tmp_path):
    eps = []
    rng = random.Random(1)
    for i in range(120):
        analyst = rng.uniform(-1, 1)
        eps.append(_episode(i, {"analyst": analyst, "put_call": rng.uniform(-1, 1)}, 0.4 * analyst))
    result = crowd_fit.fit_crowd_weights(eps, min_samples=30)
    assert result is not None

    crowd_fit.save_weights(result, cache_dir=str(tmp_path))
    cfg = crowd_fit.load_crowd_config(cache_dir=str(tmp_path))
    assert isinstance(cfg, CrowdConfig)
    # Learned weights should differ from the flat 1.0 default.
    assert (cfg.w_analyst, cfg.w_put_call) != (1.0, 1.0)
    # Non-weight parameters are preserved from the base config.
    assert cfg.crowd_threshold == CrowdConfig().crowd_threshold


def test_load_crowd_config_default_when_no_file(tmp_path):
    cfg = crowd_fit.load_crowd_config(cache_dir=str(tmp_path))
    assert cfg == CrowdConfig()
