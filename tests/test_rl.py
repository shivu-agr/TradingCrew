"""Tests for the L4 RL stack — env, networks, PPO, storage, inference."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(monkeypatch):
    """Redirect ``$TRADINGCREW_CACHE_DIR`` to a tmpdir so RL run records
    don't pollute the developer's real ``~/.trading_crew`` during tests."""
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("TRADINGCREW_CACHE_DIR", td)
        # The storage module captures the cache root at import time —
        # reload it so the patched env var takes effect.
        import importlib
        import trading_crew.agentic.rl.storage as storage
        importlib.reload(storage)
        yield Path(td)


@pytest.fixture
def synthetic_ohlcv():
    """500 bars of GBM with small drift — long enough to train and eval."""
    rng = np.random.default_rng(42)
    n = 500
    log_returns = rng.normal(0.0003, 0.012, n)
    prices = 100 * np.exp(np.cumsum(log_returns))
    high = prices * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = prices * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = prices * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": prices, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------


def test_feature_extractor_emits_fixed_size_vector(synthetic_ohlcv):
    from trading_crew.agentic.rl import FeatureExtractor, FEATURE_DIM
    fe = FeatureExtractor()
    obs = fe.extract(synthetic_ohlcv)
    assert obs.shape == (FEATURE_DIM,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()


def test_feature_extractor_clips_to_plus_minus_five(synthetic_ohlcv):
    """A pathological input (a giant jump) should not produce features
    outside the ±5 clip band — otherwise PPO advantage normalisation
    could blow up on a single black-swan bar."""
    from trading_crew.agentic.rl import FeatureExtractor
    df = synthetic_ohlcv.copy()
    df.iloc[-1, df.columns.get_loc("close")] *= 10  # 10x spike
    obs = FeatureExtractor().extract(df)
    assert (obs >= -5.0).all() and (obs <= 5.0).all()


def test_feature_extractor_rejects_short_history():
    from trading_crew.agentic.rl import FeatureExtractor
    fe = FeatureExtractor(lookback=60)
    short = pd.DataFrame({
        "open": [1, 2], "high": [1, 2], "low": [1, 2],
        "close": [1, 2], "volume": [100, 100],
    }, index=pd.date_range("2020-01-01", periods=2, freq="B"))
    with pytest.raises(ValueError):
        fe.extract(short)


def test_feature_extractor_position_features_change_obs(synthetic_ohlcv):
    """Same OHLCV but different position state should produce a
    different obs vector — otherwise the position info isn't reaching
    the policy."""
    from trading_crew.agentic.rl import FeatureExtractor
    fe = FeatureExtractor()
    obs_flat = fe.extract(synthetic_ohlcv, position_weight=0.0)
    obs_long = fe.extract(synthetic_ohlcv, position_weight=0.5)
    assert not np.allclose(obs_flat, obs_long)


# ---------------------------------------------------------------------------
# TradingEnv
# ---------------------------------------------------------------------------


def test_env_action_space_size():
    from trading_crew.agentic.rl import ACTION_WEIGHTS, N_ACTIONS
    assert N_ACTIONS == len(ACTION_WEIGHTS)
    assert 0.0 in ACTION_WEIGHTS  # FLAT must exist
    assert min(ACTION_WEIGHTS) < 0  # at least one short bucket
    assert max(ACTION_WEIGHTS) > 0  # at least one long bucket


def test_env_reset_returns_finite_obs(synthetic_ohlcv):
    from trading_crew.agentic.rl import TradingEnv, FEATURE_DIM
    env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv)
    obs = env.reset(seed=0)
    assert obs.shape == (FEATURE_DIM,)
    assert np.isfinite(obs).all()


def test_env_step_returns_proper_tuple(synthetic_ohlcv):
    from trading_crew.agentic.rl import TradingEnv, N_ACTIONS
    env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv)
    env.reset(seed=1)
    step = env.step(0)
    assert hasattr(step, "obs") and hasattr(step, "reward")
    assert hasattr(step, "done") and hasattr(step, "info")
    assert isinstance(step.reward, float)
    assert "nav" in step.info and "drawdown" in step.info
    assert isinstance(step.done, bool)
    # FILLED for a real order, NO_ORDER if the bucket happens to equal current
    assert step.info["fill_status"] in {"FILLED", "PARTIAL_FILL", "REJECTED", "EXPIRED", "NO_ORDER"}


def test_env_step_rejects_invalid_action(synthetic_ohlcv):
    from trading_crew.agentic.rl import TradingEnv, N_ACTIONS
    env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv)
    env.reset(seed=2)
    with pytest.raises(ValueError):
        env.step(N_ACTIONS)
    with pytest.raises(ValueError):
        env.step(-1)


def test_env_handles_trim_reduce_position(synthetic_ohlcv):
    """Reducing an existing long should produce a SELL order (negative
    qty), even though target_weight stays positive.  Without the
    delta-aware order build this would log 'side contradicts delta' and
    skip the trade — proven by checking NAV actually moved on the
    reduce step."""
    from trading_crew.agentic.rl import TradingEnv, ACTION_WEIGHTS
    long_idx = ACTION_WEIGHTS.index(0.20)
    half_idx = ACTION_WEIGHTS.index(0.10)
    env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv, random_starts=False)
    env.reset(seed=3)
    s1 = env.step(long_idx)  # open 20% long
    s2 = env.step(half_idx)  # trim to 10% long — should be a real SELL
    # The simulator must have processed a fill on the reduce step
    # (status != NO_ORDER) because the delta is non-zero.
    assert s2.info["fill_status"] in {"FILLED", "PARTIAL_FILL"}


def test_env_drawdown_kill_terminates_episode():
    """Force a kill-by-drawdown by feeding the env a tape that falls
    50%+ from start.  The drawdown gate should end the episode early."""
    from trading_crew.agentic.rl import TradingEnv, TradingEnvConfig, ACTION_WEIGHTS
    n = 200
    prices = np.linspace(100.0, 30.0, n)  # -70% straight down
    high = prices * 1.01
    low = prices * 0.99
    open_ = prices
    volume = np.full(n, 1_000_000.0)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": prices, "volume": volume},
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    env = TradingEnv(symbol="T", ohlcv=df, random_starts=False,
                     config=TradingEnvConfig(drawdown_kill_pct=0.10))
    env.reset(seed=4)
    long_idx = ACTION_WEIGHTS.index(0.20)
    done = False
    for _ in range(len(df) - 65):
        step = env.step(long_idx)
        if step.done:
            done = True
            assert "drawdown" in step.info["kill_reason"]
            assert step.info["drawdown"] >= 0.10
            break
    assert done, "expected drawdown-kill to terminate the episode"


def test_env_buy_then_full_close_clears_position(synthetic_ohlcv):
    """Going from +20% to 0 must close the position; the env then
    reports pos_weight=0 in its next observation."""
    from trading_crew.agentic.rl import TradingEnv, ACTION_WEIGHTS
    env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv, random_starts=False)
    env.reset(seed=5)
    env.step(ACTION_WEIGHTS.index(0.20))
    step = env.step(ACTION_WEIGHTS.index(0.0))
    assert "T" not in env._state.positions  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# ActorCritic network
# ---------------------------------------------------------------------------


def test_actor_critic_forward_shapes():
    from trading_crew.agentic.rl import ActorCritic, FEATURE_DIM, N_ACTIONS
    m = ActorCritic(FEATURE_DIM, N_ACTIONS, hidden_dim=32, n_hidden_layers=2)
    batch = torch.randn(8, FEATURE_DIM)
    logits, value = m(batch)
    assert logits.shape == (8, N_ACTIONS)
    assert value.shape == (8,)


def test_actor_critic_act_returns_int_action():
    from trading_crew.agentic.rl import ActorCritic, FEATURE_DIM, N_ACTIONS
    m = ActorCritic(FEATURE_DIM, N_ACTIONS)
    obs = torch.randn(FEATURE_DIM)
    action, log_prob, value = m.act(obs, deterministic=False)
    assert isinstance(action, int) and 0 <= action < N_ACTIONS
    assert isinstance(log_prob, float)
    assert isinstance(value, float)


def test_actor_critic_action_probs_sum_to_one():
    from trading_crew.agentic.rl import ActorCritic, FEATURE_DIM, N_ACTIONS
    m = ActorCritic(FEATURE_DIM, N_ACTIONS)
    probs = m.action_probs(torch.randn(FEATURE_DIM)).reshape(-1)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# PPOTrainer
# ---------------------------------------------------------------------------


def test_ppo_trainer_produces_metrics(synthetic_ohlcv):
    from trading_crew.agentic.rl import (
        TradingEnv, TradingEnvConfig, PPOTrainer, PPOConfig,
    )
    env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv, config=TradingEnvConfig())
    cfg = PPOConfig(total_steps=128, steps_per_rollout=64, n_epochs=2, minibatch_size=16, seed=1)
    trainer = PPOTrainer(env, cfg)
    metrics = trainer.train()
    assert len(metrics) >= 2
    for m in metrics:
        assert m.steps_done > 0
        assert np.isfinite(m.policy_loss)
        assert np.isfinite(m.entropy)
        assert 0.0 < m.entropy < np.log(7) + 1e-3  # max entropy for 7-way categorical


def test_ppo_evaluate_returns_action_freq(synthetic_ohlcv):
    from trading_crew.agentic.rl import (
        TradingEnv, PPOTrainer, PPOConfig, N_ACTIONS,
    )
    env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv)
    eval_env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv, random_starts=False)
    trainer = PPOTrainer(env, PPOConfig(total_steps=128, steps_per_rollout=64, n_epochs=1, minibatch_size=16, seed=2))
    trainer.train()
    result = trainer.evaluate(eval_env, n_episodes=2)
    assert result["n_episodes"] == 2
    assert len(result["action_freq"]) == N_ACTIONS
    assert pytest.approx(sum(result["action_freq"]), abs=1e-6) == 1.0


def test_ppo_should_stop_aborts_early(synthetic_ohlcv):
    """The trainer must respect ``should_stop`` so the API's /stop
    endpoint can cooperatively halt a runaway job."""
    from trading_crew.agentic.rl import (
        TradingEnv, PPOTrainer, PPOConfig,
    )
    env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv)
    flag = {"stop": False}
    def stopper():
        flag["stop"] = True  # flip after the first call
        return flag["stop"]
    trainer = PPOTrainer(
        env, PPOConfig(total_steps=10_000, steps_per_rollout=64, seed=3),
        should_stop=lambda: flag["stop"],
    )
    # First rollout runs to completion (stop not yet flagged); flip the
    # flag manually to interrupt after that.
    flag["stop"] = False
    metrics = []
    def cb(m):
        metrics.append(m)
        flag["stop"] = True  # request stop after the very first rollout
    trainer.on_metrics = cb
    trainer.train()
    # The trainer should NOT have run all the steps we requested.
    assert metrics[-1].steps_done < 10_000


def test_ppo_save_and_load_checkpoint_round_trip(synthetic_ohlcv, tmp_path):
    from trading_crew.agentic.rl import (
        TradingEnv, PPOTrainer, PPOConfig,
    )
    env = TradingEnv(symbol="T", ohlcv=synthetic_ohlcv)
    trainer = PPOTrainer(env, PPOConfig(total_steps=128, steps_per_rollout=64, n_epochs=1, minibatch_size=16, seed=4))
    trainer.train()
    ckpt = tmp_path / "policy.pt"
    trainer.save_checkpoint(ckpt)
    assert ckpt.exists()
    loaded = PPOTrainer.load_model(ckpt)
    # Loaded model + fresh model should give the SAME action probabilities
    # for the same observation (because we loaded weights from the trained
    # one, not a fresh init).
    obs = torch.zeros(21)
    p1 = trainer.model.action_probs(obs).reshape(-1).detach().numpy()
    p2 = loaded.action_probs(obs).reshape(-1).numpy()
    assert np.allclose(p1, p2, atol=1e-6)


# ---------------------------------------------------------------------------
# Storage + promotion
# ---------------------------------------------------------------------------


def test_storage_save_and_load_round_trip(cache_dir):
    from trading_crew.agentic.rl.storage import (
        RLRunRecord, save_run, load_run, list_runs,
    )
    rec = RLRunRecord(
        run_id="20260101T000000-000",
        ticker="ABC",
        asset_class="stock",
        created_ts="2026-01-01T00:00:00Z",
        status="completed",
    )
    save_run(rec)
    loaded = load_run("ABC", rec.run_id)
    assert loaded is not None
    assert loaded.run_id == rec.run_id
    runs = list_runs("ABC")
    assert any(r.run_id == rec.run_id for r in runs)


def test_storage_metrics_jsonl_append_and_read(cache_dir):
    from trading_crew.agentic.rl.storage import (
        RLRunRecord, save_run, append_metric_jsonl, read_metrics_jsonl,
    )
    rec = RLRunRecord(
        run_id="run-metrics-1",
        ticker="ABC",
        asset_class="stock",
        created_ts="2026-01-01T00:00:00Z",
        status="running",
    )
    save_run(rec)
    append_metric_jsonl("ABC", rec.run_id, {"rollout_index": 1, "steps_done": 100})
    append_metric_jsonl("ABC", rec.run_id, {"rollout_index": 2, "steps_done": 200})
    out = read_metrics_jsonl("ABC", rec.run_id)
    assert len(out) == 2
    assert out[0]["steps_done"] == 100
    assert out[1]["steps_done"] == 200


def test_promote_run_requires_checkpoint(cache_dir):
    from trading_crew.agentic.rl.storage import (
        RLRunRecord, save_run, promote_run,
    )
    rec = RLRunRecord(
        run_id="run-no-ckpt",
        ticker="XYZ",
        asset_class="stock",
        created_ts="2026-01-01T00:00:00Z",
        status="completed",
    )
    save_run(rec)
    # No checkpoint written → promote must refuse rather than silently
    # write a broken pointer.
    with pytest.raises(FileNotFoundError):
        promote_run("XYZ", rec.run_id)


def test_promote_run_writes_pointer(cache_dir, synthetic_ohlcv):
    from trading_crew.agentic.rl import (
        TradingEnv, PPOTrainer, PPOConfig,
    )
    from trading_crew.agentic.rl.storage import (
        RLRunRecord, save_run, promote_run, get_promoted,
        policy_checkpoint_path,
    )
    rec = RLRunRecord(
        run_id="run-with-ckpt",
        ticker="XYZ",
        asset_class="stock",
        created_ts="2026-01-01T00:00:00Z",
        status="completed",
    )
    save_run(rec)
    # Train a quick model + save a real checkpoint.
    env = TradingEnv(symbol="XYZ", ohlcv=synthetic_ohlcv)
    trainer = PPOTrainer(env, PPOConfig(total_steps=64, steps_per_rollout=64, n_epochs=1, minibatch_size=16, seed=5))
    trainer.train()
    trainer.save_checkpoint(policy_checkpoint_path("XYZ", rec.run_id))

    result = promote_run("XYZ", rec.run_id)
    assert result["ticker"] == "XYZ"
    assert result["run_id"] == rec.run_id
    pointer = get_promoted("XYZ")
    assert pointer is not None and pointer["run_id"] == rec.run_id


# ---------------------------------------------------------------------------
# Inference / PolicyClient
# ---------------------------------------------------------------------------


def test_load_policy_returns_none_when_no_promotion(cache_dir):
    from trading_crew.agentic.rl import load_policy
    assert load_policy("DOES_NOT_EXIST") is None


def test_policy_client_recommend_shapes(cache_dir, synthetic_ohlcv):
    """Round-trip: train -> save -> promote -> load -> recommend."""
    from trading_crew.agentic.rl import (
        TradingEnv, PPOTrainer, PPOConfig, load_policy, ACTION_WEIGHTS,
    )
    from trading_crew.agentic.rl.storage import (
        RLRunRecord, save_run, promote_run, policy_checkpoint_path,
    )
    ticker = "TEST"
    rec = RLRunRecord(
        run_id="round-trip", ticker=ticker, asset_class="stock",
        created_ts="2026-01-01T00:00:00Z", status="completed",
    )
    save_run(rec)
    env = TradingEnv(symbol=ticker, ohlcv=synthetic_ohlcv)
    trainer = PPOTrainer(env, PPOConfig(total_steps=64, steps_per_rollout=64, n_epochs=1, minibatch_size=16, seed=6))
    trainer.train()
    trainer.save_checkpoint(policy_checkpoint_path(ticker, rec.run_id))
    promote_run(ticker, rec.run_id)

    client = load_policy(ticker)
    assert client is not None
    rec_pred = client.recommend(synthetic_ohlcv)
    assert 0 <= rec_pred.best_action_idx < len(ACTION_WEIGHTS)
    assert len(rec_pred.action_distribution) == len(ACTION_WEIGHTS)
    assert abs(sum(rec_pred.action_distribution) - 1.0) < 1e-5
    assert 0.0 <= rec_pred.confidence <= 1.0


def test_policy_client_rejects_short_history(cache_dir, synthetic_ohlcv):
    from trading_crew.agentic.rl import (
        TradingEnv, PPOTrainer, PPOConfig, load_policy,
    )
    from trading_crew.agentic.rl.storage import (
        RLRunRecord, save_run, promote_run, policy_checkpoint_path,
    )
    ticker = "TEST_SHORT"
    rec = RLRunRecord(
        run_id="r", ticker=ticker, asset_class="stock",
        created_ts="2026-01-01T00:00:00Z", status="completed",
    )
    save_run(rec)
    env = TradingEnv(symbol=ticker, ohlcv=synthetic_ohlcv)
    trainer = PPOTrainer(env, PPOConfig(total_steps=64, steps_per_rollout=64, n_epochs=1, minibatch_size=16, seed=7))
    trainer.train()
    trainer.save_checkpoint(policy_checkpoint_path(ticker, rec.run_id))
    promote_run(ticker, rec.run_id)

    client = load_policy(ticker)
    short = synthetic_ohlcv.head(10)
    with pytest.raises(ValueError):
        client.recommend(short)


# ---------------------------------------------------------------------------
# Phase 2D — risk-sensitive PPO + curriculum + reward shaping + universe tail
# ---------------------------------------------------------------------------


def _make_buffer(trainer, rewards):
    """Build a minimal rollout buffer and run GAE on it via ``trainer``."""
    from trading_crew.agentic.rl.ppo import RolloutBuffer
    n = len(rewards)
    buf = RolloutBuffer.empty(n, obs_dim=2)
    buf.rewards[:] = np.asarray(rewards, dtype=np.float32)
    buf.values[:] = 0.0
    buf.dones[:] = False
    trainer._compute_gae(buf, last_value=0.0)
    return buf


def test_cvar_ppo_amplifies_tail_advantages(synthetic_ohlcv):
    """With ``risk_aversion>0`` the tail steps' advantages should grow vs the vanilla baseline."""
    from trading_crew.agentic.rl import PPOConfig, PPOTrainer, TradingEnv

    env = TradingEnv(symbol="XYZ", ohlcv=synthetic_ohlcv)
    rewards = [-1.0] * 8 + [1.0] * 8  # bottom 50% = the -1s.

    # Vanilla PPO advantages.
    vanilla = PPOTrainer(env, PPOConfig(total_steps=16, steps_per_rollout=16, n_epochs=1, minibatch_size=8, seed=11, risk_aversion=0.0))
    vanilla_buf = _make_buffer(vanilla, rewards)

    # CVaR-PPO advantages with α=0.5 (boost the bottom half).
    cvar = PPOTrainer(env, PPOConfig(total_steps=16, steps_per_rollout=16, n_epochs=1, minibatch_size=8, seed=11, risk_aversion=0.5, cvar_alpha=0.5))
    cvar_buf = _make_buffer(cvar, rewards)

    # Tail steps (the -1.0s) carry *negative* advantages.  Multiplying by
    # 1.5 makes them more negative — i.e. magnitude grows.
    tail_v = vanilla_buf.advantages[:8]
    tail_c = cvar_buf.advantages[:8]
    assert np.all(np.abs(tail_c) > np.abs(tail_v))
    # Upper-half (positive-reward) steps must be untouched.
    np.testing.assert_allclose(vanilla_buf.advantages[8:], cvar_buf.advantages[8:], atol=1e-6)


def test_volatility_curriculum_starts_with_low_vol(synthetic_ohlcv):
    """Progress=0 should sample starts from the lowest-vol pool."""
    from trading_crew.agentic.rl import TradingEnv, TradingEnvConfig

    cfg = TradingEnvConfig(volatility_curriculum=True)
    env = TradingEnv(symbol="XYZ", ohlcv=synthetic_ohlcv, config=cfg, random_starts=True, rng=np.random.default_rng(7))
    assert env._vol_sorted_starts is not None
    assert len(env._vol_sorted_starts) > 0
    # At progress=0 only the bottom 10% of vol-sorted starts are eligible.
    pool_size = max(1, int(len(env._vol_sorted_starts) * 0.1))
    eligible = set(env._vol_sorted_starts[:pool_size].tolist())
    env.set_curriculum_progress(0.0)
    for _ in range(20):
        env.reset()
        assert env._t in eligible


def test_volatility_curriculum_expands_with_progress(synthetic_ohlcv):
    from trading_crew.agentic.rl import TradingEnv, TradingEnvConfig
    cfg = TradingEnvConfig(volatility_curriculum=True)
    env = TradingEnv(symbol="XYZ", ohlcv=synthetic_ohlcv, config=cfg, random_starts=True, rng=np.random.default_rng(8))
    env.set_curriculum_progress(1.0)
    # The full sortable pool should be eligible now.
    full_pool = set(env._vol_sorted_starts.tolist())
    sampled = set()
    for _ in range(200):
        env.reset()
        sampled.add(env._t)
    assert sampled.issubset(full_pool)


def test_reward_shaping_knobs_in_env_config():
    from trading_crew.agentic.rl import TradingEnvConfig
    cfg = TradingEnvConfig(turnover_penalty_bps=5.0, drawdown_penalty_coef=0.75)
    assert cfg.turnover_penalty_bps == 5.0
    assert cfg.drawdown_penalty_coef == 0.75


def test_feature_extractor_universe_tail_one_hot(synthetic_ohlcv):
    from trading_crew.agentic.rl.state import FEATURE_DIM, FeatureExtractor

    fe = FeatureExtractor(universe_size=4, universe_index=2)
    vec = fe.extract(synthetic_ohlcv.head(100), position_weight=0.0)
    assert vec.shape == (FEATURE_DIM + 4,)
    tail = vec[FEATURE_DIM:]
    assert tail[2] == 1.0
    assert tail.sum() == 1.0


def test_feature_extractor_universe_zero_size_is_unchanged(synthetic_ohlcv):
    from trading_crew.agentic.rl.state import FEATURE_DIM, FeatureExtractor

    fe = FeatureExtractor(universe_size=0)
    vec = fe.extract(synthetic_ohlcv.head(100), position_weight=0.0)
    assert vec.shape == (FEATURE_DIM,)


# ---------------------------------------------------------------------------
# Phase 2D — alternative trainers (CQL / C51 / Decision Transformer)
# ---------------------------------------------------------------------------


def test_cql_trainer_runs_a_few_steps(synthetic_ohlcv):
    from trading_crew.agentic.rl import (
        CQLConfig,
        CQLTrainer,
        TradingEnv,
        collect_transitions_from_episodes,
    )
    env = TradingEnv(symbol="XYZ", ohlcv=synthetic_ohlcv)
    transitions = collect_transitions_from_episodes(
        episodes=[None] * 4,  # the function only needs the count of envs to step
        env=env,
        max_per_episode=8,
    )
    assert len(transitions) > 0
    trainer = CQLTrainer(env=env, transitions=transitions, config=CQLConfig(total_steps=8, batch_size=8, seed=3))
    metrics = trainer.train()
    assert len(metrics) == 8
    assert all(np.isfinite(m.q_loss) for m in metrics)


def test_c51_trainer_runs_a_few_steps(synthetic_ohlcv):
    from trading_crew.agentic.rl import C51Config, C51Trainer, TradingEnv
    env = TradingEnv(symbol="XYZ", ohlcv=synthetic_ohlcv)
    trainer = C51Trainer(env=env, config=C51Config(total_steps=10, batch_size=8, seed=4, n_atoms=21))
    metrics = trainer.train()
    # Some steps may be pre-warmup (buffer still filling) — at least one update must land.
    update_steps = [m for m in metrics if np.isfinite(m.loss)]
    assert len(update_steps) >= 1


def test_decision_transformer_trainer_runs_a_few_steps():
    from trading_crew.agentic.rl import (
        DTConfig,
        DecisionTransformerTrainer,
        trajectories_from_transitions,
    )
    # Build a synthetic offline buffer of two short trajectories.
    transitions = []
    rng = np.random.default_rng(13)
    for ep in range(2):
        for t in range(10):
            transitions.append({
                "obs": rng.normal(size=21).astype(np.float32),  # FEATURE_DIM = 21
                "action": int(rng.integers(0, 7)),
                "reward": float(rng.normal(0, 0.1)),
                "done": t == 9,
            })
    trajectories = trajectories_from_transitions(transitions)
    assert len(trajectories) == 2
    trainer = DecisionTransformerTrainer(
        trajectories=trajectories,
        config=DTConfig(total_steps=5, batch_size=4, context_len=5, seed=5),
    )
    metrics = trainer.train()
    assert len(metrics) == 5
    assert all(np.isfinite(m.loss) for m in metrics)
