"""Background L4 RL training runner.

The HTTP layer in :mod:`web.backend.app` hands a config to
``start_training(...)`` and returns the run id immediately.  Training
runs in a dedicated worker thread (PPO is CPU-bound and PyTorch
releases the GIL during tensor ops, so a thread is the right primitive
here — we don't need a full process).  Status + metrics are exposed
via simple in-memory registries and the on-disk JSONL log so the UI
can poll without holding any lock.

The runner is **single-tenant** by ticker: starting a new run for the
same ticker cooperatively asks the previous one to stop.  This keeps
the disk-write contention bounded and gives the user a sane mental
model ("one policy training per ticker at a time").
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from trading_crew.agentic.rl import (
    PPOConfig,
    PPOTrainer,
    RLRunRecord,
    TradingEnv,
    TradingEnvConfig,
    TrainingMetrics,
)
from trading_crew.agentic.rl.storage import (
    append_metric_jsonl,
    new_run_id,
    policy_checkpoint_path,
    read_metrics_jsonl,
    save_run,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory registry
# ---------------------------------------------------------------------------


@dataclass
class _RunHandle:
    """Per-run mutable state held in memory while training is live."""

    record: RLRunRecord
    thread: threading.Thread
    stop_event: threading.Event
    started_ts: float
    last_metric_ts: float = 0.0
    # Cached config payload — the API echoes it back so the UI can show
    # what was launched without re-loading the record from disk.
    request: Dict[str, Any] = field(default_factory=dict)


_LOCK = threading.RLock()
_ACTIVE: Dict[str, _RunHandle] = {}  # ticker -> handle (one at a time)
_ALL_RUNS: Dict[str, _RunHandle] = {}  # run_id -> handle (history)


def _ticker_key(ticker: str) -> str:
    return ticker.upper().strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def start_training(
    *,
    ticker: str,
    asset_class: str,
    train_window_days: int,
    eval_window_days: int,
    ppo_overrides: Optional[Dict[str, Any]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    algorithm: str = "ppo",
    policy_universe: Optional[list] = None,
    horizon_mode: str = "balanced",
) -> Dict[str, Any]:
    """Kick off a background training run.

    Returns the new ``RLRunRecord`` (as dict) immediately — the actual
    training happens in a worker thread.

    Algorithm:

    1. Fetch a 3-year buffered OHLCV window so even the shortest train
       window has enough warm-up bars.
    2. Slice into ``train`` and ``eval`` segments (eval is the most
       recent ``eval_window_days`` bars; train is what remains).
    3. Build a ``TradingEnv`` for each.
    4. Construct a ``PPOTrainer`` with metric callback that appends a
       JSONL line.
    5. Launch the worker.
    """
    ticker = _ticker_key(ticker)
    if not ticker:
        raise ValueError("ticker is required")

    with _LOCK:
        # If a run for this ticker is in progress, ask it to stop and
        # wait briefly so we can spawn the new one without contention.
        existing = _ACTIVE.get(ticker)
        if existing is not None and existing.thread.is_alive():
            logger.info("Stopping previous RL run for %s (%s)", ticker, existing.record.run_id)
            existing.stop_event.set()
            existing.thread.join(timeout=5.0)

    # Pull a buffered window so RSI/MACD warm-ups + eval split fit.
    from .charts import _fetch_ohlcv  # local import — avoids circular
    end_dt = datetime.utcnow()
    buffer = max(train_window_days + eval_window_days + 120, 600)
    df = _fetch_ohlcv(ticker, end_dt, buffer)
    if df is None or df.empty:
        raise ValueError(f"No OHLCV history available for {ticker}")

    df = df.rename(columns={c: c.lower() for c in df.columns}).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")

    if len(df) < (train_window_days + eval_window_days):
        raise ValueError(
            f"Only {len(df)} bars available for {ticker}, "
            f"need at least {train_window_days + eval_window_days}"
        )

    eval_df = df.tail(eval_window_days).set_index("date")
    train_df = df.iloc[-(train_window_days + eval_window_days):-eval_window_days].set_index("date")

    env_cfg_kwargs = {"starting_cash": 100_000.0, "horizon_days": 1}
    env_cfg_kwargs.update(env_overrides or {})
    env_cfg = TradingEnvConfig(**env_cfg_kwargs)

    train_env = TradingEnv(symbol=ticker, ohlcv=train_df, config=env_cfg, random_starts=True, rng=np.random.default_rng(seed))
    eval_env = TradingEnv(symbol=ticker, ohlcv=eval_df, config=env_cfg, random_starts=False, rng=np.random.default_rng(seed + 1))

    ppo_cfg_kwargs: Dict[str, Any] = {
        "total_steps": 20_000,
        "steps_per_rollout": 512,
        "n_epochs": 4,
        "minibatch_size": 64,
        "learning_rate": 3e-4,
        "seed": seed,
    }
    ppo_cfg_kwargs.update(ppo_overrides or {})
    # PPOConfig doesn't carry an explicit algorithm field — its
    # ``risk_aversion > 0`` already gives us CVaR-PPO.  We still
    # accept the dropdown choice from the UI so the run record
    # reflects which algorithm the user picked; alternative
    # algorithms (CQL/C51/Decision-Transformer) are dispatched in
    # the worker below.
    algorithm_choice = (algorithm or "ppo").lower()
    if algorithm_choice == "cvar_ppo" and ppo_cfg_kwargs.get("risk_aversion", 0.0) == 0.0:
        ppo_cfg_kwargs["risk_aversion"] = 0.25  # sensible default
    ppo_cfg = PPOConfig(**ppo_cfg_kwargs)

    run_id = new_run_id()
    record = RLRunRecord(
        run_id=run_id,
        ticker=ticker,
        asset_class=asset_class,
        created_ts=datetime.now(timezone.utc).isoformat(),
        status="running",
        env_config={k: getattr(env_cfg, k) for k in env_cfg.__dataclass_fields__.keys()},
        ppo_config={k: getattr(ppo_cfg, k) for k in ppo_cfg.__dataclass_fields__.keys()},
        data_window={
            "train_start": str(train_df.index.min()),
            "train_end": str(train_df.index.max()),
            "eval_start": str(eval_df.index.min()),
            "eval_end": str(eval_df.index.max()),
            "bars_train": len(train_df),
            "bars_eval": len(eval_df),
            "horizon_mode": horizon_mode,
        },
        baseline_buy_and_hold=_buy_and_hold(eval_df),
    )
    save_run(record)

    stop_event = threading.Event()
    request = {
        "ticker": ticker,
        "asset_class": asset_class,
        "train_window_days": train_window_days,
        "eval_window_days": eval_window_days,
        "horizon_mode": horizon_mode,
        "ppo_config": record.ppo_config,
        "env_config": record.env_config,
        "seed": seed,
        "algorithm": algorithm_choice,
        "policy_universe": list(policy_universe or []),
    }

    thread = threading.Thread(
        target=_worker,
        name=f"rl-train-{ticker}-{run_id}",
        args=(record, train_env, eval_env, ppo_cfg, stop_event, algorithm_choice),
        daemon=True,
    )
    handle = _RunHandle(record=record, thread=thread, stop_event=stop_event, started_ts=time.time(), request=request)
    with _LOCK:
        _ACTIVE[ticker] = handle
        _ALL_RUNS[run_id] = handle
    thread.start()
    return record.to_dict()


def stop_training(ticker: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
    """Cooperatively stop the active run for ``ticker`` (or a specific run)."""
    with _LOCK:
        handle = None
        if run_id is not None:
            handle = _ALL_RUNS.get(run_id)
        elif ticker is not None:
            handle = _ACTIVE.get(_ticker_key(ticker))
        if handle is None:
            return {"stopped": False, "reason": "no active run found"}
        handle.stop_event.set()
    handle.thread.join(timeout=10.0)
    return {"stopped": True, "ticker": handle.record.ticker, "run_id": handle.record.run_id}


def get_status(ticker: str) -> Dict[str, Any]:
    """Snapshot of the active run for ``ticker`` (or empty if none)."""
    ticker = _ticker_key(ticker)
    with _LOCK:
        handle = _ACTIVE.get(ticker)
        if handle is None:
            return {"ticker": ticker, "active": False}
        rec = handle.record
    # Read metrics from disk every call — keeps the API stateless and
    # cheap even when 4 polling clients connect simultaneously.
    metrics = read_metrics_jsonl(rec.ticker, rec.run_id)
    return {
        "ticker": ticker,
        "active": handle.thread.is_alive(),
        "run_id": rec.run_id,
        "status": rec.status,
        "metrics": metrics,
        "request": handle.request,
        "elapsed_sec": time.time() - handle.started_ts,
    }


def get_run_snapshot(ticker: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Return a full snapshot for any past or active run."""
    from trading_crew.agentic.rl import load_run
    record = load_run(ticker, run_id)
    if record is None:
        return None
    metrics = read_metrics_jsonl(ticker, run_id)
    return {
        "record": record.to_dict(),
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _worker(
    record: RLRunRecord,
    train_env: TradingEnv,
    eval_env: TradingEnv,
    ppo_cfg: PPOConfig,
    stop_event: threading.Event,
    algorithm: str = "ppo",
) -> None:
    """Thread body — runs the PPO loop, persists, then evaluates."""
    t0 = time.time()

    def on_metric(m: TrainingMetrics) -> None:
        payload = {
            "rollout_index": m.rollout_index,
            "steps_done": m.steps_done,
            "episodes_completed": m.episodes_completed,
            "mean_episode_return": m.mean_episode_return,
            "mean_episode_length": m.mean_episode_length,
            "mean_episode_pnl_pct": m.mean_episode_pnl_pct,
            "sharpe_per_step": m.sharpe_per_step,
            "policy_loss": m.policy_loss,
            "value_loss": m.value_loss,
            "entropy": m.entropy,
            "approx_kl": m.approx_kl,
            "clip_fraction": m.clip_fraction,
            "explained_variance": m.explained_variance,
            "elapsed_sec": m.elapsed_sec,
            "final_action_dist": m.final_action_dist,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        record.metrics.append(payload)
        append_metric_jsonl(record.ticker, record.run_id, payload)

    def on_alt_metric(m: Any) -> None:
        """Generic metric callback for non-PPO trainers — coerces to JSONL."""
        payload = {k: getattr(m, k) for k in m.__dataclass_fields__.keys()}
        payload["ts"] = datetime.now(timezone.utc).isoformat()
        record.metrics.append(payload)
        append_metric_jsonl(record.ticker, record.run_id, payload)

    try:
        algorithm = (algorithm or "ppo").lower()
        if algorithm in ("ppo", "cvar_ppo"):
            trainer = PPOTrainer(
                env=train_env,
                config=ppo_cfg,
                on_metrics=on_metric,
                should_stop=lambda: stop_event.is_set(),
            )
            trainer.train()
            if stop_event.is_set():
                record.status = "stopped"
            else:
                record.status = "completed"
            # Final OOS evaluation on the held-out window.
            record.eval_result = trainer.evaluate(eval_env, n_episodes=3, deterministic=True)
            # Save the policy checkpoint next to the record.
            trainer.save_checkpoint(policy_checkpoint_path(record.ticker, record.run_id))
        elif algorithm == "cql":
            from trading_crew.agentic.rl import (
                CQLConfig,
                CQLTrainer,
                collect_transitions_from_episodes,
            )
            transitions = collect_transitions_from_episodes(
                episodes=[None] * 8,
                env=train_env,
                max_per_episode=64,
            )
            cql_trainer = CQLTrainer(
                env=train_env,
                transitions=transitions,
                config=CQLConfig(total_steps=min(2000, ppo_cfg.total_steps), seed=ppo_cfg.seed),
                on_metrics=on_alt_metric,
                should_stop=lambda: stop_event.is_set(),
            )
            cql_trainer.train()
            record.status = "stopped" if stop_event.is_set() else "completed"
            cql_trainer.save_checkpoint(policy_checkpoint_path(record.ticker, record.run_id))
        elif algorithm == "c51":
            from trading_crew.agentic.rl import C51Config, C51Trainer
            c51_trainer = C51Trainer(
                env=train_env,
                config=C51Config(total_steps=min(2000, ppo_cfg.total_steps), seed=ppo_cfg.seed),
                on_metrics=on_alt_metric,
                should_stop=lambda: stop_event.is_set(),
            )
            c51_trainer.train()
            record.status = "stopped" if stop_event.is_set() else "completed"
            c51_trainer.save_checkpoint(policy_checkpoint_path(record.ticker, record.run_id))
        elif algorithm == "decision_transformer":
            from trading_crew.agentic.rl import (
                DTConfig,
                DecisionTransformerTrainer,
                collect_transitions_from_episodes,
                trajectories_from_transitions,
            )
            transitions = collect_transitions_from_episodes(
                episodes=[None] * 8,
                env=train_env,
                max_per_episode=128,
            )
            trajectories = trajectories_from_transitions(transitions)
            if not trajectories:
                raise RuntimeError("Decision Transformer needs >=1 trajectory from the env")
            dt_trainer = DecisionTransformerTrainer(
                trajectories=trajectories,
                config=DTConfig(total_steps=min(1000, ppo_cfg.total_steps), seed=ppo_cfg.seed),
                on_metrics=on_alt_metric,
                should_stop=lambda: stop_event.is_set(),
            )
            dt_trainer.train()
            record.status = "stopped" if stop_event.is_set() else "completed"
            dt_trainer.save_checkpoint(policy_checkpoint_path(record.ticker, record.run_id))
        else:
            raise ValueError(f"Unknown algorithm: {algorithm!r}")
    except Exception as exc:
        logger.exception("RL training failed for %s/%s", record.ticker, record.run_id)
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
    finally:
        record.duration_sec = time.time() - t0
        try:
            save_run(record)
        except Exception:
            logger.exception("Could not persist final RL record")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _buy_and_hold(df: pd.DataFrame) -> Dict[str, float]:
    """Buy-and-hold baseline over ``df``'s eval window.

    Returns ``{"pnl_pct": ..., "sharpe": ..., "max_drawdown": ...}``.
    The UI surfaces this so the user sees the policy's edge *over*
    passive exposure, not just absolute return (the latter can look
    great in a bull market without any skill being learned).
    """
    if df is None or df.empty:
        return {"pnl_pct": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    closes = df["close"].astype(float).to_numpy()
    if len(closes) < 2:
        return {"pnl_pct": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    rets = np.diff(np.log(closes))
    pnl_pct = float((closes[-1] - closes[0]) / closes[0])
    sharpe = float(rets.mean() / (rets.std() + 1e-8) * np.sqrt(252)) if rets.size > 1 else 0.0
    peak = np.maximum.accumulate(closes)
    dd_series = (peak - closes) / peak
    return {
        "pnl_pct": pnl_pct,
        "sharpe": sharpe,
        "max_drawdown": float(dd_series.max()),
    }
