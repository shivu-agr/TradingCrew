"""PPO trainer for the L4 RL stack.

A from-scratch implementation of Proximal Policy Optimization with all
the standard PPO tricks:

* **Clipped surrogate objective** (Schulman 2017 eq. 7).
* **Generalised Advantage Estimation (GAE)** (Schulman 2016) — variance
  reduction without truncating credit assignment.
* **Mini-batch updates with multiple epochs per rollout** — the
  defining feature of PPO vs vanilla policy gradient.
* **Value-function clipping** — prevents the critic from making huge
  jumps that destabilise the actor (Mnih et al. 2016, used in PPO2).
* **Entropy bonus** — keeps exploration alive long enough to escape
  early local optima ("always go LONG" is a deep trap for a financial
  RL agent on a bull-market window).
* **Gradient clipping** — caps ``||grad||`` to keep updates well-behaved.

We deliberately do not use stable-baselines3 / cleanrl as a dependency.
The training loop runs in seconds on CPU, the code is ~200 lines, and
keeping it in-tree means the audit trail (paper §13.1) stays inside one
process.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from trading_crew.agentic.rl.env import N_ACTIONS, TradingEnv
from trading_crew.agentic.rl.networks import ActorCritic
from trading_crew.agentic.rl.state import FEATURE_DIM


# ---------------------------------------------------------------------------
# Config & metrics
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """Hyperparameters for the PPO trainer.

    Defaults are tuned for the small-history financial-RL regime — a
    deeper network or longer rollout doesn't help when the underlying
    OHLCV history is only ~1000 bars.
    """

    total_steps: int = 20_000
    """Total environment steps to collect across all rollouts."""

    steps_per_rollout: int = 512
    """Steps collected per policy update.  Smaller = more frequent
    updates; larger = lower-variance gradient estimates.  512 is the
    Schulman et al. default."""

    n_epochs: int = 4
    """How many times we reshuffle + iterate over each rollout buffer."""

    minibatch_size: int = 64
    """SGD mini-batch size during the update phase."""

    learning_rate: float = 3e-4
    """Adam LR — the original PPO default, robust on small nets."""

    gamma: float = 0.99
    """Discount factor.  At 0.99 we credit returns ~100 steps out, which
    matches the timescale of a daily-bar swing trade."""

    gae_lambda: float = 0.95
    """GAE bias-variance trade-off.  0.95 = recommended default."""

    clip_eps: float = 0.2
    """PPO clipping range.  0.2 is the canonical setting."""

    value_clip_eps: float = 0.2
    """Value-function clipping range.  Off when 0 (then we use plain MSE)."""

    entropy_coef: float = 0.01
    """Coefficient on the entropy bonus.  Higher = more exploration."""

    value_coef: float = 0.5
    """Coefficient on the value-function loss."""

    max_grad_norm: float = 0.5
    """Gradient clipping threshold."""

    normalize_advantages: bool = True
    """Z-score advantages within each mini-batch — standard PPO trick."""

    hidden_dim: int = 64
    n_hidden_layers: int = 2

    seed: int = 42

    device: str = "cpu"
    """Device for torch tensors.  CPU is faster for a ~8K-param net + a
    tiny env — moving to GPU pays its overhead only at >1M-param nets."""

    # ---------------------------------------------------------------------
    # Phase 2D — risk-sensitive (CVaR) variant.
    # ---------------------------------------------------------------------

    risk_aversion: float = 0.0
    """When ``> 0``, switch to **CVaR-PPO**: advantages whose realised
    rewards fall in the lower ``cvar_alpha`` tail of the rollout get
    extra weight equal to ``1 + risk_aversion``.  ``0`` (default) is
    vanilla PPO with neutral risk preferences.

    Interpretation: ``risk_aversion = 0.5`` means "I'd give up 50% of
    expected return to avoid a worst-case-tail draw".  Mirrors the M5
    fractional-Kelly mindset but inside the policy gradient."""

    cvar_alpha: float = 0.20
    """CVaR quantile.  ``0.20`` = the worst 20% of bar-level rewards in
    each rollout get re-weighted upward."""


@dataclass
class TrainingMetrics:
    """Per-rollout metrics emitted by the trainer.

    Streamed to the UI so the user sees the policy improve in
    real time.  Persisted to the run record so the leaderboard +
    promote-policy flow can show a stable view of training.
    """

    rollout_index: int
    steps_done: int
    episodes_completed: int
    mean_episode_return: float
    mean_episode_length: float
    mean_episode_pnl_pct: float
    sharpe_per_step: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    elapsed_sec: float
    final_action_dist: List[float] = field(default_factory=list)
    """Probability distribution over the action set, evaluated on the
    final observation of the rollout.  Helps diagnose policy collapse
    (all mass on one action == bad)."""


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


@dataclass
class RolloutBuffer:
    """Storage for one PPO rollout.  Plain numpy for fast slicing."""

    obs: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None

    @classmethod
    def empty(cls, capacity: int, obs_dim: int) -> "RolloutBuffer":
        return cls(
            obs=np.zeros((capacity, obs_dim), dtype=np.float32),
            actions=np.zeros(capacity, dtype=np.int64),
            log_probs=np.zeros(capacity, dtype=np.float32),
            values=np.zeros(capacity, dtype=np.float32),
            rewards=np.zeros(capacity, dtype=np.float32),
            dones=np.zeros(capacity, dtype=np.bool_),
        )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PPOTrainer:
    """End-to-end PPO trainer.

    Lifecycle::

        trainer = PPOTrainer(env, PPOConfig())
        for metrics in trainer.train():
            ...  # stream to UI, persist, etc.
        trainer.evaluate(eval_env, n_episodes=10)
        trainer.save_checkpoint(path)
    """

    def __init__(
        self,
        env: TradingEnv,
        config: Optional[PPOConfig] = None,
        *,
        on_metrics: Optional[Callable[[TrainingMetrics], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.env = env
        self.config = config or PPOConfig()
        self.on_metrics = on_metrics
        self.should_stop = should_stop or (lambda: False)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.device = torch.device(self.config.device)
        self.model = ActorCritic(
            obs_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=self.config.hidden_dim,
            n_hidden_layers=self.config.n_hidden_layers,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Episode-tracking accumulators (reset between rollouts).
        self._ep_returns: List[float] = []
        self._ep_lengths: List[int] = []
        self._ep_pnl_pcts: List[float] = []
        self._current_ep_return = 0.0
        self._current_ep_length = 0
        self._current_ep_start_nav: Optional[float] = None
        self._last_eval_nav: Optional[float] = None

    # -- public training loop --------------------------------------------

    def train(self) -> List[TrainingMetrics]:
        """Run training until ``total_steps`` is reached.  Returns the
        list of metrics emitted (also pushed live via ``on_metrics``)."""
        all_metrics: List[TrainingMetrics] = []
        obs = self.env.reset(seed=self.config.seed)
        self._current_ep_start_nav = self.env._last_nav  # type: ignore[attr-defined]
        steps_done = 0
        rollout_index = 0

        while steps_done < self.config.total_steps and not self.should_stop():
            rollout_index += 1
            t0 = time.time()
            # Phase 2D — push the volatility-curriculum pointer to the
            # env (no-op when the env didn't enable the curriculum).
            if hasattr(self.env, "set_curriculum_progress"):
                progress = steps_done / max(1, self.config.total_steps)
                try:
                    self.env.set_curriculum_progress(progress)
                except Exception:  # never let the curriculum break training
                    pass
            buf, last_value, obs = self._collect_rollout(obs)
            self._compute_gae(buf, last_value)
            policy_loss, value_loss, entropy, approx_kl, clip_frac, explained = self._update(buf)
            steps_done += len(buf.rewards)

            mean_ret = float(np.mean(self._ep_returns)) if self._ep_returns else 0.0
            mean_len = float(np.mean(self._ep_lengths)) if self._ep_lengths else 0.0
            mean_pnl = float(np.mean(self._ep_pnl_pcts)) if self._ep_pnl_pcts else 0.0
            # Per-step Sharpe is more robust than per-episode Sharpe when
            # rollouts span multiple episodes of varying length.
            r = buf.rewards
            sharpe = float(r.mean() / (r.std() + 1e-8) * math.sqrt(252)) if r.size > 1 else 0.0
            # Action distribution on the last observation — diagnostic
            # for policy collapse.
            with torch.no_grad():
                probs = self.model.action_probs(
                    torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                ).cpu().numpy().reshape(-1).tolist()

            metric = TrainingMetrics(
                rollout_index=rollout_index,
                steps_done=steps_done,
                episodes_completed=len(self._ep_returns),
                mean_episode_return=mean_ret,
                mean_episode_length=mean_len,
                mean_episode_pnl_pct=mean_pnl,
                sharpe_per_step=sharpe,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                approx_kl=approx_kl,
                clip_fraction=clip_frac,
                explained_variance=explained,
                elapsed_sec=time.time() - t0,
                final_action_dist=probs,
            )
            all_metrics.append(metric)
            if self.on_metrics is not None:
                self.on_metrics(metric)

            # Reset per-rollout episode accumulators so the dashboard
            # shows *recent* progress, not a lifetime average.
            self._ep_returns.clear()
            self._ep_lengths.clear()
            self._ep_pnl_pcts.clear()

        return all_metrics

    # -- rollout ----------------------------------------------------------

    def _collect_rollout(self, obs: np.ndarray):
        """Step the env for ``steps_per_rollout`` actions, storing each."""
        capacity = self.config.steps_per_rollout
        buf = RolloutBuffer.empty(capacity, FEATURE_DIM)

        for i in range(capacity):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            action, log_prob, value = self.model.act(obs_t, deterministic=False)
            buf.obs[i] = obs
            buf.actions[i] = action
            buf.log_probs[i] = log_prob
            buf.values[i] = value
            step = self.env.step(action)
            buf.rewards[i] = step.reward
            buf.dones[i] = step.done
            self._current_ep_return += step.reward
            self._current_ep_length += 1
            if step.done:
                self._ep_returns.append(self._current_ep_return)
                self._ep_lengths.append(self._current_ep_length)
                if self._current_ep_start_nav is not None:
                    ending_nav = step.info.get("nav", self._current_ep_start_nav)
                    self._ep_pnl_pcts.append(
                        (ending_nav - self._current_ep_start_nav)
                        / max(self._current_ep_start_nav, 1e-8)
                    )
                self._current_ep_return = 0.0
                self._current_ep_length = 0
                obs = self.env.reset()
                self._current_ep_start_nav = self.env._last_nav  # type: ignore[attr-defined]
            else:
                obs = step.obs

        # Bootstrap value for the truncated last step.
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_value = self.model(obs_t)
            last_value = float(last_value.item())

        return buf, last_value, obs

    def _compute_gae(self, buf: RolloutBuffer, last_value: float) -> None:
        """GAE-λ advantage estimation, returns = adv + values.

        Computed in numpy reverse-time-loop (fast and clear)."""
        advantages = np.zeros_like(buf.rewards)
        last_gae = 0.0
        for t in reversed(range(len(buf.rewards))):
            if t == len(buf.rewards) - 1:
                next_non_terminal = 0.0 if buf.dones[t] else 1.0
                next_value = last_value
            else:
                next_non_terminal = 0.0 if buf.dones[t] else 1.0
                next_value = buf.values[t + 1]
            delta = buf.rewards[t] + self.config.gamma * next_value * next_non_terminal - buf.values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        # Phase 2D — CVaR-PPO advantage re-weighting.  We multiply the
        # per-step advantage by ``1 + risk_aversion`` for any step whose
        # realised reward sits below the ``cvar_alpha`` empirical
        # quantile of the rollout.  Net effect: the policy pays *extra*
        # attention to avoiding tail-loss steps, mirroring how the M5
        # CVaR sizing cap re-weights position size away from the lower
        # quantile of returns.
        if self.config.risk_aversion > 0.0 and buf.rewards.size > 1:
            q = float(np.quantile(buf.rewards, self.config.cvar_alpha))
            tail_mask = buf.rewards <= q
            advantages = np.where(
                tail_mask,
                advantages * (1.0 + self.config.risk_aversion),
                advantages,
            )

        buf.advantages = advantages
        buf.returns = advantages + buf.values

    # -- update -----------------------------------------------------------

    def _update(self, buf: RolloutBuffer):
        """Run ``n_epochs`` of mini-batch PPO updates over ``buf``.

        Returns scalar diagnostics averaged across the whole pass."""
        cfg = self.config
        obs_t = torch.as_tensor(buf.obs, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(buf.actions, dtype=torch.long, device=self.device)
        old_log_probs_t = torch.as_tensor(buf.log_probs, dtype=torch.float32, device=self.device)
        old_values_t = torch.as_tensor(buf.values, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(buf.returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(buf.advantages, dtype=torch.float32, device=self.device)

        n = obs_t.shape[0]
        idx = np.arange(n)

        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fracs = []

        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                if len(mb) < 2:
                    continue
                mb_obs = obs_t[mb]
                mb_actions = actions_t[mb]
                mb_old_log_probs = old_log_probs_t[mb]
                mb_old_values = old_values_t[mb]
                mb_returns = returns_t[mb]
                mb_adv = advantages_t[mb]

                if cfg.normalize_advantages:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                new_log_probs, entropy, new_values = self.model.evaluate(mb_obs, mb_actions)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                # Clipped surrogate (Schulman 2017 eq. 7).
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value-function loss (with optional clipping).
                if cfg.value_clip_eps > 0:
                    clipped_v = mb_old_values + torch.clamp(
                        new_values - mb_old_values,
                        -cfg.value_clip_eps,
                        cfg.value_clip_eps,
                    )
                    v_loss_unclipped = (new_values - mb_returns).pow(2)
                    v_loss_clipped = (clipped_v - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (new_values - mb_returns).pow(2).mean()

                entropy_bonus = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # Approximate KL — the "robust" estimator from
                # http://joschu.net/blog/kl-approx.html.  Easier on the
                # numerics than mean(old - new).
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean()
                    )

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_bonus.item()))
                approx_kls.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))

        # Explained variance — how well the critic predicts returns.
        # Computed once over the whole rollout.
        with torch.no_grad():
            var_returns = returns_t.var().item()
            explained = (
                1.0 - ((returns_t - old_values_t).var().item() / (var_returns + 1e-8))
                if var_returns > 1e-8 else 0.0
            )

        return (
            float(np.mean(policy_losses)) if policy_losses else 0.0,
            float(np.mean(value_losses)) if value_losses else 0.0,
            float(np.mean(entropies)) if entropies else 0.0,
            float(np.mean(approx_kls)) if approx_kls else 0.0,
            float(np.mean(clip_fracs)) if clip_fracs else 0.0,
            float(explained),
        )

    # -- evaluation -------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        eval_env: TradingEnv,
        n_episodes: int = 5,
        deterministic: bool = True,
    ) -> dict:
        """Run ``n_episodes`` greedy rollouts in ``eval_env``.

        Returns aggregate stats: mean return, mean PnL %, max drawdown,
        per-action frequency.  Used to score a trained policy out-of-
        sample before letting the user promote it.
        """
        ep_returns: List[float] = []
        ep_pnl_pcts: List[float] = []
        ep_drawdowns: List[float] = []
        ep_lengths: List[int] = []
        action_counts = np.zeros(N_ACTIONS, dtype=np.int64)

        for _ in range(n_episodes):
            obs = eval_env.reset()
            start_nav = eval_env._last_nav  # type: ignore[attr-defined]
            total_reward = 0.0
            steps = 0
            while True:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                action, _, _ = self.model.act(obs_t, deterministic=deterministic)
                action_counts[action] += 1
                step = eval_env.step(action)
                total_reward += step.reward
                steps += 1
                if step.done:
                    ep_returns.append(total_reward)
                    ep_lengths.append(steps)
                    ep_pnl_pcts.append((step.info["nav"] - start_nav) / max(start_nav, 1e-8))
                    ep_drawdowns.append(step.info["drawdown"])
                    break
                obs = step.obs

        return {
            "n_episodes": n_episodes,
            "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "mean_pnl_pct": float(np.mean(ep_pnl_pcts)) if ep_pnl_pcts else 0.0,
            "median_pnl_pct": float(np.median(ep_pnl_pcts)) if ep_pnl_pcts else 0.0,
            "mean_drawdown": float(np.mean(ep_drawdowns)) if ep_drawdowns else 0.0,
            "max_drawdown": float(np.max(ep_drawdowns)) if ep_drawdowns else 0.0,
            "mean_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
            "action_freq": (action_counts / max(action_counts.sum(), 1)).tolist(),
        }

    # -- checkpointing ----------------------------------------------------

    def save_checkpoint(self, path) -> None:
        """Persist the policy + optimiser to ``path``.  ``.pt`` extension recommended."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": _config_to_dict(self.config),
                "obs_dim": FEATURE_DIM,
                "n_actions": N_ACTIONS,
            },
            str(path),
        )

    @classmethod
    def load_model(cls, path, device: str = "cpu") -> ActorCritic:
        """Load just the policy network (no optimiser) from a checkpoint."""
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = ActorCritic(
            obs_dim=ckpt.get("obs_dim", FEATURE_DIM),
            n_actions=ckpt.get("n_actions", N_ACTIONS),
            hidden_dim=cfg.get("hidden_dim", 64),
            n_hidden_layers=cfg.get("n_hidden_layers", 2),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model


def _config_to_dict(cfg: PPOConfig) -> dict:
    return {
        k: getattr(cfg, k) for k in cfg.__dataclass_fields__.keys()
    }
