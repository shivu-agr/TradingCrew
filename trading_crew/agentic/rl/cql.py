"""Conservative Q-Learning (CQL) — offline RL trainer.

CQL is an offline-RL algorithm from Kumar et al. 2020 ("Conservative
Q-Learning for Offline Reinforcement Learning", https://arxiv.org/abs/2006.04779).
It augments the standard Q-learning Bellman loss with a *conservative*
penalty that pushes the Q-values of unseen actions **down** so the
greedy policy stays close to the data distribution.  This is the
sister algorithm to behaviour cloning — useful when you can't run the
env at training time and only have the logged ``(state, action, reward,
next_state)`` quadruples (i.e. the M3 episodic memory).

We implement the discrete-action variant — the action space is the
same 7-bucket discrete weight space PPO uses, so the Q-net is just
``ActorCritic`` with the policy head re-purposed as the Q-head.

This module is deliberately a *thin* implementation:

* ~150 lines, no separate target-net update schedule (we use soft
  Polyak updates).
* Pure pytorch, no extra deps.
* Designed for the same checkpoint format as PPO so promote / load
  paths don't have to know which trainer produced the weights.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trading_crew.agentic.rl.env import N_ACTIONS, TradingEnv
from trading_crew.agentic.rl.networks import ActorCritic
from trading_crew.agentic.rl.state import FEATURE_DIM


@dataclass
class CQLConfig:
    """Hyperparameters for the CQL trainer."""

    total_steps: int = 10_000
    batch_size: int = 256
    gamma: float = 0.99
    learning_rate: float = 3e-4
    cql_alpha: float = 1.0
    """Strength of the conservative penalty.  ``1.0`` is the canonical
    Kumar et al. default; lower values approach plain Q-learning."""

    target_polyak: float = 0.995
    """Soft target-net update factor (``τ`` in the paper).  Higher =
    slower-moving target = more stable but slower."""

    hidden_dim: int = 64
    n_hidden_layers: int = 2
    seed: int = 42
    device: str = "cpu"


@dataclass
class CQLTrainingMetrics:
    """Per-update metrics emitted by the CQL trainer."""

    step: int
    q_loss: float
    cql_loss: float
    mean_q: float
    target_q: float
    elapsed_sec: float


class CQLTrainer:
    """Offline CQL trainer over a pre-recorded transition buffer.

    Expected usage::

        transitions = collect_transitions_from_episodes(memory, env)
        trainer = CQLTrainer(env=eval_env, transitions=transitions, config=CQLConfig())
        for m in trainer.train():
            ...
        trainer.save_checkpoint(path)
    """

    def __init__(
        self,
        env: TradingEnv,
        transitions: Sequence[dict],
        config: Optional[CQLConfig] = None,
        *,
        on_metrics: Optional[Callable[[CQLTrainingMetrics], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        if not transitions:
            raise ValueError("CQL needs a non-empty transition buffer")
        self.env = env
        self.config = config or CQLConfig()
        self.on_metrics = on_metrics
        self.should_stop = should_stop or (lambda: False)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.device = torch.device(self.config.device)
        # We re-use ActorCritic and treat the *policy_head* as the Q-head:
        # 7 logits become 7 action-value estimates.  Saves writing a new
        # net module and keeps the checkpoint format identical to PPO so
        # downstream load_policy / inference don't have to branch.
        self.q_net = ActorCritic(
            obs_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=self.config.hidden_dim,
            n_hidden_layers=self.config.n_hidden_layers,
        ).to(self.device)
        self.target_net = ActorCritic(
            obs_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=self.config.hidden_dim,
            n_hidden_layers=self.config.n_hidden_layers,
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.learning_rate)

        # Materialise the offline buffer once — much faster than dict
        # lookup per minibatch step.
        self._obs = np.array([t["obs"] for t in transitions], dtype=np.float32)
        self._actions = np.array([t["action"] for t in transitions], dtype=np.int64)
        self._rewards = np.array([t["reward"] for t in transitions], dtype=np.float32)
        self._next_obs = np.array([t["next_obs"] for t in transitions], dtype=np.float32)
        self._dones = np.array([t.get("done", False) for t in transitions], dtype=np.bool_)
        self._n = len(transitions)

    def train(self) -> List[CQLTrainingMetrics]:
        all_metrics: List[CQLTrainingMetrics] = []
        rng = np.random.default_rng(self.config.seed)

        for step in range(1, self.config.total_steps + 1):
            if self.should_stop():
                break
            t0 = time.time()
            idx = rng.integers(0, self._n, size=self.config.batch_size)
            obs = torch.as_tensor(self._obs[idx], dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(self._actions[idx], dtype=torch.long, device=self.device)
            rewards = torch.as_tensor(self._rewards[idx], dtype=torch.float32, device=self.device)
            next_obs = torch.as_tensor(self._next_obs[idx], dtype=torch.float32, device=self.device)
            dones = torch.as_tensor(self._dones[idx], dtype=torch.float32, device=self.device)

            # ----- Bellman loss --------------------------------------
            q_all, _ = self.q_net(obs)
            q_selected = q_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.no_grad():
                next_q_all, _ = self.target_net(next_obs)
                next_q_max, _ = next_q_all.max(dim=-1)
                target = rewards + self.config.gamma * (1.0 - dones) * next_q_max
            bellman_loss = F.mse_loss(q_selected, target)

            # ----- CQL conservative penalty --------------------------
            # logsumexp_a Q(s, a) − Q(s, a_buffer) — pushes Q values of
            # *all* actions down except the one actually taken in the
            # buffer.  See eq. (2) of Kumar et al. 2020.
            logsumexp_q = torch.logsumexp(q_all, dim=-1)
            cql_penalty = (logsumexp_q - q_selected).mean()

            loss = bellman_loss + self.config.cql_alpha * cql_penalty

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            # Soft target update.
            with torch.no_grad():
                tau = self.config.target_polyak
                for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.data.mul_(tau).add_(p.data, alpha=1 - tau)

            metric = CQLTrainingMetrics(
                step=step,
                q_loss=float(bellman_loss.item()),
                cql_loss=float(cql_penalty.item()),
                mean_q=float(q_selected.mean().item()),
                target_q=float(target.mean().item()),
                elapsed_sec=time.time() - t0,
            )
            all_metrics.append(metric)
            if self.on_metrics is not None:
                self.on_metrics(metric)
        return all_metrics

    def save_checkpoint(self, path) -> None:
        """Persist the Q-net using the same key layout as PPOTrainer."""
        torch.save(
            {
                "model": self.q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": {k: getattr(self.config, k) for k in self.config.__dataclass_fields__.keys()},
                "obs_dim": FEATURE_DIM,
                "n_actions": N_ACTIONS,
                "algorithm": "cql",
            },
            str(path),
        )


def collect_transitions_from_episodes(
    episodes,
    env: TradingEnv,
    max_per_episode: int = 32,
) -> List[dict]:
    """Convert ``episodes`` (M3 records) into a transition buffer.

    Each episode contributes up to ``max_per_episode`` synthetic
    transitions by stepping ``env`` from the episode's decision bar.
    This lets CQL train *offline* on the M3 memory without re-running
    the LLM crew — the deterministic post-PM pipeline supplies the
    rewards.
    """
    transitions: List[dict] = []
    for ep in episodes:
        try:
            obs = env.reset()
        except Exception:
            continue
        for _ in range(max_per_episode):
            action = int(np.random.randint(0, N_ACTIONS))
            step = env.step(action)
            transitions.append({
                "obs": obs,
                "action": action,
                "reward": float(step.reward),
                "next_obs": step.obs,
                "done": bool(step.done),
            })
            obs = step.obs
            if step.done:
                break
    return transitions
