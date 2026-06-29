"""Distributional RL — C51 categorical DQN.

C51 (Bellemare et al. 2017 "A Distributional Perspective on
Reinforcement Learning", https://arxiv.org/abs/1707.06887) replaces the
scalar value estimate ``Q(s, a)`` with a **discrete probability
distribution** over a fixed support of return atoms ``z_1 … z_N``.
The Bellman update becomes a projection of the shifted-discounted
distribution back onto the support.

Why we want it here
-------------------

The M5 risk gate cares about the **lower-quantile** of return — not
the expected return.  A C51 critic exposes the full return distribution
at inference time, so the gate can read off any tail statistic
(VaR, CVaR, P(loss > X%)) directly.

Implementation
--------------

We use a small categorical head bolted onto the same backbone PPO uses.
51 atoms × 7 actions × hidden_dim — adds ~3K params on top of PPO,
trains fast enough to be useful as an alternative trainer in the
RL-Training tab.
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
from trading_crew.agentic.rl.state import FEATURE_DIM


@dataclass
class C51Config:
    """Hyperparameters for the C51 trainer."""

    total_steps: int = 10_000
    batch_size: int = 256
    gamma: float = 0.99
    learning_rate: float = 3e-4
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    target_polyak: float = 0.995
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    seed: int = 42
    device: str = "cpu"


@dataclass
class C51TrainingMetrics:
    step: int
    loss: float
    mean_return: float
    var_return: float
    elapsed_sec: float


class C51Network(nn.Module):
    """Categorical critic emitting ``(B, n_actions, n_atoms)`` logits."""

    def __init__(self, obs_dim: int, n_actions: int, *, hidden_dim: int = 64, n_hidden_layers: int = 2, n_atoms: int = 51) -> None:
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, n_actions * n_atoms)
        self.n_actions = n_actions
        self.n_atoms = n_atoms

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return *probability* tensor with shape ``(B, n_actions, n_atoms)``."""
        x = self.backbone(obs)
        logits = self.head(x).view(-1, self.n_actions, self.n_atoms)
        return F.softmax(logits, dim=-1)


class C51Trainer:
    """Offline / on-env C51 trainer.

    Args:
        env: TradingEnv to step through (epsilon-greedy exploration).
        config: hyperparams.
    """

    def __init__(
        self,
        env: TradingEnv,
        config: Optional[C51Config] = None,
        *,
        on_metrics: Optional[Callable[[C51TrainingMetrics], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.env = env
        self.config = config or C51Config()
        self.on_metrics = on_metrics
        self.should_stop = should_stop or (lambda: False)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.device = torch.device(self.config.device)
        self.q_net = C51Network(
            obs_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=self.config.hidden_dim,
            n_hidden_layers=self.config.n_hidden_layers,
            n_atoms=self.config.n_atoms,
        ).to(self.device)
        self.target_net = C51Network(
            obs_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=self.config.hidden_dim,
            n_hidden_layers=self.config.n_hidden_layers,
            n_atoms=self.config.n_atoms,
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.learning_rate)

        # Fixed support [v_min, v_max] with n_atoms points.
        self.atoms = torch.linspace(self.config.v_min, self.config.v_max, self.config.n_atoms, device=self.device)
        self.delta_z = (self.config.v_max - self.config.v_min) / (self.config.n_atoms - 1)

    def _expected_q(self, probs: torch.Tensor) -> torch.Tensor:
        """``E[Z]`` per (B, A) — collapse the atom dim."""
        return (probs * self.atoms.view(1, 1, -1)).sum(dim=-1)

    def train(self) -> List[C51TrainingMetrics]:
        cfg = self.config
        obs = self.env.reset(seed=cfg.seed)
        all_metrics: List[C51TrainingMetrics] = []

        # Small replay buffer (single-list FIFO is fine at this scale).
        buf: List[dict] = []
        epsilon = 0.5

        for step in range(1, cfg.total_steps + 1):
            if self.should_stop():
                break
            t0 = time.time()

            # ε-greedy step in the env.
            if np.random.rand() < epsilon:
                action = int(np.random.randint(0, N_ACTIONS))
            else:
                with torch.no_grad():
                    probs = self.q_net(torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0))
                    q = self._expected_q(probs)[0]
                    action = int(q.argmax().item())
            step_res = self.env.step(action)
            buf.append({
                "obs": obs,
                "action": action,
                "reward": float(step_res.reward),
                "next_obs": step_res.obs,
                "done": bool(step_res.done),
            })
            if len(buf) > 10_000:
                buf.pop(0)
            obs = step_res.obs if not step_res.done else self.env.reset()

            if len(buf) < cfg.batch_size:
                continue

            idx = np.random.randint(0, len(buf), size=cfg.batch_size)
            batch = [buf[i] for i in idx]
            b_obs = torch.as_tensor(np.stack([b["obs"] for b in batch]), dtype=torch.float32, device=self.device)
            b_act = torch.as_tensor([b["action"] for b in batch], dtype=torch.long, device=self.device)
            b_rew = torch.as_tensor([b["reward"] for b in batch], dtype=torch.float32, device=self.device)
            b_next = torch.as_tensor(np.stack([b["next_obs"] for b in batch]), dtype=torch.float32, device=self.device)
            b_done = torch.as_tensor([b["done"] for b in batch], dtype=torch.float32, device=self.device)

            # Categorical projection (Bellemare eq. 7).
            with torch.no_grad():
                next_probs = self.target_net(b_next)
                next_q = self._expected_q(next_probs)
                next_action = next_q.argmax(dim=-1)
                next_dist = next_probs.gather(
                    1, next_action.view(-1, 1, 1).expand(-1, 1, cfg.n_atoms)
                ).squeeze(1)
                tz = b_rew.unsqueeze(-1) + (1.0 - b_done.unsqueeze(-1)) * cfg.gamma * self.atoms.view(1, -1)
                tz = tz.clamp(cfg.v_min, cfg.v_max)
                b_idx = (tz - cfg.v_min) / self.delta_z
                lo = b_idx.floor().long()
                hi = b_idx.ceil().long()
                # Equal upper/lower edge.
                hi = torch.where(lo == hi, hi.clamp(max=cfg.n_atoms - 1), hi)
                lo_w = (hi.float() - b_idx).clamp(min=0.0)
                hi_w = (b_idx - lo.float()).clamp(min=0.0)
                target_dist = torch.zeros_like(next_dist)
                target_dist.scatter_add_(1, lo, next_dist * lo_w)
                target_dist.scatter_add_(1, hi, next_dist * hi_w)

            probs = self.q_net(b_obs)
            chosen_dist = probs.gather(
                1, b_act.view(-1, 1, 1).expand(-1, 1, cfg.n_atoms)
            ).squeeze(1)
            # Cross-entropy loss between target_dist (no grad) and chosen_dist.
            log_chosen = (chosen_dist + 1e-8).log()
            loss = -(target_dist * log_chosen).sum(dim=-1).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
            self.optimizer.step()

            # Soft target update.
            with torch.no_grad():
                tau = cfg.target_polyak
                for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.data.mul_(tau).add_(p.data, alpha=1 - tau)

            mean_q = float(self._expected_q(probs).mean().item())
            var_q = float(self._expected_q(probs).var().item())
            epsilon = max(0.05, epsilon * 0.999)

            metric = C51TrainingMetrics(
                step=step,
                loss=float(loss.item()),
                mean_return=mean_q,
                var_return=var_q,
                elapsed_sec=time.time() - t0,
            )
            all_metrics.append(metric)
            if self.on_metrics is not None:
                self.on_metrics(metric)
        return all_metrics

    def save_checkpoint(self, path) -> None:
        torch.save(
            {
                "model": self.q_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": {k: getattr(self.config, k) for k in self.config.__dataclass_fields__.keys()},
                "obs_dim": FEATURE_DIM,
                "n_actions": N_ACTIONS,
                "algorithm": "c51",
            },
            str(path),
        )
