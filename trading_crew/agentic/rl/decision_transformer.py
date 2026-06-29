"""Decision Transformer — offline conditional sequence modelling.

Chen et al. 2021 "Decision Transformer: Reinforcement Learning via
Sequence Modeling" (https://arxiv.org/abs/2106.01345).

The trick: cast the offline-RL problem as **supervised sequence
modelling**.  Each training example is the sequence
``(R̂_1, s_1, a_1, R̂_2, s_2, a_2, …)`` where ``R̂_t`` is the
*remaining* return-to-go from step ``t``.  At inference time, condition
on a *desired* return-to-go and let the model auto-regressively predict
the next action.

Strengths for our trading L4 pipeline:

* No bootstrapping → no Q-divergence risk on a small offline dataset.
* Easy to expose a "target Sharpe / target return" slider in the UI
  at inference time (we condition on different R̂_1 values).
* Stitches sub-trajectories from the offline buffer without
  re-running the env.

Caveat: classical Decision Transformer uses GPT-2-scale models.  Ours
is intentionally a *toy* transformer (1 layer, 2 heads, hidden=64) so
training fits inside our seconds-budget on CPU.  It's enough to teach
the model that ``high R̂ ⇒ aggressive action``.
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

from trading_crew.agentic.rl.env import N_ACTIONS
from trading_crew.agentic.rl.state import FEATURE_DIM


@dataclass
class DTConfig:
    """Hyperparameters for the Decision Transformer trainer."""

    total_steps: int = 5_000
    batch_size: int = 32
    context_len: int = 20
    learning_rate: float = 3e-4
    hidden_dim: int = 64
    n_heads: int = 2
    n_layers: int = 1
    seed: int = 42
    device: str = "cpu"


@dataclass
class DTTrainingMetrics:
    step: int
    loss: float
    elapsed_sec: float


class DecisionTransformer(nn.Module):
    """Minimal Decision Transformer.

    Tokens per timestep: ``(R̂, s, a)`` → 3 tokens, each projected to
    ``hidden_dim``.  Causal self-attention then predicts the next
    action token from the (R̂, s) prefix at each position.
    """

    def __init__(self, obs_dim: int, n_actions: int, *, hidden_dim: int = 64, n_heads: int = 2, n_layers: int = 1, context_len: int = 20) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.context_len = context_len

        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(obs_dim, hidden_dim)
        self.embed_action = nn.Embedding(n_actions, hidden_dim)
        self.position_embedding = nn.Embedding(context_len * 3, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.action_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, returns_to_go: torch.Tensor, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return action logits at every timestep ``(B, T, n_actions)``."""
        B, T = returns_to_go.shape
        r_tok = self.embed_return(returns_to_go.unsqueeze(-1))
        s_tok = self.embed_state(states)
        a_tok = self.embed_action(actions)
        # Interleave (R, s, a, R, s, a, …) so timestep t has 3 tokens.
        seq = torch.stack([r_tok, s_tok, a_tok], dim=2).reshape(B, T * 3, self.hidden_dim)
        positions = torch.arange(T * 3, device=seq.device).unsqueeze(0).expand(B, -1)
        seq = seq + self.position_embedding(positions)
        # Causal mask so token i can attend only to tokens <= i.
        mask = torch.triu(torch.ones(T * 3, T * 3, device=seq.device, dtype=torch.bool), diagonal=1)
        out = self.transformer(seq, mask=mask)
        # Take the *state* tokens (every 3rd token starting at index 1)
        # — they're the prefix from which we predict the next action.
        state_out = out[:, 1::3, :]  # (B, T, hidden_dim)
        return self.action_head(state_out)  # (B, T, n_actions)


class DecisionTransformerTrainer:
    def __init__(
        self,
        trajectories: Sequence[dict],
        config: Optional[DTConfig] = None,
        *,
        on_metrics: Optional[Callable[[DTTrainingMetrics], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        if not trajectories:
            raise ValueError("Decision Transformer needs >=1 trajectory")
        self.trajectories = list(trajectories)
        self.config = config or DTConfig()
        self.on_metrics = on_metrics
        self.should_stop = should_stop or (lambda: False)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.device = torch.device(self.config.device)

        self.model = DecisionTransformer(
            obs_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            hidden_dim=self.config.hidden_dim,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            context_len=self.config.context_len,
        ).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def _sample_batch(self, rng: np.random.Generator):
        """Sample ``batch_size`` slices of length ``context_len`` from the offline trajectories."""
        cfg = self.config
        states_b, actions_b, rtg_b = [], [], []
        for _ in range(cfg.batch_size):
            traj = self.trajectories[rng.integers(0, len(self.trajectories))]
            L = len(traj["states"])
            if L < cfg.context_len:
                pad = cfg.context_len - L
                states = np.concatenate([np.zeros((pad, FEATURE_DIM), dtype=np.float32), traj["states"]], axis=0)
                actions = np.concatenate([np.zeros(pad, dtype=np.int64), traj["actions"]], axis=0)
                rtg = np.concatenate([np.zeros(pad, dtype=np.float32), traj["rtg"]], axis=0)
            else:
                start = int(rng.integers(0, L - cfg.context_len + 1))
                states = traj["states"][start:start + cfg.context_len]
                actions = traj["actions"][start:start + cfg.context_len]
                rtg = traj["rtg"][start:start + cfg.context_len]
            states_b.append(states)
            actions_b.append(actions)
            rtg_b.append(rtg)
        s = torch.as_tensor(np.stack(states_b), dtype=torch.float32, device=self.device)
        a = torch.as_tensor(np.stack(actions_b), dtype=torch.long, device=self.device)
        r = torch.as_tensor(np.stack(rtg_b), dtype=torch.float32, device=self.device)
        return s, a, r

    def train(self) -> List[DTTrainingMetrics]:
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        all_metrics: List[DTTrainingMetrics] = []
        for step in range(1, cfg.total_steps + 1):
            if self.should_stop():
                break
            t0 = time.time()
            states, actions, rtg = self._sample_batch(rng)
            logits = self.model(rtg, states, actions)
            # Predict the *current* action token at every timestep —
            # the causal mask guarantees we don't peek.
            loss = F.cross_entropy(logits.reshape(-1, N_ACTIONS), actions.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            metric = DTTrainingMetrics(
                step=step,
                loss=float(loss.item()),
                elapsed_sec=time.time() - t0,
            )
            all_metrics.append(metric)
            if self.on_metrics is not None:
                self.on_metrics(metric)
        return all_metrics

    def save_checkpoint(self, path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": {k: getattr(self.config, k) for k in self.config.__dataclass_fields__.keys()},
                "obs_dim": FEATURE_DIM,
                "n_actions": N_ACTIONS,
                "algorithm": "decision_transformer",
            },
            str(path),
        )


def trajectories_from_transitions(transitions: Sequence[dict], gamma: float = 0.99) -> List[dict]:
    """Group consecutive transitions into episodes + compute return-to-go.

    Each output dict has::

        {"states": np.ndarray, "actions": np.ndarray, "rtg": np.ndarray}

    A new episode is started whenever ``done=True`` is seen.
    """
    trajectories: List[dict] = []
    current = {"states": [], "actions": [], "rewards": []}
    for tr in transitions:
        current["states"].append(np.asarray(tr["obs"], dtype=np.float32))
        current["actions"].append(int(tr["action"]))
        current["rewards"].append(float(tr["reward"]))
        if tr.get("done"):
            trajectories.append(_finalise_trajectory(current, gamma))
            current = {"states": [], "actions": [], "rewards": []}
    if current["states"]:
        trajectories.append(_finalise_trajectory(current, gamma))
    return trajectories


def _finalise_trajectory(t: dict, gamma: float) -> dict:
    states = np.stack(t["states"])
    actions = np.array(t["actions"], dtype=np.int64)
    rewards = np.array(t["rewards"], dtype=np.float32)
    # Return-to-go = sum of discounted future rewards from each step.
    rtg = np.zeros_like(rewards)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = rewards[i] + gamma * acc
        rtg[i] = acc
    return {"states": states, "actions": actions, "rtg": rtg}
