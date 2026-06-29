"""Actor-critic network for L4 PPO.

Design choices
--------------

* **Shared backbone** — two FC layers shared by the policy head and the
  value head.  Sharing improves sample efficiency on tiny histories and
  is the standard PPO architecture from the original paper (Schulman et
  al., 2017).
* **Small.**  Two hidden layers of width 64 → ~8K parameters total.
  Trains in seconds on CPU.  Bigger nets overfit the trivial amount of
  daily OHLCV data we have to work with (~1000 bars per ticker).
* **Tanh activations** — the standard PPO choice (Andrychowicz et al.,
  2020 "What Matters in On-Policy RL"); ReLU works too but tanh keeps
  features bounded which helps with our clip-normalised observations.
* **Orthogonal init with appropriate gain** — important PPO trick: the
  policy head gets gain 0.01 so it starts close to a uniform
  distribution and exploration is preserved early.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


def _orthogonal_(layer: nn.Linear, gain: float) -> nn.Linear:
    """Orthogonal init for a linear layer with zero bias.

    This is the standard PPO initialisation — improves convergence
    materially over PyTorch's default Kaiming uniform on small nets
    (Andrychowicz et al., 2020).
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic MLP for a discrete action space.

    Forward returns ``(logits, value)``; callers wrap the logits in a
    :class:`torch.distributions.Categorical` to sample / score actions.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        *,
        n_hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")

        layers = []
        in_dim = obs_dim
        for _ in range(n_hidden_layers):
            layers.append(_orthogonal_(nn.Linear(in_dim, hidden_dim), gain=2 ** 0.5))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.policy_head = _orthogonal_(nn.Linear(hidden_dim, n_actions), gain=0.01)
        self.value_head = _orthogonal_(nn.Linear(hidden_dim, 1), gain=1.0)

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

    # -- forward ---------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, value).  ``obs`` is ``(B, obs_dim)``."""
        x = self.backbone(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value

    # -- convenience -----------------------------------------------------

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[int, float, float]:
        """Sample an action.  Returns ``(action, log_prob, value)``.

        Used by the rollout loop in :mod:`ppo` and by the inference
        client in :mod:`inference`.  When ``deterministic`` is True we
        take the argmax instead of sampling — that's the right choice
        at deployment time but disastrous during training (no exploration).
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used by PPO update: re-score (obs, action) pairs.

        Returns ``(log_probs, entropies, values)`` — one element per
        row of the input batch.
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value

    @torch.no_grad()
    def action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Probability distribution over actions for ``obs``.

        Used by the inference client to surface action probabilities to
        the LLM — the analyst gets to see "policy says 60% LONG, 30%
        NEUTRAL, 10% SHORT" rather than just an argmax.
        """
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        logits, _ = self.forward(obs)
        return torch.softmax(logits, dim=-1)
