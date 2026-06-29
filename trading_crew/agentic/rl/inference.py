"""Inference helper: load a trained policy and score the live state.

Used by:

* The CrewAI tool ``rl_policy_recommendation`` (see :mod:`trading_crew.tools`)
  which surfaces the policy's recommendation to the Market Analyst and
  Trader.  The LLM still owns the final decision — the policy is an
  *advisor*.
* The web UI's "preview" panel that lets you see what the promoted
  policy would do today before queueing a full crew run.

Why a separate module?  The trainer pulls in ``torch.optim`` and a
~200-line PPO update; inference only needs ``torch.no_grad`` + a forward
pass.  Splitting them keeps the ``CrewAI tool import + import the
inference module`` path lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from trading_crew.agentic.rl.env import ACTION_WEIGHTS, N_ACTIONS
from trading_crew.agentic.rl.networks import ActorCritic
from trading_crew.agentic.rl.state import FEATURE_DIM, FeatureExtractor
from trading_crew.agentic.rl.storage import (
    RL_RUN_DIR,
    get_promoted,
    load_run,
    policy_checkpoint_path,
)


# ---------------------------------------------------------------------------
# Recommendation payload — returned by PolicyClient.recommend()
# ---------------------------------------------------------------------------


@dataclass
class PolicyRecommendation:
    """The policy's view of a single decision.

    Fields:

    * ``best_action_idx`` — argmax over the categorical.
    * ``best_target_weight`` — the corresponding entry from
      :data:`ACTION_WEIGHTS`.
    * ``action_distribution`` — full softmax probabilities (one per
      bucket in :data:`ACTION_WEIGHTS`).  This is what the LLM sees:
      "policy thinks 60% LONG, 25% NEUTRAL, 15% SHORT".
    * ``value_estimate`` — the critic's value estimate at this state.
      Roughly the policy's expected forward return.  Useful as a
      conviction proxy — high |value| means the policy thinks there's
      a strong edge in *some* direction.
    * ``confidence`` — max softmax probability.  A flat distribution
      ≈ "policy is indifferent"; a peaky one ≈ "policy is sure".
    """

    ticker: str
    run_id: str
    asset_class: str
    best_action_idx: int
    best_target_weight: float
    action_distribution: list[float]
    action_weights: list[float]
    value_estimate: float
    confidence: float
    as_of: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "run_id": self.run_id,
            "asset_class": self.asset_class,
            "best_action_idx": self.best_action_idx,
            "best_target_weight": self.best_target_weight,
            "action_distribution": self.action_distribution,
            "action_weights": self.action_weights,
            "value_estimate": self.value_estimate,
            "confidence": self.confidence,
            "as_of": self.as_of,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# PolicyClient
# ---------------------------------------------------------------------------


class PolicyClient:
    """Wrapper around a loaded torch policy + the feature extractor.

    Holds the model in memory so repeated ``recommend`` calls reuse the
    same forward graph (matters when the analyst calls the tool 20
    times in a single crew run).
    """

    def __init__(
        self,
        model: ActorCritic,
        ticker: str,
        run_id: str,
        asset_class: str = "stock",
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.ticker = ticker.upper()
        self.run_id = run_id
        self.asset_class = asset_class
        self.device = torch.device(device)
        self.feature_extractor = FeatureExtractor()
        self.model.to(self.device)
        self.model.eval()

    # -- inference --------------------------------------------------------

    @torch.no_grad()
    def recommend(
        self,
        ohlcv_history: pd.DataFrame,
        *,
        position_weight: float = 0.0,
        unrealised_pct: float = 0.0,
        bars_in_trade: int = 0,
        as_of: Optional[str] = None,
    ) -> PolicyRecommendation:
        """Score ``ohlcv_history`` and produce a structured recommendation.

        ``ohlcv_history`` must contain ≥ ``warmup_bars()`` rows so the
        feature extractor can produce a vector.  If you pass fewer the
        extractor raises — we do *not* silently pad with zeros (that
        would corrupt the policy's input distribution).
        """
        if len(ohlcv_history) < self.feature_extractor.warmup_bars():
            raise ValueError(
                f"Need >= {self.feature_extractor.warmup_bars()} bars to "
                f"score, got {len(ohlcv_history)}."
            )
        obs = self.feature_extractor.extract(
            ohlcv_history,
            position_weight=position_weight,
            unrealised_pct=unrealised_pct,
            bars_in_trade=bars_in_trade,
        )
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        probs = self.model.action_probs(obs_t).cpu().numpy().reshape(-1)
        _, value = self.model(obs_t)
        value_f = float(value.item())

        best_idx = int(np.argmax(probs))
        best_w = ACTION_WEIGHTS[best_idx]
        conf = float(np.max(probs))
        as_of_str = as_of or datetime.utcnow().isoformat()

        # Build a short summary line for the analyst's tool output.
        direction = "LONG" if best_w > 0 else ("SHORT" if best_w < 0 else "FLAT")
        summary = (
            f"Policy recommends {direction} {abs(best_w):.0%} "
            f"(confidence {conf:.0%}, value-estimate {value_f:+.3f})."
        )

        return PolicyRecommendation(
            ticker=self.ticker,
            run_id=self.run_id,
            asset_class=self.asset_class,
            best_action_idx=best_idx,
            best_target_weight=float(best_w),
            action_distribution=probs.tolist(),
            action_weights=list(ACTION_WEIGHTS),
            value_estimate=value_f,
            confidence=conf,
            as_of=as_of_str,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Factory: resolve "ticker -> promoted policy" to a PolicyClient
# ---------------------------------------------------------------------------


def load_policy(ticker: str, run_id: Optional[str] = None, device: str = "cpu") -> Optional[PolicyClient]:
    """Load the promoted policy for ``ticker`` (or a specific run).

    Returns ``None`` if no policy has been promoted *and* no explicit
    ``run_id`` was supplied.  Refusing to silently fall back here is
    deliberate — we don't want the analyst to think "no policy" when the
    user merely forgot to promote.
    """
    if run_id is None:
        promoted = get_promoted(ticker)
        if promoted is None:
            return None
        run_id = promoted["run_id"]
        asset_class = promoted.get("asset_class", "stock")
    else:
        rec = load_run(ticker, run_id)
        if rec is None:
            return None
        asset_class = rec.asset_class

    ckpt = policy_checkpoint_path(ticker, run_id)
    if not ckpt.exists():
        return None

    # Resolve hidden_dim / n_hidden_layers from the saved config so the
    # PolicyClient reconstructs the *exact* architecture used during
    # training (otherwise state_dict() load fails with a shape mismatch).
    payload = torch.load(str(ckpt), map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    model = ActorCritic(
        obs_dim=payload.get("obs_dim", FEATURE_DIM),
        n_actions=payload.get("n_actions", N_ACTIONS),
        hidden_dim=cfg.get("hidden_dim", 64),
        n_hidden_layers=cfg.get("n_hidden_layers", 2),
    )
    model.load_state_dict(payload["model"])
    model.eval()
    return PolicyClient(model, ticker=ticker, run_id=run_id, asset_class=asset_class, device=device)
