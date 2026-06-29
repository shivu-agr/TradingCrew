"""L4 Agentic Training — Reinforcement Learning on the M2 simulator.

The RL stack closes the training loop the M1–M7 milestones leave open:

* **M2** gave us a deterministic execution simulator (next-bar fill + costs).
* **M3** records every trade as an episode with realised outcomes.
* **M6** replays past proposals through the deterministic pipeline.
* **L2** writes LLM reflections on resolved episodes.
* **L3** does a one-shot grid search over sizing/risk parameters.
* **L4 (this module)** trains a *parametric policy* end-to-end using PPO,
  where the M2 simulator is the environment, the policy emits target
  weights, and gradient updates flow back through the realised PnL.

The trained policy is then exposed as an **advisory tool** to the LLM
crew — it does not replace the LLM's PortfolioDecision, it augments it
with a calibrated, history-trained prior the analyst + trader can lean
on.

Components::

    state.py        FeatureExtractor — OHLCV -> normalised state vector
    env.py          TradingEnv — Gym-style env wrapping M2 simulator
    networks.py     ActorCritic — small PyTorch MLP shared backbone
    ppo.py          PPOTrainer — clipped surrogate + GAE + entropy bonus
    storage.py      RLRunRecord persistence + checkpoint layout
    inference.py    PolicyClient — load a checkpoint, run inference
"""

from trading_crew.agentic.rl.state import FeatureExtractor, FEATURE_DIM
from trading_crew.agentic.rl.env import (
    ACTION_WEIGHTS,
    N_ACTIONS,
    TradingEnv,
    TradingEnvConfig,
)
from trading_crew.agentic.rl.networks import ActorCritic
from trading_crew.agentic.rl.ppo import PPOConfig, PPOTrainer, TrainingMetrics
# Phase 2D — alternative trainers (gracefully degraded if torch
# isn't available with all the necessary modules).
try:
    from trading_crew.agentic.rl.cql import (
        CQLConfig,
        CQLTrainer,
        CQLTrainingMetrics,
        collect_transitions_from_episodes,
    )
    from trading_crew.agentic.rl.distributional import (
        C51Config,
        C51Network,
        C51Trainer,
        C51TrainingMetrics,
    )
    from trading_crew.agentic.rl.decision_transformer import (
        DTConfig,
        DTTrainingMetrics,
        DecisionTransformer,
        DecisionTransformerTrainer,
        trajectories_from_transitions,
    )
except Exception:  # pragma: no cover — keep PPO usable if extras break.
    CQLConfig = CQLTrainer = CQLTrainingMetrics = None  # type: ignore
    collect_transitions_from_episodes = None  # type: ignore
    C51Config = C51Network = C51Trainer = C51TrainingMetrics = None  # type: ignore
    DTConfig = DTTrainingMetrics = DecisionTransformer = DecisionTransformerTrainer = None  # type: ignore
    trajectories_from_transitions = None  # type: ignore
from trading_crew.agentic.rl.storage import (
    RL_RUN_DIR,
    PROMOTED_LINK_DIR,
    RLRunRecord,
    list_runs,
    list_promoted,
    load_run,
    promote_run,
    save_run,
)
from trading_crew.agentic.rl.inference import PolicyClient, load_policy

__all__ = [
    "FeatureExtractor",
    "FEATURE_DIM",
    "ACTION_WEIGHTS",
    "N_ACTIONS",
    "TradingEnv",
    "TradingEnvConfig",
    "ActorCritic",
    "PPOConfig",
    "PPOTrainer",
    "TrainingMetrics",
    "RL_RUN_DIR",
    "PROMOTED_LINK_DIR",
    "RLRunRecord",
    "list_runs",
    "list_promoted",
    "load_run",
    "promote_run",
    "save_run",
    "PolicyClient",
    "load_policy",
    "CQLConfig",
    "CQLTrainer",
    "CQLTrainingMetrics",
    "collect_transitions_from_episodes",
    "C51Config",
    "C51Network",
    "C51Trainer",
    "C51TrainingMetrics",
    "DTConfig",
    "DTTrainingMetrics",
    "DecisionTransformer",
    "DecisionTransformerTrainer",
    "trajectories_from_transitions",
]
