"""Live signal fusion (Phase 4).

Fuses the four views the desk produces into one guardrailed prediction
signal:

- **LLM decision** — the crew's ``PortfolioDecision`` (side + conviction).
- **L4 RL policy** — the PPO policy's recommended target weight + confidence.
- **L3 tuned expectancy** — the per-trade edge of the tuned M5 config.
- **Crowd positioning** — the swarm/crowd score (used contrarian at extremes).

See ``fusion.py`` for the combiner + guardrails.
"""

from __future__ import annotations

from .alignment import apply_signal_alignment, computed_trend_lean
from .fusion import (
    ExpectancySignal,
    FusedSignal,
    FusionConfig,
    LLMSignal,
    PolicySignal,
    fuse_signals,
    pso_tuned_expectancy,
)

__all__ = [
    "ExpectancySignal",
    "FusedSignal",
    "FusionConfig",
    "LLMSignal",
    "PolicySignal",
    "apply_signal_alignment",
    "computed_trend_lean",
    "fuse_signals",
    "pso_tuned_expectancy",
]
