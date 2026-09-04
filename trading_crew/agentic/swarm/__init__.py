"""Swarm-intelligence layer for TradingCrew (Firefly / Fintrens concepts).

The two reference platforms centre their pitch on *swarm intelligence* —
the collective behaviour of many simple agents (bees, ants) converging on
good decisions — plus a *probability-edge* discipline and data-driven,
unbiased risk control.  This package folds those ideas into TradingCrew's
existing LLM-free deterministic layer, keeping everything seeded,
reproducible and leakage-free (the same contract as ``agentic.backtest``).

Three submodules:

- ``pso``       — a **Particle Swarm Optimizer** over the sizing / gate
                  parameter space, ranked by Deflated Sharpe.  A swarm of
                  particles explores the space and converges on the robust
                  configuration in far fewer backtests than a brute-force
                  grid.
- ``consensus`` — a **strategy swarm / consensus ensemble**: replay the
                  top-K configurations and aggregate their per-timestamp
                  target weights into one robustness-weighted decision,
                  reporting how tightly the swarm agrees.
- ``crowd``     — a deterministic **crowd-positioning signal**: fuse
                  options / insider / analyst / shareholding crowding into
                  a single score used for an optional "fade-the-crowd"
                  sizing tilt (probability-edge).
"""

from .pso import (
    PSOConfig,
    PSOResult,
    SwarmPoint,
    particle_swarm_optimize,
    run_swarm_search,
)
from .consensus import (
    ConsensusConfig,
    ConsensusResult,
    run_consensus_backtest,
)
from .crowd import (
    CrowdConfig,
    CrowdInputs,
    CrowdSignal,
    compute_crowd_signal,
    crowd_size_multiplier,
)

__all__ = [
    "PSOConfig",
    "PSOResult",
    "SwarmPoint",
    "particle_swarm_optimize",
    "run_swarm_search",
    "ConsensusConfig",
    "ConsensusResult",
    "run_consensus_backtest",
    "CrowdConfig",
    "CrowdInputs",
    "CrowdSignal",
    "compute_crowd_signal",
    "crowd_size_multiplier",
]
