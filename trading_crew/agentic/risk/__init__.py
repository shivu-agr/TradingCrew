"""Quantitative risk + position sizing (paper §7).

Three submodules:

- ``var``     — Value-at-Risk / Conditional VaR estimators (historical +
                parametric) over a configurable lookback window.
- ``sizing``  — Fractional Kelly + vol-target + CVaR clamp.  Converts a
                proposal's expected-return / conviction into a *deterministic*
                position size.
- ``gates``   — Pre-trade hard risk gates: concentration, leverage,
                drawdown stop, kill-switch.  Refuses any order that breaches
                a hard limit, regardless of LLM intent.

The layer's contract: it consumes the LLM-emitted ``ActionProposal`` from
M1 and produces either an executable ``Order`` (subject to M2's simulator)
or a structured rejection.  No LLM judgement is allowed past this point.
"""

from .var import (
    VarConfig,
    VarResult,
    compute_historical_var,
    compute_parametric_var,
)
from .sizing import (
    SizingConfig,
    SizingResult,
    compute_size,
    debate_to_risk_mult,
)
from .gates import (
    GateConfig,
    GateResult,
    RiskGate,
    run_risk_gates,
)

__all__ = [
    "VarConfig",
    "VarResult",
    "compute_historical_var",
    "compute_parametric_var",
    "SizingConfig",
    "SizingResult",
    "compute_size",
    "debate_to_risk_mult",
    "GateConfig",
    "GateResult",
    "RiskGate",
    "run_risk_gates",
]
