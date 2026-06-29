"""Execution layer — turns LLM intent into deterministic, costed market actions.

This package implements paper §6 (Action and Execution Architecture):

- ``contracts``     — typed ``ActionProposal`` from the agent loop (M1).
- ``cost``          — explicit fee / spread / slippage / impact model (M2).
- ``simulator``     — next-bar fill model with rejection / partial fill (M2).
- ``pipeline``      — ``ActionProposal`` -> ``Sizer`` -> ``RiskGate`` -> ``Simulator``
                     -> ``Fill`` -> ``PortfolioState`` mutation (M2).

Everything downstream of ``ActionProposal`` is deterministic Python so the
LLM cannot dictate sizing, prices, or fills — only intent.  That separation
is what the survey calls the *Action I/O contract* (§6.1).
"""

from .contracts import (
    ActionProposal,
    ActionSide,
    ConvictionTier,
    OrderTimeInForce,
    SizingBasis,
    ValidityCheck,
)
from .cost import (
    CostModel,
    CostScenario,
    COST_MODEL_LIBRARY,
    cost_sweep,
    get_cost_model,
)
from .simulator import (
    Bar,
    ExecutionSimulator,
    Fill,
    FillStatus,
    Order,
    proposal_to_order,
)

__all__ = [
    # Contracts
    "ActionProposal",
    "ActionSide",
    "ConvictionTier",
    "OrderTimeInForce",
    "SizingBasis",
    "ValidityCheck",
    # Cost model
    "CostModel",
    "CostScenario",
    "COST_MODEL_LIBRARY",
    "cost_sweep",
    "get_cost_model",
    # Simulator
    "Bar",
    "ExecutionSimulator",
    "Fill",
    "FillStatus",
    "Order",
    "proposal_to_order",
]
