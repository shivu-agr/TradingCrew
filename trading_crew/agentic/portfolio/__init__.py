"""Portfolio-level state and accounting.

This package implements the audit-grade ``Layer A`` from the agentic-trading
survey (Xia et al., 2026, §4.1 "Working Memory: Deterministic State Store"):
ground-truth positions, cash, P&L, and risk metrics that are *read-only* to the
LLM and updated only by deterministic code (fills from the execution layer,
mark-to-market from the data layer).

The LLM may *read* portfolio state through explicit tool calls, but it cannot
mutate it directly through generated text.  That separation is what makes the
state auditable: every change has a corresponding ``Fill`` event with a
timestamp and a snapshot hash.
"""

from .state import (
    Position,
    PortfolioState,
    PortfolioStateStore,
    load_portfolio_state,
    save_portfolio_state,
)
from .allocator import (
    AllocationMethod,
    AllocationResult,
    AllocatorConfig,
    allocate,
)

__all__ = [
    "Position",
    "PortfolioState",
    "PortfolioStateStore",
    "load_portfolio_state",
    "save_portfolio_state",
    "AllocationMethod",
    "AllocationResult",
    "AllocatorConfig",
    "allocate",
]
