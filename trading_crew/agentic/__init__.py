"""Agentic-trading roadmap (M1-M7) — vendored from the TradingAgents project.

These modules implement the deterministic Layer-A pieces from the agentic-
trading survey (arxiv:2605.19337):

- ``portfolio``  — audit-grade ``PortfolioState`` + multi-ticker allocator (M1, M7)
- ``execution``  — ``ActionProposal`` contract + cost model + fill simulator (M1, M2)
- ``memory``     — episodic memory with outcome embargo + semantic KB (M3)
- ``risk``       — VaR / sizing / hard risk gates (M5)
- ``backtest``   — walk-forward harness + metrics + run manifest (M6)

The package is intentionally LangChain-free: the LLM-driven pieces (M1 Action
Compiler, M4 Reflective Critic + Cascaded Controller) are implemented as
CrewAI-native tasks / runner extensions in ``trading_crew.crew`` and
``web.backend.runner`` so the CrewAI process stays the single LLM driver.
"""

from . import backtest, execution, memory, portfolio, risk

__all__ = ["backtest", "execution", "memory", "portfolio", "risk"]
