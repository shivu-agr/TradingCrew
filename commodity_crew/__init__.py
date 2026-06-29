"""Commodity research crew — sibling of ``trading_crew`` for futures markets.

Why a separate package
----------------------
The deterministic M1-M7 backbone (sizing / risk gates / execution
simulator / episodic memory / walk-forward backtest / allocator) lives in
``trading_crew.agentic`` and is genuinely asset-class-agnostic — it works
on any symbol with OHLCV data.  What's commodity-specific is the LLM
crew composition (8 agents instead of 18, different specialties), the
tool surface (futures curve, CFTC COT, seasonality), and the schemas
(``FuturesDecision`` carries a ``contract_month`` and roll context that
``PortfolioDecision`` doesn't).

By isolating those in ``commodity_crew`` we:
- avoid bloating ``trading_crew``'s prompts with futures jargon,
- keep stock and commodity behaviour independently versionable,
- share 100% of the deterministic pipeline via ``trading_crew.agentic``.
"""

from .crew import CommodityCrew, get_agent_catalog
from .schemas import FuturesDecision

__all__ = ["CommodityCrew", "FuturesDecision", "get_agent_catalog"]
