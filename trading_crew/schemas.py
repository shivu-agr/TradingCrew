"""Pydantic schema for the Portfolio Manager's structured decision.

The PM is forced to emit a JSON object matching this schema via
``output_pydantic=PortfolioDecision`` on its task, so the pipeline ends with
a typed artefact a downstream system can persist or feed into a position-
keeping engine.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class PortfolioDecision(BaseModel):
    """Structured final decision emitted by the Portfolio Manager."""

    action: Literal["OVERWEIGHT", "NEUTRAL", "UNDERWEIGHT"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    size_pct_of_book: float = Field(..., ge=0.0, le=100.0)
    entry_price: float
    stop_loss: float
    target_price: float
    horizon_days: int
    expected_return_pct: float
    rationale: str
    key_drivers: List[str]
    key_risks: List[str]
    falsifiers: List[str]
    geopolitical_flags: List[str]
    compliance_status: Literal["CLEAR", "FLAGGED", "BLOCKED"]

    # Provenance: list of source identifiers cited inline in ``rationale``
    # as ``[source: <identifier>]`` tags.  Each entry is a free-form string
    # produced by the tool layer (e.g. "yfinance OHLCV NTNX · retrieved
    # 2026-06-12T14:32Z" or "Tavily News Search · q='NTNX earnings'").
    # The Reflective Critic uses this list to verify every quantitative
    # claim is traceable to a real fetch — claims without a matching tag
    # are marked UNSUPPORTED and the decision is REVISED or ABSTAINED.
    sources: List[str] = Field(
        default_factory=list,
        description=(
            "Distinct source identifiers cited inline in the rationale via "
            "[source: <id>] tags. Populated by the Portfolio Manager from "
            "the analyst reports' provenance lines."
        ),
    )
