"""Pydantic schema for the futures trader's structured decision.

This intentionally mirrors ``trading_crew.schemas.PortfolioDecision`` so
that downstream code (bridge, episodic memory, run record) can treat
both crews uniformly — but adds the futures-specific fields that don't
make sense for cash equities:

- ``contract_month``: the specific delivery month being traded
  (e.g. ``"2026-08"`` for August 2026 WTI).  Carries the position roll
  responsibility — the operator must close or roll before this month.
- ``curve_view``: typed assessment of the term-structure (CONTANGO,
  BACKWARDATION, FLAT) at decision time, so the run record captures
  *why* a roll cost was tolerated.
- ``roll_yield_pct_annualised``: expected drag (negative for contango)
  or lift (positive for backwardation) from rolling the front month
  through the holding period.
- ``contract_size``: notional multiplier — 1000 bbl for crude, 100
  troy oz for gold, etc.  Required so the M2 simulator computes the
  right exposure when target_weight is converted to contracts.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


CurveView = Literal["CONTANGO", "BACKWARDATION", "FLAT"]


class FuturesDecision(BaseModel):
    """Structured final decision emitted by the futures Trader.

    Field defaults are deliberately absent for the core action/sizing
    fields — the LLM must commit to each.  Optional advisory fields
    (curve_view, roll_yield_pct_annualised) default to neutral values
    so an LLM that can't form a view on them gets a sane fallback.
    """

    action: Literal["LONG", "NEUTRAL", "SHORT"]
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Sizing is expressed as % of risk budget (≈ equity NAV at risk),
    # not % of notional, because futures notional is leveraged and
    # noisier as a sizing input.  M5's vol-target / CVaR clamps still
    # apply downstream of this number.
    size_pct_of_book: float = Field(..., ge=0.0, le=100.0)

    entry_price: float
    stop_loss: float
    target_price: float
    horizon_days: int = Field(..., ge=1)
    expected_return_pct: float

    # Futures-specific context. ``contract_month`` is ISO YYYY-MM so
    # the bridge can compute days-to-expiry against ``decision_ts``.
    contract_month: str = Field(
        ...,
        description=(
            "Delivery month of the contract being traded, ISO YYYY-MM "
            "(e.g. '2026-08' for August 2026 WTI). Determines when the "
            "position must be closed or rolled."
        ),
    )
    contract_size: float = Field(
        ..., gt=0.0,
        description=(
            "Notional multiplier per contract: 1000 (bbl) for crude, "
            "100 (troy oz) for gold, 5000 (bu) for grains, etc."
        ),
    )
    curve_view: CurveView = Field(
        ...,
        description="Term-structure assessment at decision time.",
    )
    roll_yield_pct_annualised: float = Field(
        0.0,
        description=(
            "Expected annualised drag/lift from rolling the front-month "
            "through the holding period. Negative = contango drag."
        ),
    )

    rationale: str = Field(..., min_length=20)
    key_drivers: List[str]
    key_risks: List[str]
    falsifiers: List[str]
    geopolitical_flags: List[str] = []
    compliance_status: Literal["CLEAR", "FLAGGED", "BLOCKED"] = "CLEAR"

    # Provenance: list of source identifiers cited inline in ``rationale``
    # as ``[source: <identifier>]`` tags.  Mirrors the equity
    # PortfolioDecision.sources field — the Reflective Critic uses this
    # list to verify every quantitative claim is traceable to a real fetch
    # (yfinance OHLCV, CFTC COT, Tavily news URL, etc.) rather than
    # hallucinated.  Claims without a matching tag are marked UNSUPPORTED
    # and the decision is REVISED or ABSTAINED.
    sources: List[str] = Field(
        default_factory=list,
        description=(
            "Distinct source identifiers cited inline in the rationale via "
            "[source: <id>] tags. Populated by the Portfolio Manager from "
            "the analyst reports' provenance lines."
        ),
    )
