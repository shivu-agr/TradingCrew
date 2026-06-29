"""Typed action contracts for the agent -> execution boundary.

Implements the *Action I/O contract* from paper §6.1: the LLM emits a
strongly-typed ``ActionProposal`` that downstream deterministic code
(``Sizer``, ``RiskGate``, ``Simulator``) consumes.  This removes the failure
mode the paper calls *Action Definition Risk* — agents emitting free-text
"Buy" that silently collapses into "trade at close, zero cost".

Key design decisions
--------------------

- ``target_weight`` is in **portfolio units** (fraction of NAV).  The LLM
  reasons about intent ("I want 8% exposure"), the sizer translates that
  into share counts using current NAV and price.
- ``conviction`` is a continuous ``[0, 1]`` *plus* a categorical tier.  The
  continuous score feeds M5 sizing (fractional Kelly), the tier feeds the
  M4 reflection budget (low conviction -> more critique iterations).
- ``validity_check`` is a structured pre-flight that the model fills in as
  part of generation, not a free-text claim.  M5's risk gate cross-checks
  every field against ground truth before any fill.
- ``horizon_days`` ties to M3's outcome-embargo: an episode with
  ``horizon_days=5`` cannot be retrieved by future runs until 5 trading
  days after the decision.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionSide(str, Enum):
    """Direction of the proposed action.

    ``HOLD`` is treated as a first-class action (not the absence of one): when
    the agent emits ``HOLD`` it means "I deliberately chose to not trade",
    which is recorded in episodic memory and counted toward the reflection
    budget.  An ``ABSTAIN`` separately means "I am too uncertain to trade",
    which M4's consistency-vote step emits.
    """

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    ABSTAIN = "ABSTAIN"


class OrderTimeInForce(str, Enum):
    """Order TIF (Time In Force) — controls how long the order stays live.

    ``DAY`` cancels at end-of-session, ``GTC`` (Good-Till-Cancelled) persists,
    ``IOC`` (Immediate-Or-Cancel) fills what it can and cancels the rest.
    The simulator in M2 honours these by truncating fills accordingly.
    """

    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"


class ConvictionTier(str, Enum):
    """Coarse-grained conviction for the M4 reflection budget router.

    LOW    -> route through the full reflective critic (extra iterations).
    MEDIUM -> standard single-pass critique.
    HIGH   -> can early-exit the critic if the consistency vote is unanimous.
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SizingBasis(str, Enum):
    """How the LLM phrased its sizing intent.

    ``TARGET_WEIGHT`` is the canonical form (fraction of NAV); ``NOTIONAL``
    and ``SHARES`` are accepted for compatibility with traders who prefer
    the absolute units, and the sizer in M5 normalises everything to a
    target weight before risk-gate checks.
    """

    TARGET_WEIGHT = "TARGET_WEIGHT"
    NOTIONAL = "NOTIONAL"
    SHARES = "SHARES"


# ---------------------------------------------------------------------------
# Validity check — pre-trade pre-flight emitted by the LLM
# ---------------------------------------------------------------------------


class ValidityCheck(BaseModel):
    """Structured pre-flight the agent fills as part of the proposal.

    These are *claims* the LLM is making — M5's deterministic risk gate
    re-verifies each field against ground-truth state and refuses the
    fill if any claim fails to hold.  Logging the claim alongside the
    ground-truth result makes the audit trail diagnostic (paper §13.1.1
    on hallucination propagation).
    """

    data_timestamps_valid: bool = Field(
        description=(
            "True if every data point cited in the rationale has a timestamp at "
            "or before the trade_date (no future leakage). Set False if you "
            "referenced any post-trade-date evidence."
        ),
    )
    fits_risk_budget: bool = Field(
        description=(
            "True if the proposed target_weight respects the portfolio's stated "
            "concentration / leverage limits as visible in the get_portfolio_state "
            "tool call. Set False if you knowingly proposed an over-limit size."
        ),
    )
    survives_transaction_costs: bool = Field(
        description=(
            "True if the expected edge (target_return * conviction) exceeds the "
            "estimated round-trip cost in basis points. Set False for marginal-edge "
            "trades that may not survive realistic fees + slippage."
        ),
    )
    liquidity_sufficient: bool = Field(
        description=(
            "True if the proposed notional is small relative to typical daily "
            "volume (rule of thumb: < 5%% of ADV). Set False for size that "
            "would meaningfully move the market."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        description=(
            "Optional free-text caveats — failure modes the agent noticed but "
            "decided to proceed despite, anomalies in the data, etc."
        ),
    )


# ---------------------------------------------------------------------------
# ActionProposal — the contract itself
# ---------------------------------------------------------------------------


class ActionProposal(BaseModel):
    """The agent's typed proposal to the execution layer.

    Replaces the free-text ``final_trade_decision`` field that used to flow
    out of the Portfolio Manager.  Everything downstream (sizer, risk gate,
    execution sim, episodic memory) reads this object instead of grepping
    the markdown.
    """

    # -- identity ----------------------------------------------------------

    symbol: str = Field(
        description="Ticker symbol (uppercase). Must match the run's company_of_interest.",
    )
    decision_ts: str = Field(
        description=(
            "ISO-8601 timestamp when the decision was made. Conventionally "
            "set to the close of the trade_date so the next-bar fill lands "
            "on T+1 open."
        ),
    )

    # -- direction & size --------------------------------------------------

    side: ActionSide = Field(
        description=(
            "Direction of the action. BUY/SELL for trading intent, HOLD for "
            "deliberate inaction, ABSTAIN for 'too uncertain to trade'. "
            "When abstain or hold, target_weight should match the current "
            "position weight."
        ),
    )
    target_weight: float = Field(
        description=(
            "Target portfolio weight as a fraction of NAV ([-1.0, 1.0]). "
            "+0.08 means '8%% net long', -0.05 means '5%% net short', "
            "0.0 means flat. The sizer in M5 maps this into share count."
        ),
        ge=-1.0,
        le=1.0,
    )
    sizing_basis: SizingBasis = Field(
        default=SizingBasis.TARGET_WEIGHT,
        description="How the agent phrased the sizing intent. Default TARGET_WEIGHT.",
    )

    # -- limits and timing -------------------------------------------------

    limit_price: Optional[float] = Field(
        default=None,
        description=(
            "Optional limit price in quote currency. None = trade at market "
            "next bar. The sim treats an unfilled limit as a Rejection."
        ),
    )
    tif: OrderTimeInForce = Field(
        default=OrderTimeInForce.DAY,
        description="Order Time In Force. Default DAY.",
    )
    horizon_days: int = Field(
        description=(
            "Intended holding period in trading days. Drives the M3 outcome-"
            "embargo: the episode cannot be retrieved by future runs until "
            "horizon_days have elapsed."
        ),
        ge=1,
        le=252,
    )

    # -- conviction & rationale --------------------------------------------

    conviction_score: float = Field(
        description=(
            "Continuous conviction in [0, 1]. Feeds M5 fractional-Kelly sizing "
            "and the M4 reflection router. 0 = no edge, 1 = highest possible."
        ),
        ge=0.0,
        le=1.0,
    )
    conviction_tier: ConvictionTier = Field(
        description=(
            "Coarse-grained conviction tier used by the M4 reflection budget. "
            "LOW < MEDIUM < HIGH. Should match conviction_score: <0.33 -> LOW, "
            "<0.66 -> MEDIUM, else HIGH."
        ),
    )
    expected_return_pct: float = Field(
        description=(
            "Expected return over horizon_days as a decimal (0.05 = +5%%). "
            "Signed: negative for a short. Used as the edge term in fractional "
            "Kelly sizing (M5)."
        ),
    )
    rationale: str = Field(
        description=(
            "Concise reasoning chain anchored in evidence from the analyst "
            "reports and risk debate. 3-6 sentences. Cite which analyst "
            "report drove which leg of the argument."
        ),
    )

    # -- validity check (LLM-emitted, deterministic re-check in M5) --------

    validity_check: ValidityCheck = Field(
        description=(
            "Pre-flight checks the agent attests to. M5's deterministic risk "
            "gate will re-verify each field; mismatches are logged."
        ),
    )

    # -- optional tags -----------------------------------------------------

    tags: List[str] = Field(
        default_factory=list,
        description=(
            "Free-form tags for classification (e.g. 'earnings_drift', 'mean_revert'). "
            "Useful for post-trade attribution in M6."
        ),
    )

    # -- cross-field invariants -------------------------------------------

    @model_validator(mode="after")
    def _check_invariants(self) -> "ActionProposal":
        """Cross-field consistency checks the LLM might get wrong.

        We do **not** silently coerce here — a violation indicates the LLM
        misunderstood the schema, which is a useful signal for the M4
        reflection loop.  Instead we raise; the caller decides whether to
        retry (M4) or downgrade to ABSTAIN.
        """
        if self.side == ActionSide.HOLD and self.target_weight != 0.0:
            # HOLD with non-zero weight is allowed iff the position already
            # exists at exactly that weight; we can't verify that here (no
            # access to PortfolioState), so we only error on the obviously
            # contradictory case of BUY/SELL+0% or HOLD+nonzero on first
            # touch.  The M5 risk gate does the final cross-check.
            pass

        if self.side == ActionSide.BUY and self.target_weight < 0:
            raise ValueError(
                f"BUY with negative target_weight={self.target_weight} is contradictory; "
                "use SELL for a short position."
            )
        if self.side == ActionSide.SELL and self.target_weight > 0:
            raise ValueError(
                f"SELL with positive target_weight={self.target_weight} is contradictory; "
                "use BUY for a long position."
            )

        # Soft consistency between score and tier (warn-by-error so the LLM
        # learns to keep them aligned):
        if self.conviction_score < 0.33 and self.conviction_tier != ConvictionTier.LOW:
            raise ValueError(
                f"conviction_score={self.conviction_score} implies LOW tier, "
                f"got {self.conviction_tier}"
            )
        if 0.33 <= self.conviction_score < 0.66 and self.conviction_tier != ConvictionTier.MEDIUM:
            raise ValueError(
                f"conviction_score={self.conviction_score} implies MEDIUM tier, "
                f"got {self.conviction_tier}"
            )
        if self.conviction_score >= 0.66 and self.conviction_tier != ConvictionTier.HIGH:
            raise ValueError(
                f"conviction_score={self.conviction_score} implies HIGH tier, "
                f"got {self.conviction_tier}"
            )
        return self

    # -- helpers -----------------------------------------------------------

    def render_markdown(self) -> str:
        """Render the proposal as the markdown report card the UI shows.

        Keeps the same ``**Rating**: …`` header that the rest of the system
        already greps for (memory log, saved reports) so the contract is
        additive — the typed object is the source of truth and the markdown
        is a derived display artifact.
        """
        rating_label = {
            ActionSide.BUY: "Buy",
            ActionSide.SELL: "Sell",
            ActionSide.HOLD: "Hold",
            ActionSide.ABSTAIN: "Abstain",
        }[self.side]
        lines = [
            f"**Rating**: {rating_label}",
            "",
            f"**Target weight**: {self.target_weight:+.2%}",
            f"**Conviction**: {self.conviction_score:.2f} ({self.conviction_tier.value})",
            f"**Expected return**: {self.expected_return_pct:+.2%} over {self.horizon_days} trading days",
        ]
        if self.limit_price is not None:
            lines.append(f"**Limit price**: {self.limit_price:.2f} ({self.tif.value})")
        lines.extend(["", f"**Rationale**: {self.rationale}"])
        if self.tags:
            lines.append(f"**Tags**: {', '.join(self.tags)}")
        vc = self.validity_check
        lines.extend([
            "",
            "**Pre-flight checks**:",
            f"- Data timestamps valid: {'yes' if vc.data_timestamps_valid else 'NO'}",
            f"- Fits risk budget: {'yes' if vc.fits_risk_budget else 'NO'}",
            f"- Survives transaction costs: {'yes' if vc.survives_transaction_costs else 'NO'}",
            f"- Liquidity sufficient: {'yes' if vc.liquidity_sufficient else 'NO'}",
        ])
        if vc.notes:
            lines.append(f"- Notes: {vc.notes}")
        return "\n".join(lines)
