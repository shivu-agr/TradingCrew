"""Horizon vocabulary — SHORT / MEDIUM / LONG.

The Portfolio Manager, trader, backtest engine, and outcome-resolution
loop all need a single, canonical way to talk about *how long we're
holding the trade*.  Historically the codebase used ``horizon_days``
alone, defaulted to 90–120, and any horizon > ~252d fell off the
prompt's cognitive map.  This module gives us a compact three-tier
alphabet that survives across the schema, the backtest, the prompts,
the UI, and the RLMF training loop.

Rationale for three tiers (not more, not fewer)
-----------------------------------------------
* **SHORT (≤ 60 trading days, ~≤ 3 calendar months)** — swing trades,
  earnings plays, near-term catalyst bets.  This is what the trader's
  short panel already covered; we keep it because a few strategies
  (binary FDA event, earnings surprise, buyback close) genuinely need
  a sub-quarter window.
* **MEDIUM (61–252 trading days, ~3–12 calendar months)** — the "few
  months" bucket in the user's spec.  This is where most retail
  buy-and-hold-for-a-quarter positions live, and where a Ind-AS
  quarterly-print thesis has time to play out.  The desk's *default*.
* **LONG (253+ trading days, i.e. 1–10 calendar years)** — the "5–10
  years" bucket in the user's spec.  A long-term thesis on a
  compounder (revenue CAGR, ROCE compounding, promoter buy-ins) —
  the analyst basically becomes a fundamental value investor here.
  We cap the outcome-resolution horizon at ~5 years (1260 trading
  days) because holding an episode as PENDING for a decade would
  make the reflection loop unusable; the *thesis* however is written
  for a 5–10 year window (see ``horizon_days_to_horizon`` for the
  mapping).

Legacy compat
-------------
Old ``RunRecord`` JSONs only carry a numeric ``horizon_days``.  The
Pydantic validator on ``PortfolioDecision.horizon`` falls back to
``horizon_days_to_horizon(horizon_days)`` when the field is missing so
we never need an on-disk migration.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class Horizon(str, Enum):
    """Coarse investment-horizon classification.

    Kept as a ``str`` Enum so Pydantic can serialise/deserialise it
    transparently and downstream consumers (JSON logs, the UI, the
    RLMF dataset) see a plain string.
    """

    SHORT = "SHORT"
    """Sub-quarter trades — earnings, buyback closes, technical setups.

    Days: [5, 60].  Sizing is typically smaller (event risk is
    binary), and backtest hit-rates are the primary edge check.
    """

    MEDIUM = "MEDIUM"
    """A few months — the desk's *default* horizon.

    Days: [61, 252].  This is where the Ind-AS quarterly cadence,
    peer-relative valuation, and multi-quarter trend evidence
    dominate.  The PM biases toward this tier unless the analyst
    evidence specifically pulls short or long.
    """

    LONG = "LONG"
    """1–10 years — long-term compounder / value thesis.

    Days: [253, ~2520].  The evidence surface is:
    - Multi-year revenue / earnings CAGR
    - ROCE / ROE trend (up = compounder, down = value trap)
    - Debt-to-equity trajectory
    - Promoter shareholding stability (Indian names)
    - Sector-level secular themes (EVs, AI capex, energy transition)

    The backtest panel covers 2y / 5y windows to bound the risk of
    the thesis.  Because outcome-resolution can't wait 10 years,
    the outcome_ts is clamped to *5 years* (1260 trading days) and
    the RLMF loop uses the 2y realised return as the reward proxy.
    """


# -----------------------------------------------------------------------------
# Day-count reference tables
# -----------------------------------------------------------------------------

# The canonical horizons the backtest panel and the LLM prompts consume.
# We keep these as *ordered* tuples so downstream code (esp. the backtest
# panel and the frontend legend) can display them in the same order.
SHORT_HORIZON_DAYS: tuple[int, ...] = (10, 21, 60)
"""Trading-day horizons expressed by the SHORT tier.

10d ≈ 2 weeks (technical breakout / catalyst run-up),
21d ≈ 1 month (typical earnings hold),
60d ≈ 3 months (sub-quarter catalyst window)."""

MEDIUM_HORIZON_DAYS: tuple[int, ...] = (126, 189, 252)
"""6 months / 9 months / 1 year — the "few months" bucket."""

LONG_HORIZON_DAYS: tuple[int, ...] = (504, 1260, 2520)
"""2 years / 5 years / 10 years — the "long-term thesis" bucket.

The 2520d = 10y panel is only *simulated* against the trailing 15
years of history when available; when history is shorter we drop it
and note the omission so the LLM does not treat the missing panel
as neutral evidence."""

ALL_HORIZON_DAYS: tuple[int, ...] = (
    SHORT_HORIZON_DAYS + MEDIUM_HORIZON_DAYS + LONG_HORIZON_DAYS
)
"""Convenience: every canonical panel day-count in one flat tuple."""


MAX_ACTIONABLE_HORIZON_DAYS: int = 1260
"""Upper bound (in trading days) on the outcome-accounting horizon.

LONG-tier decisions carry a 5-10 year *narrative* view (see
``PortfolioDecision.long_term_view``), but the outcome-resolution loop
and the M6 backtester need a bounded review window — holding an
episode as PENDING for a decade would make the reflection loop
unusable in practice.  We cap at 5 trading years (1260d).

Consumed by two places that must agree:
  * ``trading_crew.agentic.bridge`` — clamps ``ActionProposal.horizon_days``
    when translating from ``PortfolioDecision``.
  * ``trading_crew.agentic.execution.contracts.ActionProposal`` —
    imposes the ``le=`` schema constraint on ``horizon_days`` itself.
"""


# -----------------------------------------------------------------------------
# Classification helpers
# -----------------------------------------------------------------------------


def horizon_days_to_horizon(days: Optional[int]) -> Horizon:
    """Bucket a raw ``horizon_days`` value into a ``Horizon`` tier.

    Boundaries mirror the docstring above (60 / 252 trading-day
    breakpoints).  ``None`` / non-positive values fall into MEDIUM
    because that is the desk's default and choosing SHORT/LONG on
    missing data would be biased.

    Deliberately *inclusive* on the upper boundary so a trader who
    writes ``horizon_days=60`` still lands in SHORT — the prompt
    tells the trader that "60d" is the top of the short-term bucket
    and matches the M2 simulator convention.
    """
    if days is None or days <= 0:
        return Horizon.MEDIUM
    if days <= 60:
        return Horizon.SHORT
    if days <= 252:
        return Horizon.MEDIUM
    return Horizon.LONG


def horizon_representative_days(h: Horizon) -> int:
    """Return the *representative* day-count for a horizon tier.

    Used when the LLM emitted only the tier (e.g. ``horizon=LONG``)
    but downstream code (bridge → outcome_ts, backtest → panel) needs
    a concrete number.  We pick the *middle* of the panel:
    SHORT ⇒ 21d (1 month), MEDIUM ⇒ 189d (9 months), LONG ⇒ 1260d
    (5 years).  These are intentionally middle-of-tier so a
    directional mistake at the boundary doesn't flip conviction.
    """
    if h == Horizon.SHORT:
        return 21
    if h == Horizon.LONG:
        return 1260
    return 189


def horizon_panel_days(h: Horizon) -> tuple[int, ...]:
    """Return the canonical backtest panel for a horizon tier."""
    if h == Horizon.SHORT:
        return SHORT_HORIZON_DAYS
    if h == Horizon.LONG:
        return LONG_HORIZON_DAYS
    return MEDIUM_HORIZON_DAYS


def horizon_human_label(h: Horizon) -> str:
    """Frontend-friendly label, e.g. ``"Medium term (3–12 months)"``.

    Not used by the LLM (which sees the enum name directly), but the
    web UI + logs render this so a human operator can see "5–10 years"
    without decoding SHORT/MEDIUM/LONG mentally.
    """
    if h == Horizon.SHORT:
        return "Short term (up to 3 months)"
    if h == Horizon.LONG:
        return "Long term (1–10 years)"
    return "Medium term (3–12 months)"


def normalise_horizon(value: object) -> Optional[Horizon]:
    """Coerce a free-form value to a ``Horizon``.

    Accepts:
      * ``Horizon`` instances (returned unchanged),
      * bare strings ("short" / "MEDIUM" / "long term" / ...),
      * ``None`` (returns ``None`` so callers can distinguish
        "not-provided" from "explicitly-medium").

    Anything else returns ``None`` so a downstream ``Optional[Horizon]``
    validator can decide whether to fall back to
    ``horizon_days_to_horizon``.
    """
    if value is None:
        return None
    if isinstance(value, Horizon):
        return value
    if not isinstance(value, str):
        return None
    up = value.strip().upper()
    # Tolerate common variants — "short-term", "medium term", "long
    # holding" — so the LLM's freeform emit doesn't fail validation
    # on a trailing word.
    for h in (Horizon.SHORT, Horizon.MEDIUM, Horizon.LONG):
        if up.startswith(h.value):
            return h
    return None
