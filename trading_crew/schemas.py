"""Pydantic schema for the Portfolio Manager's structured decision.

The PM is forced to emit a JSON object matching this schema via
``output_pydantic=PortfolioDecision`` on its task, so the pipeline ends with
a typed artefact a downstream system can persist or feed into a position-
keeping engine.

Decision tier (5-way, retail-friendly)
======================================
The ``action`` field takes a 5-tier label that mirrors the equity-research
convention retail investors are used to reading:

* ``STRONG_BUY``  — high-conviction long.  Typical shape: size >= ~2 %
                    of book AND confidence >= ~0.7 AND at least one of
                    the mid-/long-term backtest panels shows positive
                    expectancy with payoff >= 1.5.
* ``BUY``         — standard long.  Directional thesis is supported but
                    conviction or edge does not warrant a "strong" tier
                    call.  Size is > 0, usually < ~2 % of book.
* ``HOLD``        — sit out.  ``size_pct_of_book`` MUST be 0.  Chosen
                    when directional evidence is genuinely mixed or the
                    Compliance Officer BLOCKED the trade.
* ``SELL``        — standard short (mirror of BUY).
* ``STRONG_SELL`` — high-conviction short (mirror of STRONG_BUY).

Backwards compatibility
-----------------------
Earlier runs persisted actions as ``OVERWEIGHT`` / ``NEUTRAL`` /
``UNDERWEIGHT``.  A ``@field_validator`` coerces the legacy triplet into
the new 5-tier space so old ``RunRecord`` JSON files still load and the
UI still renders them.  The mapping is deliberately conservative —
OVERWEIGHT/UNDERWEIGHT collapse to plain BUY/SELL (not STRONG_*) because
the caller has no way to know the historical conviction level.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from trading_crew.horizons import (
    Horizon,
    horizon_days_to_horizon,
    normalise_horizon,
)


PortfolioAction = Literal[
    "STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL",
]


# Legacy 3-tier -> new 5-tier coercion table.  Kept module-level (rather
# than nested inside the validator) so runners, migrations, and tests can
# reuse it without re-implementing the mapping.
LEGACY_ACTION_MAPPING: dict[str, str] = {
    "OVERWEIGHT": "BUY",
    "UNDERWEIGHT": "SELL",
    "NEUTRAL": "HOLD",
}


def normalise_action(value: str) -> str:
    """Return the canonical 5-tier action string for ``value``.

    Accepts either the current tier ("BUY", "STRONG_BUY", ...) or the
    legacy tier ("OVERWEIGHT", "NEUTRAL", "UNDERWEIGHT") and returns the
    5-tier equivalent.  Unknown strings are returned uppercase and
    stripped so Pydantic's Literal validator can produce a clean error.
    """
    if not isinstance(value, str):
        return value  # type: ignore[return-value]
    key = value.strip().upper()
    return LEGACY_ACTION_MAPPING.get(key, key)


class PortfolioDecision(BaseModel):
    """Structured final decision emitted by the Portfolio Manager."""

    action: PortfolioAction
    confidence: float = Field(..., ge=0.0, le=1.0)
    size_pct_of_book: float = Field(..., ge=0.0, le=100.0)
    entry_price: float
    stop_loss: float
    target_price: float
    horizon_days: int
    horizon: Horizon = Field(
        default=Horizon.MEDIUM,
        description=(
            "Primary investment horizon tier: SHORT (≤60 trading days), "
            "MEDIUM (61-252 days = a few months), LONG (253+ days, "
            "up to 10 years).  The desk's default is MEDIUM; SHORT is "
            "reserved for genuinely near-term catalyst plays and LONG "
            "for multi-year compounder / value theses.  When the LLM "
            "omits this field it is auto-derived from horizon_days for "
            "backwards compatibility."
        ),
    )
    expected_return_pct: float
    rationale: str
    key_drivers: List[str]
    key_risks: List[str]
    falsifiers: List[str]
    geopolitical_flags: List[str]
    compliance_status: Literal["CLEAR", "FLAGGED", "BLOCKED"]

    # Long-term view — required on EVERY decision.
    #
    # For horizon == LONG this is the full multi-year narrative (2-5
    # sentences on revenue / EPS CAGR, ROCE trend, balance-sheet
    # trajectory, and one falsifier).  For SHORT / MEDIUM decisions
    # this is a shorter 1-2 sentence "long-term context" line:
    # "Multi-year compounding thesis is intact / broken / not
    # applicable, based on <cite of get_long_term_thesis or long
    # backtest row>".  This is what makes the LONG panel evidence
    # visible in the final report even when the desk chose MEDIUM —
    # historically the trader reasoned over the LONG row but that
    # context was silently dropped by the time the PM's JSON was
    # emitted.
    #
    # ``long_term_thesis`` is accepted as an alias (see the
    # ``_alias_long_term_thesis`` validator) so pre-existing
    # RunRecord JSONs and any in-flight LLM outputs that still emit
    # the old key round-trip cleanly.
    long_term_view: str = Field(
        default="",
        description=(
            "Long-term view on the ticker — required on every "
            "decision. For horizon == LONG: 2-5 sentence narrative "
            "covering multi-year CAGR / ROCE / debt / falsifier. For "
            "horizon in {SHORT, MEDIUM}: 1-2 sentence 'the multi-year "
            "compounding thesis is intact/broken/not applicable' "
            "line citing get_long_term_thesis output or the LONG "
            "backtest row.  Never leave this empty — if you have no "
            "long-term data, write 'Long-term data unavailable for "
            "this ticker' so the final report is honest about the gap."
        ),
    )

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

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def _alias_long_term_thesis(cls, data: object) -> object:
        """Accept ``long_term_thesis`` as an alias for ``long_term_view``.

        The field was called ``long_term_thesis`` in an earlier revision
        of this schema and was only populated for LONG-tier decisions.
        We renamed it to ``long_term_view`` to signal that EVERY
        decision must carry a long-term note (even a one-liner
        confirming the multi-year thesis is intact / broken / N/A).

        This validator keeps the old field name working for:
          * Persisted ``RunRecord`` JSON files written before the
            rename.
          * In-flight LLM outputs that still remember the old prompt
            language.
          * Third-party callers writing decisions programmatically.
        """
        if not isinstance(data, dict):
            return data
        # Only migrate when the new key is truly absent — an explicit
        # empty string on ``long_term_view`` is a legitimate signal
        # from the LLM that "no long-term data was cited".
        if "long_term_view" not in data and "long_term_thesis" in data:
            data = dict(data)
            data["long_term_view"] = data.pop("long_term_thesis")
        return data

    @field_validator("horizon", mode="before")
    @classmethod
    def _coerce_horizon(cls, v: object) -> object:
        """Accept free-form horizon strings ("Short", "medium term", …)
        and map them to the canonical ``Horizon`` enum.

        Leaves already-valid ``Horizon`` instances and ``None`` untouched
        — ``None`` is later replaced by ``horizon_days_to_horizon`` in
        the model-level validator so the LLM can safely omit the field.
        """
        if v is None:
            return v
        norm = normalise_horizon(v)
        return norm if norm is not None else v

    @model_validator(mode="after")
    def _infer_horizon_from_days(self) -> "PortfolioDecision":
        """Back-fill ``horizon`` from ``horizon_days`` when the LLM
        emitted only the numeric field.

        This runs *after* the individual field validators — at this
        point ``horizon`` is guaranteed to be a valid ``Horizon`` (the
        Pydantic default kicks in if the JSON payload lacked the key),
        but a legacy ``RunRecord`` written before this field existed
        will have ``MEDIUM`` regardless of its ``horizon_days``.
        We detect that case (``horizon == MEDIUM`` AND
        ``horizon_days`` falls outside the medium range) and re-bucket
        so the display / RLMF loop pick up the right tier.
        """
        inferred = horizon_days_to_horizon(self.horizon_days)
        if self.horizon == Horizon.MEDIUM and inferred != Horizon.MEDIUM:
            # Pydantic ``model_validator(mode="after")`` returns the
            # (possibly mutated) instance — we use object.__setattr__
            # to bypass the frozen-check on the model.
            object.__setattr__(self, "horizon", inferred)
        return self

    @field_validator("action", mode="before")
    @classmethod
    def _coerce_legacy_action(cls, v: object) -> object:
        """Migrate the pre-5-tier ``action`` strings to the new tier space.

        Prior versions of this schema used a 3-tier label
        (``OVERWEIGHT`` / ``NEUTRAL`` / ``UNDERWEIGHT``) mirroring the
        institutional weighting convention.  We now use the retail-
        friendly 5-tier space (``STRONG_BUY`` … ``STRONG_SELL``).  The
        LLM is instructed to emit the new labels, but historical
        ``RunRecord`` JSON files, third-party callers, and older
        prompts may still hand us the legacy triplet.  Coerce here so
        the schema stays a single source of truth without forcing a
        one-shot data-migration script over all persisted runs.

        A plain string that doesn't match either alphabet is left
        as-is (uppercased) so Pydantic surfaces a clean ``Literal``
        validation error to the caller.
        """
        return normalise_action(v) if isinstance(v, str) else v
