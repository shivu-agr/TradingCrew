"""Reflective Critic — paper §5.2 (M4).

A bounded self-audit that critiques the ``PortfolioDecision`` *before* the
deterministic execution pipeline runs.  Three guarantees from the survey
the existing Quality Reviewer / Risk Team / Compliance Officer agents do
*not* provide:

1. **Typed verdict over the proposal itself** — the critic emits a
   ``CritiqueResponse`` with structured booleans per stage (intent /
   evidence / risk) so the UI can render a checklist.
2. **Bounded reflection budget** — at most ``max_iterations`` REVISE loops;
   RATIFY is terminal, ABSTAIN is terminal.
3. **Multi-temperature consistency vote** — sample the critic at 3
   temperatures.  If the modal verdict has < 2/3 share, force ABSTAIN
   regardless of any single sample's vote (paper §5.2.b).

The critic is deliberately not a CrewAI ``Agent``.  Using ``LLM.call()``
directly (with ``response_model=CritiqueResponse``) avoids spinning up a
mini-Crew per sample and lets us vary the temperature per sample, which
the CrewAI Agent abstraction doesn't expose cleanly.

This module mirrors ``tradingagents/agents/reflection/critic.py`` in the
LangGraph project but operates on ``PortfolioDecision`` (CrewAI's typed
output) instead of ``ActionProposal``.  The downstream bridge
(``trading_crew.agentic.bridge``) still converts to ``ActionProposal``,
but now starting from the *post-critic* decision.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

from pydantic import BaseModel, Field

from trading_crew.schemas import PortfolioDecision, normalise_action


# Canonical 5-tier action space the critic understands.  Kept in sync with
# ``PortfolioDecision.action`` — the schema is the source of truth; this is
# a fast lookup so ``_apply_revision`` can validate a critic-suggested
# replacement without instantiating a full ``PortfolioDecision`` just to
# leverage the Pydantic Literal.
_VALID_ACTION_TIERS: frozenset[str] = frozenset(
    {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict + structured output
# ---------------------------------------------------------------------------


class CritiqueVerdict(str, Enum):
    RATIFY = "RATIFY"
    REVISE = "REVISE"
    ABSTAIN = "ABSTAIN"


class CritiqueResponse(BaseModel):
    """Structured output of one critique pass.

    Field descriptions double as prompting instructions to the LLM.
    """

    intent_ok: bool = Field(
        description=(
            "True if the proposal's action (STRONG_BUY / BUY / HOLD / SELL "
            "/ STRONG_SELL) follows from the analyst evidence as cited in "
            "the rationale."
        ),
    )
    intent_reason: str = Field(
        description="Brief justification for the intent check (1-2 sentences). Cite which analyst report supports or contradicts the action.",
    )

    evidence_ok: bool = Field(
        description=(
            "True if every quantitative claim in the rationale carries an "
            "inline [source: <identifier>] tag AND that <identifier> "
            "appears in the decision's `sources` list. False if any number "
            "is bare (no tag) or its tag identifier is not in `sources`. "
            "Treat the *presence* of a matching tag as sufficient evidence "
            "of provenance — you don't need to re-fetch the data. Do NOT "
            "mark False just because you can't independently verify a "
            "number when its [source: …] tag is present and listed in "
            "`sources`."
        ),
    )
    evidence_reason: str = Field(
        description=(
            "Brief justification for the evidence check (1-2 sentences). "
            "When evidence_ok=False, name the specific bare claim or the "
            "tag identifier that is missing from `sources`."
        ),
    )

    counterfactual_flip_evidence: str = Field(
        description="Name the single most important piece of evidence that, if reversed, would flip the action. Keep to one short sentence.",
    )

    risk_ok: bool = Field(
        description="True if the key_risks list is honest and complete relative to the analyst evidence; False if a major risk category (macro / liquidity / regime / compliance) is missing.",
    )
    risk_reason: str = Field(
        description="Brief justification for the risk check (1-2 sentences). Name the missing risk if risk_ok=False.",
    )

    verdict: CritiqueVerdict = Field(
        description="Final verdict: RATIFY (proposal is sound), REVISE (suggest adjustments), or ABSTAIN (too uncertain or compromised to trade).",
    )

    revised_action: Optional[str] = Field(
        default=None,
        description=(
            "When verdict=REVISE, the suggested replacement action. MUST be "
            "one of STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL. Otherwise null."
        ),
    )
    revised_size_pct: Optional[float] = Field(
        default=None,
        description="When verdict=REVISE, the suggested replacement size_pct_of_book in [0, 100]. Otherwise null.",
    )
    revised_confidence: Optional[float] = Field(
        default=None,
        description="When verdict=REVISE, the suggested replacement confidence in [0, 1]. Otherwise null.",
    )
    overall_comment: str = Field(
        description="One-paragraph summary the user reads in the critic panel.",
    )


# ---------------------------------------------------------------------------
# Reflection records — what the UI renders
# ---------------------------------------------------------------------------


@dataclass
class ReflectionRecord:
    iteration: int
    response: CritiqueResponse
    decision_before: dict
    decision_after: dict
    abstained: bool = False
    temperature: Optional[float] = None  # populated by the consistency vote


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _build_prompt(decision: PortfolioDecision, ticker: str, iteration: int, max_iters: int) -> str:
    drivers = "\n  - ".join(decision.key_drivers or []) or "(none listed)"
    risks = "\n  - ".join(decision.key_risks or []) or "(none listed)"
    falsifiers = "\n  - ".join(decision.falsifiers or []) or "(none listed)"
    sources_list = getattr(decision, "sources", None) or []
    if sources_list:
        sources = "\n  - " + "\n  - ".join(sources_list)
    else:
        sources = " (none — every numeric claim in the rationale should " \
                  "have been mirrored here as a [source: <id>] identifier)"
    return f"""You are the Reflective Critic for a multi-agent trading system.

Apply the 5-stage protocol below to the Portfolio Manager's decision. Be strict but fair — your job is to catch failure modes before any fill is attempted, not to second-guess every reasonable trade.

**Default to RATIFY when the decision is internally consistent and supported.** Only choose REVISE for a specific, named flaw (e.g. confidence > 0.8 while the qualitative evidence is thin or contradictory, or a fabricated source).  Only choose ABSTAIN when the decision is **structurally compromised** — i.e. the rationale internally contradicts the action, compliance is BLOCKED, OR every numeric claim is unsourced.  A weak / mixed signal is NOT grounds for ABSTAIN; it is grounds for REVISE with a smaller size or a lower-conviction tier (e.g. STRONG_BUY → BUY, or BUY → HOLD).  HOLD is a legitimate action — do NOT REVISE a HOLD to a directional bet unless the rationale already argues for direction.

The 5-tier action alphabet is: **STRONG_BUY**, **BUY**, **HOLD**, **SELL**, **STRONG_SELL**.  Use ``revised_action`` (only when verdict=REVISE) to suggest a tier change — e.g. cut size AND downgrade STRONG_BUY to BUY when the analyst evidence is thinner than the conviction implies, or promote BUY to STRONG_BUY only if the evidence is overwhelming (which is rare).

Iteration {iteration + 1} of {max_iters}. On REVISE the decision is amended and re-fed; on RATIFY it is locked in; ABSTAIN is terminal.

## Decision under audit

- Ticker: {ticker}
- Action: {decision.action}
- Confidence: {decision.confidence:.2f}
- Size: {decision.size_pct_of_book:.2f}% of book
- Entry/Stop/Target: ${decision.entry_price:.2f} / ${decision.stop_loss:.2f} / ${decision.target_price:.2f}
- Horizon: {decision.horizon_days} days
- Expected return: {decision.expected_return_pct * 100:+.2f}%
- Compliance: {decision.compliance_status}

## Rationale

{decision.rationale}

## Key drivers
  - {drivers}

## Key risks
  - {risks}

## Falsifiers
  - {falsifiers}

## Sources cited (provenance trail produced by the analyst tools)
{sources}

## Protocol

1. **Intent**         — Does the action (STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL) follow from the rationale and drivers? Does the *tier* (STRONG_* vs plain) match the conviction level implied by the evidence?

2. **Evidence**       — Provenance check. Every numeric / quantitative claim in the rationale (and drivers / risks / falsifiers) MUST end with an inline ``[source: <identifier>]`` tag, AND that ``<identifier>`` MUST appear in the "Sources cited" list above. The tag IS the evidence — you do not need to (and must not) re-verify the underlying data. Mark ``evidence_ok=False`` ONLY when:
     (a) a numeric claim has no inline ``[source: …]`` tag at all, OR
     (b) a tag identifier in the rationale is missing from "Sources cited", OR
     (c) the "Sources cited" list is empty while the rationale carries quantitative claims.
   Do NOT mark False just because the numbers feel "uncited" — if the tag is there and listed, that is sufficient provenance for this stage.

3. **Source realism**  — Judge the HONESTY of the citations, not their completeness.  Two very different cases:

     (a) FABRICATED source — a citation that points at an obviously placeholder / made-up URL (``example.com``, ``example.org``, ``example.net``, ``test.com``, a bare ``http(s)://`` with no host, or a plainly invented path).  This is a real integrity problem: the model invented a source to satisfy a provenance rule.  Flag ``evidence_ok=False`` and REVISE — the fabricated citation must be removed and the underlying claim either re-sourced or re-labelled as ``[analyst judgement]``.

     (b) HONEST qualitative judgement — a claim the analyst explicitly labelled ``[analyst judgement]`` (i.e. reasoned domain knowledge with no single citable URL).  This is ACCEPTABLE, not a violation.  Do NOT push such a claim to REVISE / ABSTAIN and do NOT force HOLD merely because a well-reasoned thesis leans on labelled judgement rather than a URL for every sentence.  A missing URL downgrades at most that ONE claim's weight — never the whole thesis.

     (c) FABRICATED PRECISION — a claim that attaches a REAL tool's source tag to a figure that tool cannot produce.  The clearest tells: a **specific calendar-year range** (e.g. "2019–2023") or a **ROCE trend** tagged ``[source: yfinance get_long_term_thesis]`` — the yfinance path reports only relative fiscal-year windows and NO ROCE, so precise years / ROCE trends carrying that tag are invented.  Treat this exactly like a fabricated URL: ``evidence_ok=False`` and REVISE — the invented specificity must be dropped or re-labelled ``[analyst judgement]``.

   A rationale that says "management guided 18% revenue CAGR [source: yfinance ...]" plus "the domestic capex cycle favours the sector [analyst judgement]" is CORRECT — ratify.  A rationale citing "[source: https://example.com/hal-order-book]" is INCORRECT (fabricated) — revise and strip the fake source.  A rationale citing "ROCE fell 18%→12% (2019–2023) [source: yfinance get_long_term_thesis]" is INCORRECT (fabricated precision) — revise: that tool never emitted a ROCE trend or those calendar years.

4. **Counterfactual** — What single piece of evidence, if reversed, would flip the decision? Keep to one sentence.

5. **Risk**           — Is key_risks complete? Missing macro / liquidity / regime / compliance risks → False.

6. **Verdict**        — RATIFY, REVISE (with revised_action/size/confidence), or ABSTAIN.

Emit the CritiqueResponse now."""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class ReflectiveCritic:
    """Runs the critic with a bounded reflection budget and a consistency vote."""

    llm: Any
    max_iterations: int = 2
    consistency_samples: int = 3
    consistency_threshold: float = 2 / 3
    consistency_temperatures: Tuple[float, ...] = (0.0, 0.5, 0.9)

    # ------------------------------------------------------ reflection loop
    def review(
        self,
        decision: PortfolioDecision,
        ticker: str,
    ) -> Tuple[PortfolioDecision, List[ReflectionRecord]]:
        current = decision
        records: List[ReflectionRecord] = []
        for iteration in range(self.max_iterations):
            response = self._invoke(current, ticker, iteration)
            if response is None:
                abstained = self._abstain(current, "Critic provider could not produce structured output.")
                records.append(ReflectionRecord(
                    iteration=iteration,
                    response=self._abstain_response("Provider unsupported."),
                    decision_before=current.model_dump(),
                    decision_after=abstained.model_dump(),
                    abstained=True,
                ))
                return abstained, records

            before = current.model_dump()
            if response.verdict == CritiqueVerdict.RATIFY:
                records.append(ReflectionRecord(iteration, response, before, before))
                return current, records
            if response.verdict == CritiqueVerdict.ABSTAIN:
                abstained = self._abstain(current, response.overall_comment)
                records.append(ReflectionRecord(
                    iteration, response, before, abstained.model_dump(), abstained=True,
                ))
                return abstained, records

            revised = self._apply_revision(current, response)
            records.append(ReflectionRecord(iteration, response, before, revised.model_dump()))
            current = revised
        return current, records

    # ------------------------------------------------------ consistency vote
    def consistency_vote(
        self,
        decision: PortfolioDecision,
        ticker: str,
    ) -> Tuple[PortfolioDecision, List[ReflectionRecord]]:
        records: List[ReflectionRecord] = []
        verdicts: List[CritiqueVerdict] = []
        for i, temp in enumerate(self.consistency_temperatures[: self.consistency_samples]):
            response = self._invoke(decision, ticker, iteration=i, temperature=temp)
            if response is None:
                continue
            verdicts.append(response.verdict)
            records.append(ReflectionRecord(
                iteration=i,
                response=response,
                decision_before=decision.model_dump(),
                decision_after=decision.model_dump(),
                temperature=temp,
            ))

        if not verdicts:
            return self._abstain(decision, "Consistency vote produced no valid samples."), records

        counts = Counter(verdicts)
        mode_verdict, mode_count = counts.most_common(1)[0]
        if mode_count / len(verdicts) < self.consistency_threshold:
            return (
                self._abstain(
                    decision,
                    f"Consistency vote split (verdicts={dict(counts)}); downgrading to ABSTAIN.",
                ),
                records,
            )

        if mode_verdict == CritiqueVerdict.ABSTAIN:
            return self._abstain(decision, "Consistency vote converged on ABSTAIN."), records
        if mode_verdict == CritiqueVerdict.RATIFY:
            return decision, records

        # REVISE — apply the lowest-temperature REVISE response (most stable)
        revising = next(r.response for r in records if r.response.verdict == CritiqueVerdict.REVISE)
        return self._apply_revision(decision, revising), records

    # ------------------------------------------------------ helpers
    def _invoke(
        self,
        decision: PortfolioDecision,
        ticker: str,
        iteration: int,
        temperature: Optional[float] = None,
    ) -> Optional[CritiqueResponse]:
        prompt = _build_prompt(decision, ticker, iteration, self.max_iterations)
        try:
            # CrewAI's LLM.call() accepts response_model directly. Temperature
            # overrides are not honoured per-call by every provider, but
            # LiteLLM does set them in the request payload — providers that
            # ignore it just emit the same sample (which the consistency
            # vote catches naturally as "high agreement = strong signal").
            if temperature is not None:
                # Some providers ignore per-call temperature; we still pass
                # it so structured logs reflect the intended sampling.
                self.llm.temperature = temperature
            raw = self.llm.call(
                messages=[{"role": "user", "content": prompt}],
                response_model=CritiqueResponse,
            )
        except Exception as exc:
            logger.warning("ReflectiveCritic invocation failed: %s", exc)
            return None

        if isinstance(raw, CritiqueResponse):
            return raw
        if isinstance(raw, dict):
            try:
                return CritiqueResponse(**raw)
            except Exception as exc:
                logger.warning("Critic response did not validate: %s", exc)
                return None
        # Some providers return a JSON string even with response_model.
        if isinstance(raw, str):
            try:
                import json
                return CritiqueResponse(**json.loads(raw))
            except Exception:
                logger.warning("Critic response was a string that did not parse as JSON")
                return None
        return None

    def _apply_revision(self, decision: PortfolioDecision, response: CritiqueResponse) -> PortfolioDecision:
        update: dict = {}
        if response.revised_action:
            # Accept both the current 5-tier alphabet and the legacy triplet
            # (older critic prompts / cached samples still emit
            # OVERWEIGHT/NEUTRAL/UNDERWEIGHT).  ``normalise_action`` folds
            # the legacy strings into the 5-tier space; anything else is
            # rejected so a hallucinated tier can't silently overwrite
            # the PM's original label.
            canonical = normalise_action(response.revised_action)
            if canonical in _VALID_ACTION_TIERS:
                update["action"] = canonical
        if response.revised_size_pct is not None and 0.0 <= response.revised_size_pct <= 100.0:
            update["size_pct_of_book"] = float(response.revised_size_pct)
        if response.revised_confidence is not None and 0.0 <= response.revised_confidence <= 1.0:
            update["confidence"] = float(response.revised_confidence)
        if not update:
            return decision
        try:
            return decision.model_copy(update=update)
        except Exception as exc:
            logger.warning("Could not apply critic revision: %s", exc)
            return decision

    @staticmethod
    def _abstain(decision: PortfolioDecision, reason: str) -> PortfolioDecision:
        """Force an abstain by collapsing action / size / expected_return.

        We keep the original rationale prefixed with the critic's reason so
        the audit trail explains *why* the LLM's verdict was overridden.
        """
        return decision.model_copy(update={
            "action": "HOLD",
            "size_pct_of_book": 0.0,
            "confidence": min(decision.confidence, 0.2),
            "expected_return_pct": 0.0,
            "rationale": f"[Critic ABSTAIN: {reason}]\n\n{decision.rationale}",
        })

    @staticmethod
    def _abstain_response(reason: str) -> CritiqueResponse:
        return CritiqueResponse(
            intent_ok=False, intent_reason=reason,
            evidence_ok=False, evidence_reason=reason,
            counterfactual_flip_evidence="(unable to evaluate)",
            risk_ok=False, risk_reason=reason,
            verdict=CritiqueVerdict.ABSTAIN,
            overall_comment=reason,
        )


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------


def run_reflective_critic(
    decision: PortfolioDecision,
    ticker: str,
    llm: Any,
    *,
    max_iterations: int = 2,
    consistency_samples: int = 3,
) -> Tuple[PortfolioDecision, List[ReflectionRecord]]:
    """Run the bounded reflection loop *then* the consistency vote.

    Returns ``(final_decision, all_records)``.  ``all_records`` is the
    concatenation of the reflection-loop records (with ``temperature=None``)
    and the consistency-vote records (with the temperature each sample
    used).  The runner emits these as a single ``reflection_records``
    event the UI renders as a per-sample checklist.
    """
    critic = ReflectiveCritic(
        llm=llm,
        max_iterations=max_iterations,
        consistency_samples=consistency_samples,
    )
    revised, records = critic.review(decision, ticker)
    if revised.action != "HOLD" or revised.size_pct_of_book > 0:
        # Only run the (expensive) consistency vote on non-abstained proposals.
        # ("HOLD" with size == 0 is the canonical abstain shape.)
        voted, vote_records = critic.consistency_vote(revised, ticker)
        records.extend(vote_records)
        revised = voted
    return revised, records


def records_to_payload(records: List[ReflectionRecord]) -> List[dict]:
    """Serialise reflection records for the WebSocket event."""
    out = []
    for r in records:
        out.append({
            "iteration": r.iteration,
            "abstained": r.abstained,
            "temperature": r.temperature,
            "verdict": r.response.verdict.value,
            "intent_ok": r.response.intent_ok,
            "intent_reason": r.response.intent_reason,
            "evidence_ok": r.response.evidence_ok,
            "evidence_reason": r.response.evidence_reason,
            "counterfactual_flip_evidence": r.response.counterfactual_flip_evidence,
            "risk_ok": r.response.risk_ok,
            "risk_reason": r.response.risk_reason,
            "overall_comment": r.response.overall_comment,
            "revised_action": r.response.revised_action,
            "revised_size_pct": r.response.revised_size_pct,
            "revised_confidence": r.response.revised_confidence,
        })
    return out
