"""Guardrail callable wired onto the Portfolio Manager task.

CrewAI's ``Task(guardrail=...)`` accepts a callable
``Callable[[TaskOutput], tuple[bool, Any]]``. Returning ``(False, msg)``
re-prompts the agent with ``msg`` as feedback and retries up to
``guardrail_max_retries`` times.
"""

from typing import Any, Tuple

from .schemas import PortfolioDecision


def confidence_guardrail(task_output) -> Tuple[bool, Any]:
    """Reject implausibly confident or inconsistent PM decisions.

    Triggered by CrewAI when ``pm_task`` finishes. Rejects any of:

    * Output that did not parse to ``PortfolioDecision``.
    * ``confidence > 0.88`` with ``action != NEUTRAL`` (single-name overconfidence).
    * ``size_pct_of_book > 5`` with ``confidence < 0.7`` (over-sizing on weak conviction).
    * ``compliance_status == "BLOCKED"`` but action / size not zeroed out.
    """
    decision: PortfolioDecision | None = task_output.pydantic
    if decision is None:
        return (False, (
            "Output did not parse to PortfolioDecision. Re-emit ONLY a "
            "JSON object that strictly matches the schema fields: action, "
            "confidence, size_pct_of_book, entry_price, stop_loss, "
            "target_price, horizon_days, expected_return_pct, rationale, "
            "key_drivers, key_risks, falsifiers, geopolitical_flags, "
            "compliance_status."
        ))

    if decision.confidence > 0.88 and decision.action != "NEUTRAL":
        return (False, (
            f"confidence={decision.confidence:.2f} is implausibly high "
            "for a single-name trade with macro/news/geopolitical risk. "
            "Re-emit with confidence in [0.55, 0.88] and add TWO concrete "
            "scenarios that would falsify the trade in 'falsifiers'."
        ))

    if decision.size_pct_of_book > 5.0 and decision.confidence < 0.7:
        return (False, (
            f"size_pct_of_book={decision.size_pct_of_book:.1f}% is too "
            f"large for confidence={decision.confidence:.2f}. Either "
            "reduce size to <=5% or raise confidence with explicit "
            "additional evidence."
        ))

    if decision.compliance_status == "BLOCKED" and (
        decision.action != "NEUTRAL" or decision.size_pct_of_book != 0
    ):
        return (False, (
            "compliance_status is BLOCKED but action != NEUTRAL or "
            "size != 0. Re-emit with action='NEUTRAL' and "
            "size_pct_of_book=0."
        ))

    return (True, decision)
