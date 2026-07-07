"""Unit tests for the Reflective Critic (Phase A).

The critic talks to ``crewai.LLM.call(..., response_model=...)``. We stub
that with a fake LLM that returns a script of pre-baked CritiqueResponse
objects so the tests are deterministic and run without network access.
"""

from __future__ import annotations

import pytest

from trading_crew.critic import (
    CritiqueResponse,
    CritiqueVerdict,
    ReflectiveCritic,
    run_reflective_critic,
)
from trading_crew.schemas import PortfolioDecision


def _make_decision(**overrides) -> PortfolioDecision:
    base = dict(
        action="BUY",
        confidence=0.7,
        size_pct_of_book=5.0,
        entry_price=150.0,
        stop_loss=140.0,
        target_price=180.0,
        horizon_days=21,
        expected_return_pct=0.08,
        rationale="Strong earnings beat and guidance raise.",
        key_drivers=["earnings beat", "guidance raise"],
        key_risks=["macro slowdown"],
        falsifiers=["miss next quarter"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
    )
    base.update(overrides)
    return PortfolioDecision(**base)


def _make_response(
    verdict: CritiqueVerdict,
    *,
    intent_ok: bool = True,
    evidence_ok: bool = True,
    risk_ok: bool = True,
    revised_action: str = None,
    revised_size: float = None,
    revised_confidence: float = None,
) -> CritiqueResponse:
    return CritiqueResponse(
        intent_ok=intent_ok,
        intent_reason="ok",
        evidence_ok=evidence_ok,
        evidence_reason="ok",
        counterfactual_flip_evidence="A 50% revenue miss next quarter.",
        risk_ok=risk_ok,
        risk_reason="ok",
        verdict=verdict,
        revised_action=revised_action,
        revised_size_pct=revised_size,
        revised_confidence=revised_confidence,
        overall_comment="test",
    )


class StubLLM:
    """Returns a scripted sequence of CritiqueResponse objects."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls = []
        self.temperature = 0.0

    def call(self, messages, response_model=None, **kwargs):
        self.calls.append({"messages": messages, "temp": self.temperature})
        if not self._responses:
            raise RuntimeError("StubLLM exhausted")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Reflection loop
# ---------------------------------------------------------------------------


def test_ratify_terminates_immediately():
    """A first-iteration RATIFY locks the decision; no revisions."""
    llm = StubLLM([_make_response(CritiqueVerdict.RATIFY)])
    critic = ReflectiveCritic(llm=llm, max_iterations=2, consistency_samples=0)
    decision = _make_decision()
    out, records = critic.review(decision, ticker="AAPL")
    assert out.action == decision.action
    assert out.size_pct_of_book == decision.size_pct_of_book
    assert len(records) == 1
    assert records[0].response.verdict == CritiqueVerdict.RATIFY


def test_abstain_collapses_decision_to_hold():
    """An ABSTAIN verdict forces action=HOLD and size=0.

    The critic previously forced ``NEUTRAL`` because the schema used the
    legacy 3-tier alphabet.  The 5-tier refactor renamed the flat tier to
    ``HOLD`` — we keep the same collapse semantics (size, expected return
    zeroed, confidence capped) so the assertions still document the
    "abstain destroys directionality" contract.
    """
    llm = StubLLM([_make_response(CritiqueVerdict.ABSTAIN)])
    critic = ReflectiveCritic(llm=llm, max_iterations=2, consistency_samples=0)
    out, records = critic.review(_make_decision(), ticker="AAPL")
    assert out.action == "HOLD"
    assert out.size_pct_of_book == 0.0
    assert out.expected_return_pct == 0.0
    assert records[0].abstained is True
    assert "[Critic ABSTAIN:" in out.rationale


def test_revise_then_ratify_applies_changes_then_locks():
    """REVISE on iteration 1 amends the decision; RATIFY on iteration 2 locks the amended version."""
    llm = StubLLM([
        _make_response(CritiqueVerdict.REVISE, revised_size=2.0, revised_confidence=0.4),
        _make_response(CritiqueVerdict.RATIFY),
    ])
    critic = ReflectiveCritic(llm=llm, max_iterations=2, consistency_samples=0)
    out, records = critic.review(_make_decision(size_pct_of_book=5.0, confidence=0.7), ticker="AAPL")
    assert out.size_pct_of_book == 2.0
    assert out.confidence == pytest.approx(0.4)
    assert len(records) == 2
    assert records[0].response.verdict == CritiqueVerdict.REVISE
    assert records[1].response.verdict == CritiqueVerdict.RATIFY


def test_budget_exhausted_returns_last_revision():
    """If max_iterations REVISE responses fire and the budget runs out, the latest revision wins."""
    llm = StubLLM([
        _make_response(CritiqueVerdict.REVISE, revised_size=4.0),
        _make_response(CritiqueVerdict.REVISE, revised_size=3.0),
    ])
    critic = ReflectiveCritic(llm=llm, max_iterations=2, consistency_samples=0)
    out, records = critic.review(_make_decision(), ticker="AAPL")
    assert out.size_pct_of_book == 3.0
    assert len(records) == 2


def test_provider_failure_is_a_clean_abstain():
    """If the LLM raises, the critic returns an abstain with a structured trail."""
    class BrokenLLM:
        temperature = 0.0
        def call(self, *a, **kw):
            raise RuntimeError("provider unreachable")
    critic = ReflectiveCritic(llm=BrokenLLM(), max_iterations=2, consistency_samples=0)
    out, records = critic.review(_make_decision(), ticker="AAPL")
    assert out.action == "HOLD"
    assert out.size_pct_of_book == 0.0
    assert len(records) == 1
    assert records[0].abstained is True


def test_invalid_revised_values_are_ignored():
    """Out-of-range revised values are silently dropped (proposal unchanged)."""
    llm = StubLLM([
        _make_response(
            CritiqueVerdict.REVISE,
            revised_size=150.0,         # > 100 -> ignored
            revised_confidence=2.0,     # > 1   -> ignored
            revised_action="WAT",       # not a valid action -> ignored
        ),
        _make_response(CritiqueVerdict.RATIFY),
    ])
    critic = ReflectiveCritic(llm=llm, max_iterations=2, consistency_samples=0)
    original = _make_decision(size_pct_of_book=5.0, confidence=0.7)
    out, _ = critic.review(original, ticker="AAPL")
    assert out.size_pct_of_book == original.size_pct_of_book
    assert out.confidence == original.confidence
    assert out.action == original.action


# ---------------------------------------------------------------------------
# Consistency vote
# ---------------------------------------------------------------------------


def test_unanimous_ratify_passes_through():
    """3 RATIFY samples -> proposal unchanged."""
    llm = StubLLM([_make_response(CritiqueVerdict.RATIFY)] * 3)
    critic = ReflectiveCritic(llm=llm, max_iterations=0, consistency_samples=3)
    out, records = critic.consistency_vote(_make_decision(), ticker="AAPL")
    assert out.action == "BUY"
    assert len(records) == 3
    # All three samples should record the temperature they used
    temps = [r.temperature for r in records]
    assert sorted(temps) == sorted(critic.consistency_temperatures[:3])


def test_split_vote_forces_abstain():
    """1 RATIFY + 1 REVISE + 1 ABSTAIN -> no mode reaches 2/3 -> ABSTAIN."""
    llm = StubLLM([
        _make_response(CritiqueVerdict.RATIFY),
        _make_response(CritiqueVerdict.REVISE, revised_size=2.0),
        _make_response(CritiqueVerdict.ABSTAIN),
    ])
    critic = ReflectiveCritic(llm=llm, max_iterations=0, consistency_samples=3)
    out, records = critic.consistency_vote(_make_decision(), ticker="AAPL")
    assert out.action == "HOLD"
    assert out.size_pct_of_book == 0.0
    assert len(records) == 3
    assert "Consistency vote split" in out.rationale or "[Critic ABSTAIN:" in out.rationale


def test_majority_revise_applies_lowest_temp_revision():
    """2 REVISE + 1 RATIFY -> the lowest-temperature REVISE wins."""
    llm = StubLLM([
        _make_response(CritiqueVerdict.REVISE, revised_size=2.0),   # temp=0.0
        _make_response(CritiqueVerdict.REVISE, revised_size=3.5),   # temp=0.5
        _make_response(CritiqueVerdict.RATIFY),                     # temp=0.9
    ])
    critic = ReflectiveCritic(llm=llm, max_iterations=0, consistency_samples=3)
    out, records = critic.consistency_vote(_make_decision(size_pct_of_book=5.0), ticker="AAPL")
    assert out.size_pct_of_book == 2.0   # the temp=0.0 sample's value
    assert len(records) == 3


def test_majority_abstain_forces_abstain():
    """2 ABSTAIN + 1 RATIFY -> abstain wins."""
    llm = StubLLM([
        _make_response(CritiqueVerdict.ABSTAIN),
        _make_response(CritiqueVerdict.ABSTAIN),
        _make_response(CritiqueVerdict.RATIFY),
    ])
    critic = ReflectiveCritic(llm=llm, max_iterations=0, consistency_samples=3)
    out, records = critic.consistency_vote(_make_decision(), ticker="AAPL")
    assert out.action == "HOLD"
    assert out.size_pct_of_book == 0.0


def test_no_samples_collected_abstains():
    """If every sample fails, the vote returns ABSTAIN."""
    class AlwaysFail:
        temperature = 0.0
        def call(self, *a, **kw):
            raise RuntimeError("fail")
    critic = ReflectiveCritic(llm=AlwaysFail(), max_iterations=0, consistency_samples=3)
    out, records = critic.consistency_vote(_make_decision(), ticker="AAPL")
    assert out.action == "HOLD"
    assert records == []


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def test_critic_prompt_biases_against_unwarranted_abstain():
    """Critic prompt must explicitly default to RATIFY for sound decisions
    and limit ABSTAIN to structurally compromised proposals.

    Background: the user reported "lately most of the runs are giving
    neutral decision".  Tracing the workflow showed that even when the
    PM emitted a directional decision, the Reflective Critic's ABSTAIN
    verdict was forcing the action back to the flat tier (HOLD, née
    NEUTRAL) — see ``ReflectiveCritic._abstain``.  The fix tightens the
    critic prompt with an explicit default-to-RATIFY clause and a named
    list of conditions that justify ABSTAIN — this test pins that
    language in place so a future prompt edit can't silently reintroduce
    the bias.
    """
    from trading_crew.critic import _build_prompt
    from trading_crew.schemas import PortfolioDecision

    decision = PortfolioDecision(
        action="BUY",
        confidence=0.65,
        size_pct_of_book=2.0,
        entry_price=100.0,
        stop_loss=92.0,
        target_price=120.0,
        horizon_days=90,
        expected_return_pct=0.18,
        rationale="Mid-term thesis supported by analyst evidence "
                  "[source: market_report].",
        key_drivers=["Earnings momentum [source: fundamentals]"],
        key_risks=["Macro slowdown [source: macro_report]"],
        falsifiers=["Earnings miss [source: yfinance .calendar]"],
        geopolitical_flags=[],
        compliance_status="CLEAR",
        sources=["market_report", "fundamentals", "macro_report",
                 "yfinance .calendar"],
    )
    prompt = _build_prompt(decision, "BAJAJ-AUTO.NS", 0, 2)
    low = prompt.lower()
    assert "default to ratify" in low, (
        "Critic prompt must explicitly default to RATIFY — without this "
        "the LLM tends to ABSTAIN on any non-trivial decision, which "
        "_abstain() then collapses to HOLD with size=0."
    )
    assert "structurally compromised" in low, (
        "Critic prompt must name the ABSTAIN bar (structurally "
        "compromised) so a weak signal doesn't trigger an over-veto."
    )
    assert "hold is a legitimate action" in low or "hold is" in low, (
        "Critic prompt must acknowledge HOLD as a legitimate PM action "
        "so the critic doesn't try to force directional bets when the PM "
        "correctly chose to sit out."
    )
    # Also make sure the prompt tells the LLM the exact 5-tier alphabet
    # so it doesn't hallucinate the legacy names in ``revised_action``.
    assert "strong_buy" in low and "strong_sell" in low, (
        "Critic prompt must enumerate the STRONG_BUY / BUY / HOLD / "
        "SELL / STRONG_SELL alphabet so the LLM emits valid revisions."
    )


def test_run_reflective_critic_runs_loop_then_vote():
    """The convenience entry point runs the loop first, then the vote."""
    llm = StubLLM([
        _make_response(CritiqueVerdict.RATIFY),  # loop iteration 1
        _make_response(CritiqueVerdict.RATIFY),  # vote sample 1
        _make_response(CritiqueVerdict.RATIFY),  # vote sample 2
        _make_response(CritiqueVerdict.RATIFY),  # vote sample 3
    ])
    out, records = run_reflective_critic(
        _make_decision(), ticker="AAPL", llm=llm,
        max_iterations=2, consistency_samples=3,
    )
    assert out.action == "BUY"
    # 1 loop record + 3 vote records
    assert len(records) == 4


def test_run_reflective_critic_skips_vote_after_loop_abstain():
    """If the loop abstains, the (expensive) consistency vote is skipped."""
    llm = StubLLM([_make_response(CritiqueVerdict.ABSTAIN)])
    out, records = run_reflective_critic(
        _make_decision(), ticker="AAPL", llm=llm,
        max_iterations=2, consistency_samples=3,
    )
    assert out.action == "HOLD"
    assert len(records) == 1  # only the loop's abstain record, no vote samples


# ---------------------------------------------------------------------------
# Provenance prompt — the critic must surface the PM's sources list and
# instruct the LLM to treat an inline [source: …] tag as sufficient
# evidence. Without this the model was over-eagerly flagging every number
# as "fabricated" even when the data was real.
# ---------------------------------------------------------------------------


def test_prompt_shows_pm_sources_list_when_populated():
    from trading_crew.critic import _build_prompt

    decision = _make_decision(
        rationale=(
            "Earnings beat at EPS=$0.68 [source: yfinance .info NTNX] vs "
            "consensus $0.55 [source: yfinance recommendations_summary NTNX]."
        ),
        sources=[
            "yfinance .info NTNX",
            "yfinance recommendations_summary NTNX",
        ],
    )
    prompt = _build_prompt(decision, ticker="NTNX", iteration=0, max_iters=2)
    assert "Sources cited" in prompt
    assert "yfinance .info NTNX" in prompt
    assert "yfinance recommendations_summary NTNX" in prompt


def test_prompt_warns_when_pm_sources_empty():
    from trading_crew.critic import _build_prompt

    decision = _make_decision(
        rationale="Earnings beat at EPS=$0.68 vs consensus $0.55.",
        sources=[],
    )
    prompt = _build_prompt(decision, ticker="NTNX", iteration=0, max_iters=2)
    # Empty list should trigger a hint that the rationale lost provenance.
    assert "Sources cited" in prompt
    assert "should have been mirrored here" in prompt


def test_prompt_defines_inline_tag_as_sufficient_evidence():
    """The Evidence stage MUST tell the LLM that a matching tag IS the
    provenance — otherwise it goes back to flagging legitimate numbers
    as fabricated."""
    from trading_crew.critic import _build_prompt

    prompt = _build_prompt(_make_decision(), ticker="NTNX", iteration=0, max_iters=2)
    assert "[source:" in prompt
    # The protocol step needs the must-be-in-sources rule + the
    # don't-re-verify rule, both. Either-or wouldn't fix the regression.
    assert "appear in the \"Sources cited\" list" in prompt or "appear in \"Sources cited\"" in prompt
    assert "you do not need to (and must not) re-verify" in prompt


def test_portfolio_decision_sources_defaults_to_empty_list():
    """Existing callers that built decisions without ``sources`` must
    still work — the field defaults to an empty list."""
    pd = _make_decision()
    assert pd.sources == []


# ---------------------------------------------------------------------------
# Source-realism guardrail — Stage 3 was repurposed from the old
# "Backtest role" (backtest is gone from decision reasoning entirely) to a
# "Source realism" stage.  The critic must REVISE decisions that cite a
# fabricated / placeholder URL (example.com) while RATIFYING an honestly
# labelled ``[analyst judgement]`` qualitative claim — this is the direct
# fix for the "too many unverified sources → HOLD" pathology the user hit
# on HAL.NS / COCHINSHIP.NS.
# ---------------------------------------------------------------------------


def test_prompt_has_no_backtest_language():
    """Backtest was removed from the DECISION reasoning entirely, so the
    critic prompt must no longer mention it — otherwise it would re-inject
    the past-heavy statistical framing the user asked us to remove."""
    from trading_crew.critic import _build_prompt

    prompt = _build_prompt(_make_decision(), ticker="NTNX", iteration=0, max_iters=2)
    low = prompt.lower()
    assert "backtest" not in low, (
        "Critic prompt must not mention backtest — it was removed from "
        "the decision reasoning (M6 Backtest tab is independent)."
    )
    assert "expectancy" not in low, (
        "Critic prompt must not mention expectancy — backtest framing is "
        "gone from the decision path."
    )


def test_prompt_defines_source_realism_stage():
    """The critic prompt must contain a dedicated Source-realism stage
    that (a) flags a fabricated placeholder URL as a REVISE-worthy
    integrity failure and (b) accepts an honest ``[analyst judgement]``
    label as valid qualitative support (not a HOLD trigger)."""
    from trading_crew.critic import _build_prompt

    prompt = _build_prompt(_make_decision(), ticker="NTNX", iteration=0, max_iters=2)
    low = prompt.lower()
    assert "source realism" in low, (
        "Critic prompt must have a dedicated 'Source realism' protocol "
        "stage so it catches fabricated citations."
    )
    assert "example.com" in low, (
        "Critic prompt must name the placeholder host (example.com) so "
        "the LLM can pattern-match fabricated citations."
    )
    assert "analyst judgement" in low, (
        "Critic prompt must recognise the honest '[analyst judgement]' "
        "label as acceptable qualitative support."
    )


def test_prompt_shows_correct_and_incorrect_examples_of_source_realism():
    """The critic needs a positive AND a negative example so it can
    distinguish an honest qualitative label from a fabricated URL."""
    from trading_crew.critic import _build_prompt

    prompt = _build_prompt(_make_decision(), ticker="NTNX", iteration=0, max_iters=2)
    low = prompt.lower()
    # Positive example (honest [analyst judgement] label is CORRECT).
    assert "[analyst judgement]" in low, (
        "Prompt must include a positive example using the honest "
        "'[analyst judgement]' label."
    )
    # Negative example (fabricated example.com citation is INCORRECT).
    assert "example.com" in low, (
        "Prompt must include the negative example (a fabricated "
        "example.com citation) so the critic can pattern-match it."
    )
