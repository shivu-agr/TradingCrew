"""Lightweight contract tests over ``trading_crew/config/tasks.yaml``.

These don't run a crew — they just lock in the textual contracts that
were added in Phase A so prompt regressions surface immediately
instead of silently degrading runs:

* Every analyst task description / expected output references a
  ``TLDR:`` block — so downstream tasks can lean on the TLDR when the
  LLM has a smaller context window (Gemma-4 fix) and so the OSS-120B
  tool-loop gets a clear "stop calling tools, write the summary"
  signal (BAJAJ-AUTO / LT.NS / BHARATFORG degraded-output fix).
* Tool-heavy analyst tasks (social / news / geopolitical) state an
  explicit ``Tool budget:`` line so OSS-120B doesn't loop on Indian
  tickers.
* The News task explicitly calls ``get_corporate_actions`` (the new
  hook for buyback / dividend / split / M&A) so a Bajaj-Auto-style
  share-buyback is no longer silently dropped.
* The Research Manager prompt enforces the multi-lens rule (at least
  TWO of drivers+risks must source from a non-yfinance lens) so the
  manager can't collapse onto yfinance-only output as it did in the
  user's BAJAJ-AUTO screenshot.
* The Bull / Bear / Risk debate rounds require a REBUTTAL: opener +
  a Conviction shift line on round_num > 1 so multi-turn rounds are
  actually different from each other. Bear and risk-wing round 1 are
  independent openings (not a rebuttal of a turn that has not happened).
"""

from __future__ import annotations

from pathlib import Path

import yaml


_TASKS_YAML = (
    Path(__file__).resolve().parents[1]
    / "trading_crew" / "config" / "tasks.yaml"
)


def _load_tasks() -> dict[str, dict]:
    return yaml.safe_load(_TASKS_YAML.read_text())


ANALYST_TASKS = [
    "market_task", "news_task", "fundamentals_task", "reports_task", "macro_task",
]


def test_every_analyst_task_requires_tldr_block():
    """All 5 analyst tasks must instruct the LLM to open with a
    ``TLDR:`` block.  Downstream tasks (research manager / trader /
    PM) rely on this when context is tight."""
    tasks = _load_tasks()
    analyst_tasks = ANALYST_TASKS
    for name in analyst_tasks:
        text = (tasks[name].get("expected_output") or "").lower()
        assert "tldr:" in text, (
            f"{name} expected_output must mention a 'TLDR:' opener — "
            "needed for smaller-window LLMs (Gemma-4) and for the "
            "OSS-120B tool-loop guardrail to stop cleanly."
        )


def test_tool_heavy_tasks_declare_explicit_tool_budget():
    """Every merged analyst task must state a ``Tool budget:`` line so
    the LLM is told to stop calling tools after a fixed number of calls
    (the OSS-120B tool-loop degradation guardrail on Indian tickers)."""
    tasks = _load_tasks()
    for name in ANALYST_TASKS:
        desc = (tasks[name].get("description") or "").lower()
        assert "tool budget" in desc, (
            f"{name} description must state an explicit Tool budget — "
            "without it OSS-120B loops on Indian tickers and emits a "
            "tool-call list as its final answer (BAJAJ-AUTO / LT.NS / "
            "BHARATFORG.NS degraded-output pattern)."
        )


def test_news_task_explicitly_requires_corporate_actions_call():
    """The user-reported "Bajaj Auto buyback news was not fetched"
    bug came from the generic get_news query missing corporate
    actions.  The fix wires ``get_corporate_actions`` into the
    News Analyst task — verify the prompt actually calls it."""
    tasks = _load_tasks()
    news_desc = tasks["news_task"]["description"]
    news_out = tasks["news_task"]["expected_output"]
    assert "get_corporate_actions" in news_desc, (
        "news_task must explicitly call get_corporate_actions to "
        "surface buybacks / dividends / M&A that get_news misses."
    )
    assert "corporate actions" in news_out.lower(), (
        "news_task expected_output must mention a Corporate Actions "
        "section so the report has a dedicated slot for those hits."
    )


def test_research_manager_enforces_multi_lens_rule():
    """Research Manager prompt must require at least TWO non-yfinance
    citations across drivers+risks — otherwise the manager collapses
    onto yfinance numbers (BAJAJ-AUTO screenshot pattern) and silently
    discards news / social / geopolitical findings."""
    tasks = _load_tasks()
    rs_desc = tasks["research_synthesis"]["description"].lower()
    assert "non-yfinance" in rs_desc, (
        "research_synthesis must require at least one driver+risk "
        "from a non-yfinance lens; otherwise the synthesis defaults "
        "to yfinance-only output."
    )
    assert "degraded" in rs_desc, (
        "research_synthesis must also call out degraded analyst "
        "reports rather than silently treating them as absent."
    )


def test_mixed_is_not_an_automatic_hold():
    """Autonomous HOLDs were collapsing MIXED → sit-out even on 3–0 / 3–1
    boards. MIXED is a locked thesis; HOLD is only a true split or a
    majority-opposed directional vote. Neutrals do not create a split."""
    tasks = _load_tasks()
    rs = tasks["research_synthesis"]["description"]
    pm = tasks["pm_decision"]["description"]
    assert "do not tell the PM to sit out" in rs
    assert "the PM will HOLD" not in rs
    assert "Neutrals do not create a split" in rs
    assert "bullish count == bearish count" in pm
    assert "Neutrals do not create a split" in pm
    assert "3–1 directional majority is not a split" in pm or (
        "3-1 directional majority is not a split" in pm
    )


def test_trader_task_defaults_to_mid_long_term_horizon():
    """Trader prompt must reason across the three-tier horizon
    vocabulary (SHORT / MEDIUM / LONG) with MEDIUM as the default —
    not anchor on the short-term panel.  The user reported that "most
    recent runs give a NEUTRAL decision" — the diagnosis was that the
    trader was anchoring on the short-term panel and the PM was
    inheriting that bias.  The fix asks the trader to pick the tier
    that best matches the analyst evidence and bias toward MEDIUM /
    LONG by default.
    """
    tasks = _load_tasks()
    trader_desc = tasks["trader_task"]["description"]
    trader_desc_lower = trader_desc.lower()
    # New three-tier vocabulary — the vocabulary is what makes the
    # bias explicit; the LLM sees the exact tier names.
    for tier in ("SHORT", "MEDIUM", "LONG"):
        assert tier in trader_desc, (
            f"trader_task must enumerate the '{tier}' horizon tier so "
            f"the LLM has a canonical vocabulary to reason with."
        )
    # Mid-to-long bias — either the explicit phrase or the "default to
    # MEDIUM" instruction is fine.
    trader_desc_flat = " ".join(trader_desc_lower.split())
    assert (
        "mid-to-long" in trader_desc_flat
        or "default" in trader_desc_flat and "medium" in trader_desc_flat
        or "bias toward medium" in trader_desc_flat
    ), (
        "trader_task must express the desk's mid-to-long-term bias "
        "so the trader doesn't default to short-term horizons."
    )
    # A concrete default anchor value must still be present so the LLM
    # has a starting point.  90 / 120 / 180 all remain valid MEDIUM
    # anchors; 1260 (5y) is the LONG anchor.
    assert any(n in trader_desc for n in ("90", "126", "180", "252", "1260")), (
        "trader_task should mention a concrete MEDIUM / LONG day-count "
        "so the LLM has an anchor to start from."
    )
    assert "learning" in trader_desc_lower and "30" in trader_desc, (
        "trader_task must keep the 30-day learning clock distinct from "
        "the MEDIUM/LONG thesis horizon."
    )


def test_pm_decision_anchors_on_thesis_character_and_limits_hold():
    """PM prompt must (a) reason across the three-tier horizon
    vocabulary (SHORT / MEDIUM / LONG), (b) pick the horizon from the
    CHARACTER of the analyst thesis (not from whichever backtest row
    looks best), and (c) state explicitly that HOLD is reserved for
    genuinely contradictory evidence — not a soft veto.

    Rewritten from the older "default to MEDIUM" contract: pinning
    every decision to MEDIUM by default was itself the cause of the
    horizon-fitting-to-backtest anti-pattern (the COCHINSHIP.NS
    SELL that landed on MEDIUM 252d because the MEDIUM row had the
    best expectancy, even though the thesis was a LONG-tier
    structural break).  The new contract makes the thesis character
    drive the tier and lets LONG win when the fundamentals warrant
    it, regardless of what the historical panel shows.

    ``HOLD`` is the 5-tier successor to the legacy ``NEUTRAL``
    label; the prompt was rewritten to the new alphabet at the same
    time as the schema, so this test asserts on the new terminology.
    """
    tasks = _load_tasks()
    pm_desc = tasks["pm_decision"]["description"]
    pm_desc_lower = pm_desc.lower()
    # Anchor: three-tier vocabulary must be enumerated (SHORT / MEDIUM
    # / LONG) so the LLM emits ``horizon`` alongside ``horizon_days``.
    for tier in ("SHORT", "MEDIUM", "LONG"):
        assert tier in pm_desc, (
            f"pm_decision must enumerate the '{tier}' horizon tier."
        )
    # Horizon must be picked from thesis CHARACTER, not from the
    # backtest table.  The prompt must state this explicitly.
    assert "character" in pm_desc_lower, (
        "pm_decision must instruct the PM to pick the horizon from "
        "the CHARACTER of the analyst thesis (not from whichever "
        "backtest row has the highest expectancy)."
    )
    assert "30" in pm_desc and "desk_strategy" in pm_desc_lower, (
        "pm_decision must keep the 30-day learning clock (desk_strategy) "
        "separate from the thesis horizon."
    )
    # LONG-tier escalation still references get_long_term_thesis as
    # a *strong* corroboration for LONG — its absence no longer
    # disqualifies LONG outright (a structural-break thesis is a
    # legitimate LONG-tier trade even without a dedicated tool call).
    assert "get_long_term_thesis" in pm_desc or "long_term_thesis" in pm_desc_lower, (
        "pm_decision must reference the long-term-thesis tool as "
        "strong corroboration for horizon=LONG."
    )
    # HOLD guardrail (successor to legacy NEUTRAL)
    assert "hold" in pm_desc_lower, "pm_decision must discuss HOLD explicitly."


def test_pm_decision_has_no_backtest_language():
    """Backtest was removed from the DECISION reasoning entirely (the
    user reported it was over-weighting past statistics, diluting the
    forward thesis, and dragging Indian growth names to HOLD).  The M6
    Backtest tab remains as independent validation, but the PM prompt
    must no longer reference backtest / expectancy / hit-rate at all —
    the decision is anchored purely on the forward analyst evidence."""
    tasks = _load_tasks()
    low = tasks["pm_decision"]["description"].lower()
    for term in ("backtest", "expectancy", "hit-rate", "hit rate"):
        assert term not in low, (
            f"pm_decision must not reference '{term}' — backtest was "
            "removed from decision reasoning; it belongs to the M6 tab "
            "only."
        )
    # The forward-thesis anchor must still be present.
    assert "character" in low, (
        "pm_decision must still pick the horizon from the CHARACTER of "
        "the analyst thesis."
    )


def test_trader_task_has_no_backtest_language_and_uses_character():
    """The trader was the source of the horizon downgrade: whenever the
    LONG-tier historical backtest was unfavourable, the old prompt
    switched to MEDIUM.  Backtest has now been removed from the trader
    toolset AND prompt, so the horizon is driven purely by the
    CHARACTER of the forward thesis."""
    tasks = _load_tasks()
    low = tasks["trader_task"]["description"].lower()
    for term in ("backtest", "expectancy", "hit-rate", "hit rate"):
        assert term not in low, (
            f"trader_task must not reference '{term}' — backtest_setup "
            "was removed from the trader toolset and prompt."
        )
    assert "character" in low, (
        "trader_task must instruct the trader to pick the horizon from "
        "the CHARACTER of the analyst evidence."
    )


def test_backtest_setup_removed_from_trader_toolset():
    """backtest_setup must no longer be one of the trader's tools — it
    was demoting structurally-strong names to HOLD via past-heavy
    statistics.  The @tool definition and the M6 walk-forward engine
    remain for the Backtest tab, but the decision agents must not call
    it."""
    from trading_crew.tools import DEFAULT_AGENT_TOOLS
    assert "backtest_setup" not in DEFAULT_AGENT_TOOLS["trader"], (
        "backtest_setup must be removed from the trader's toolset so "
        "the decision reasoning no longer anchors on historical "
        "price-simulation statistics."
    )
    # The tool itself still exists (M6 Backtest tab depends on it).
    from trading_crew.tools import backtest_setup  # noqa: F401


def test_pm_decision_enumerates_five_tier_action_alphabet():
    """The PM prompt must enumerate the exact 5-tier action alphabet
    (``STRONG_BUY`` / ``BUY`` / ``HOLD`` / ``SELL`` / ``STRONG_SELL``)
    so the LLM knows which strings are legal.  Without this the model
    can hallucinate the legacy ``OVERWEIGHT`` / ``UNDERWEIGHT`` labels
    (which the schema still accepts for backwards compatibility but
    silently rewrites).
    """
    tasks = _load_tasks()
    pm_desc = tasks["pm_decision"]["description"]
    for tier in ("STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"):
        assert tier in pm_desc, (
            f"pm_decision prompt must enumerate the ``{tier}`` action "
            "so the LLM emits valid tier strings."
        )
    # The prompt should also explicitly discourage the legacy labels
    # so agents don't silently emit them.
    pm_desc_low = pm_desc.lower()
    assert "do not emit the legacy labels" in pm_desc_low or (
        "overweight" in pm_desc_low and "legacy" in pm_desc_low
    ), (
        "pm_decision prompt should tell the LLM not to emit the legacy "
        "OVERWEIGHT / NEUTRAL / UNDERWEIGHT labels."
    )


def test_debate_rounds_require_rebuttal_and_conviction_shift():
    """Round-N (N>1) bull/bear/risk turns must REBUTTAL: a specific
    opposing bullet AND state a Conviction-vs-prior-round line, so
    each round is materially different from the one before."""
    tasks = _load_tasks()
    for name in ("bull_round", "bear_round",
                 "risk_aggressive_round", "risk_conservative_round"):
        desc = tasks[name]["description"]
        out = tasks[name]["expected_output"]
        assert "REBUTTAL:" in desc, f"{name} must mention a REBUTTAL: opener"
        assert "Conviction vs prior round" in desc, (
            f"{name} must require a 'Conviction vs prior round:' line"
        )
        assert "REBUTTAL:" in out, f"{name} expected_output must mention REBUTTAL:"
    # Neutral synthesises rather than rebuts — it must surface a
    # size-shift line instead.
    neut_desc = tasks["risk_neutral_round"]["description"]
    assert "Recommended size shift vs prior round" in neut_desc


def test_bear_round_one_is_independent_opening():
    """Round-1 bear is a first thought in parallel with bull, not a rebuttal."""
    tasks = _load_tasks()
    desc = tasks["bear_round"]["description"]
    out = tasks["bear_round"]["expected_output"]
    assert "If round_num == 1" in desc
    assert "independent first bearish case" in desc
    assert "Do NOT open with ``REBUTTAL:``" in desc
    assert "the bull turn that just ran" not in desc
    assert "Explicitly rebut the BULL turn" not in desc
    assert "for round_num == 1" in out
    assert "independent opening" in out


def test_bundled_debate_tasks_require_mid_exchange_tools():
    tasks = _load_tasks()
    for name in ("debate_bull_open", "debate_bear_rebut", "debate_bull_close"):
        desc = tasks[name]["description"]
        assert "research_web" in desc, f"{name} must allow research_web"
        assert "{ticker}" in desc
    assert "REBUTTAL:" in tasks["debate_bear_rebut"]["description"]
    assert "Conviction vs opening" in tasks["debate_bull_close"]["description"]


def test_bundled_risk_tasks_are_two_rounds_with_tools():
    tasks = _load_tasks()
    for name in (
        "risk_aggressive_open",
        "risk_conservative_open",
        "risk_neutral_open",
        "risk_aggressive_rebut",
        "risk_conservative_rebut",
        "risk_neutral_rebut",
    ):
        desc = tasks[name]["description"]
        assert "research_web" in desc, f"{name} must allow research_web"
        assert "{ticker}" in desc
    for name in (
        "risk_aggressive_open",
        "risk_conservative_open",
        "risk_neutral_open",
    ):
        assert "Do NOT open with ``REBUTTAL:``" in tasks[name]["description"], name
    for name in (
        "risk_aggressive_rebut",
        "risk_conservative_rebut",
        "risk_neutral_rebut",
    ):
        assert "REBUTTAL:" in tasks[name]["description"], name
        assert "Conviction vs opening" in tasks[name]["description"], name


def test_risk_round_one_is_independent_opening():
    """Round-1 aggressive/conservative are first thoughts, not a rebuttal."""
    tasks = _load_tasks()
    for name, first_case in (
        ("risk_aggressive_round", "independent first size-up case"),
        ("risk_conservative_round", "independent first size-down / skip case"),
    ):
        desc = tasks[name]["description"]
        out = tasks[name]["expected_output"]
        assert "If round_num == 1" in desc, name
        assert first_case in desc, name
        assert "Do NOT open with ``REBUTTAL:``" in desc, name
        assert "the AGGRESSIVE turn that just ran" not in desc, name
        assert "for round_num == 1" in out, name
        assert "independent opening" in out, name


def test_news_task_keeps_corporate_actions_first_tier():
    """Corporate actions returned by the new tool MUST be treated as
    first-tier catalysts in the news report (named in the expected
    output) — that's the explicit user feedback after the buyback
    miss."""
    tasks = _load_tasks()
    desc = tasks["news_task"]["description"].lower()
    out = tasks["news_task"]["expected_output"].lower()
    assert "first-tier" in desc or "first-class" in desc, (
        "news_task must treat corporate actions as a first-tier catalyst"
    )
    assert "corporate actions" in out, (
        "news_task expected_output must require a Corporate Actions section"
    )


# ---------------------------------------------------------------------------
# Anti-fabrication + conviction-retune contracts
# ---------------------------------------------------------------------------


def test_qualitative_analyst_tasks_allow_analyst_judgement_label():
    """The qualitative analyst tasks must offer the honest
    ``[analyst judgement]`` label for claims with no citable URL AND
    explicitly forbid inventing a placeholder URL (example.com).  This
    is the direct fix for the 'too many unverified sources' pathology:
    the model was fabricating URLs to satisfy the provenance rule."""
    tasks = _load_tasks()
    for name in ANALYST_TASKS:
        low = tasks[name]["description"].lower()
        assert "[analyst judgement]" in low, (
            f"{name} must offer the honest '[analyst judgement]' label "
            "for un-citable qualitative claims."
        )
        assert "example.com" in low, (
            f"{name} must explicitly forbid fabricating a placeholder "
            "URL (example.com) to satisfy the provenance rule."
        )


def test_research_synthesis_copies_source_tags_verbatim():
    tasks = _load_tasks()
    low = tasks["research_synthesis"]["description"].lower()
    assert "verbatim" in low
    assert "do not drop the fact" in low


def test_quality_review_accepts_analyst_judgement_as_supported():
    """A claim labelled ``[analyst judgement]`` must count as
    SUPPORTED-qualitative in the quality audit — not a fatal
    UNSUPPORTED gap that forces a size-down."""
    tasks = _load_tasks()
    low = tasks["quality_review"]["description"].lower()
    assert "analyst judgement" in low, (
        "quality_review must recognise the '[analyst judgement]' label."
    )
    assert "supported-qualitative" in low, (
        "quality_review must treat a labelled judgement as "
        "SUPPORTED-qualitative, not UNSUPPORTED."
    )
    # A fabricated URL, by contrast, is UNSUPPORTED.
    assert "fabricated" in low or "placeholder" in low, (
        "quality_review must flag a fabricated / placeholder URL as "
        "UNSUPPORTED."
    )
    assert "do not require" in low and "reappear" in low, (
        "quality_review must not demand the URL also appear in the "
        "analyst pack (truncated window)."
    )
    assert "do not tell the trader or pm to omit" in low, (
        "quality_review must not instruct the desk to drop tagged claims."
    )


def test_pm_keeps_tagged_claims_qr_mislabeled():
    """COCHINSHIP QR marked real news URLs UNSUPPORTED; the PM must
    keep those tagged facts."""
    tasks = _load_tasks()
    low = tasks["pm_decision"]["description"].lower()
    assert "do not drop" in low
    assert "unsupported" in low
    assert "example.com" in low


def test_compliance_review_reserves_blocked_for_concrete_hits():
    """An unverifiable exposure claim must right-size (FLAGGED), never
    force a BLOCKED / HOLD.  BLOCKED is reserved for a concrete
    sanctioned / restricted-list hit."""
    tasks = _load_tasks()
    low = tasks["compliance_review"]["description"].lower()
    assert "unverifiable" in low, (
        "compliance_review must address the unverifiable-exposure case."
    )
    assert "flagged" in low and "blocked" in low, (
        "compliance_review must distinguish FLAGGED (size-cap) from "
        "BLOCKED (concrete hit)."
    )


def test_pm_decision_hold_reserved_for_contradictory_evidence():
    """HOLD must cover mixed/split desks as well as contradictory
    fundamentals; a well-reasoned qualitative thesis still stands on its
    own and a missing / unverifiable source downgrades one claim, not the
    whole thesis."""
    tasks = _load_tasks()
    low = tasks["pm_decision"]["description"].lower()
    assert "split" in low and "hold" in low, (
        "pm_decision must HOLD on a split desk-bias board, not default BUY."
    )
    assert "qualitative thesis stands on its own" in low, (
        "pm_decision must state that a well-reasoned qualitative thesis "
        "stands on its own."
    )
    # The honest label is acknowledged and fabrication is forbidden.
    assert "[analyst judgement]" in low, (
        "pm_decision must acknowledge the '[analyst judgement]' label."
    )
    assert "example.com" in low, (
        "pm_decision must forbid fabricating a placeholder URL."
    )


def test_pm_decision_grounds_entry_to_last_close():
    """The NTNX card showed ENTRY=100 on a ~$68 print because the PM
    copied a trader placeholder. The prompt must prefer LAST_CLOSE
    over an implausible trader entry."""
    tasks = _load_tasks()
    low = tasks["pm_decision"]["description"].lower()
    assert "last_close" in low, (
        "pm_decision must name LAST_CLOSE so the injected print is used."
    )
    assert "placeholder" in low or "invent" in low
    assert "few percent" in low or "tens of percent" in low


def test_pm_and_rm_name_all_five_desks_including_reports():
    """Reports/guidance was added as a fifth desk but RM/PM prompts still
    said 'four analyst reports' and dropped it from the named list — the
    filing card then never had to be weighed."""
    tasks = _load_tasks()
    for name in ("research_synthesis", "pm_decision"):
        low = tasks[name]["description"].lower()
        assert "five" in low or "5" in low, f"{name} must name five desks"
        assert "reports" in low and "guidance" in low, (
            f"{name} must name the reports/guidance desk"
        )
        assert "four analyst" not in low, (
            f"{name} still says 'four analyst reports' — reports desk dropped"
        )


def test_fundamentals_desk_owns_live_peer_set():
    """Peers are a live screen + optional peers=, not a ticker dictionary."""
    tasks = _load_tasks()
    desc = tasks["fundamentals_task"]["description"]
    low = desc.lower()
    assert "get_sector_peers({ticker})" in desc
    assert "peers=" in desc
    assert "ticker dictionary" in low
    assert "do not invent tickers" in low
    assert "second time" in low
    assert "empty argument object" in low


def test_news_and_macro_desks_write_their_own_search_query():
    """Self-service contract: analysts compose query= from role + brief
    + gap. A canned industry seed list is not a scalable substitute."""
    tasks = _load_tasks()
    news = tasks["news_task"]["description"]
    macro = tasks["macro_task"]["description"]
    assert "query=" in news and "get_news" in news
    assert "query=" in macro
    assert "get_geopolitical_news" in macro
    assert "get_supply_chain_risk" in macro
    assert "write a 6-12 word" in macro.lower()


def test_prompts_have_no_country_cycle_prior():
    """A standing 'India is in an upcycle' prior systematically tilts
    Indian names to BUY.  Any cycle call must come from the macro desk."""
    tasks = _load_tasks()
    joined = "\n".join(
        (t.get("description") or "") + "\n" + (t.get("expected_output") or "")
        for t in tasks.values()
    ).lower()
    assert "india especially" not in joined
    assert "india in particular is in" not in joined
    # Anti-prior language (do not assume an upcycle) is allowed; a
    # standing claim that a country IS in an upcycle is not.
    assert "is in an investment / production" not in joined
