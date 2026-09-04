"""TradingCrew — the lean multi-agent workflow assembled from yaml.

Layout note (post-consolidation)
--------------------------------
The analyst team was consolidated from eight narrow desks to four wider
ones (market+quant, news+social, fundamentals+sector, macro+geopolitical)
to cut redundant LLM tool-loops and reduce information dilution; a fifth
desk — the Reports Analyst (primary-filing guidance / target-vs-actual) —
was later added back deliberately, since nothing else reads the company's
own concall / annual-report / rating filings.  The
Quality Reviewer and Compliance Officer are now OPTIONAL (toggle-gated,
default off) and, when enabled, run ASYNC as advisory inputs to the
Portfolio Manager rather than as sequential gates in front of the
trader — so they overlap the trader/risk stretch instead of extending
the critical path.

Layout
------
* Personas live in ``config/agents.yaml`` (role / goal / backstory only).
* Task descriptions and expected outputs live in ``config/tasks.yaml``.
* Tools, debate-round wiring, ``async_execution``, ``context=[...]``,
  ``output_pydantic`` and ``guardrail`` are all wired in this file —
  they are workflow concerns, not prompts.

Public API::

    TradingCrew(
        ticker="NTNX",
        debate_rounds=2,
        risk_rounds=1,
        step_callback=None,        # called per agent step
        task_callback=None,        # called per task completion
    ).crew().kickoff(inputs={"ticker": "NTNX"})
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from . import tools as T
from ._common import get_embedder_config, get_llm, get_memory_llm
from .guardrails import confidence_guardrail
from .schemas import PortfolioDecision
from .tools import ALL_TOOLS, DEFAULT_AGENT_TOOLS


# Desks whose job is to AGGREGATE / DECIDE on the evidence rather than to
# generate diverse arguments.  These run near-deterministic (low temperature)
# so the SAME upstream evidence yields the SAME call — the ~32% intra-day
# BUY<->SELL flip was largely sampling noise at the aggregation step, not new
# information.  Exploration temperature is retained for the analyst + bull/bear
# + aggressive/conservative desks, where a diversity of angles is the point.
_DECISION_AGENTS = {"research_manager", "trader", "risk_neutral", "portfolio_manager"}

# Synthesis / aggregation desks: disable Qwen3 ``<think>`` (they consume
# assembled evidence; a 10-minute monologue does not improve the call).
# Analysts + bull/bear keep thinking so argument diversity stays intact.
_SYNTHESIS_NO_THINK = {
    "research_manager",
    "trader",
    "portfolio_manager",
    "risk_aggressive",
    "risk_neutral",
    "risk_conservative",
}

# Desks that get CrewAI's plan-then-act-then-observe loop (``planning=True`` +
# ``PlanningConfig``).  Only the SYNTHESIS / DECISION desks: they reason over
# the assembled evidence, where an explicit plan + post-step observation
# measurably helps.  We deliberately EXCLUDE the tool-heavy analysts (they
# already ReAct through their tool loops, and adding a planner on top invites
# the degraded-tool-loop failure mode this crew already fights) and the
# bull/bear/risk debaters (their value is argument diversity, not planning).
# Planning is internal to the agent, so it never enlarges downstream context.
_PLANNING_AGENTS = {"research_manager", "trader", "portfolio_manager"}

# agent_key → display role string (matches the workflow diagram's node ids and
# the runner's role layout).  Used to tell the UI which tiles run ReAct.
_PLANNING_AGENT_ROLES = {
    "research_manager": "Research Manager",
    "trader": "Trader",
    "portfolio_manager": "Portfolio Manager",
}


def planning_agent_roles() -> list:
    """Display role strings for the desks running ReAct-style planning RIGHT
    NOW — empty when ``TRADINGCREW_DECISION_PLANNING`` is off.  Drives the UI's
    ReAct tile badge so it reflects the live config, never a hardcoded guess."""
    if not _planning_enabled():
        return []
    return [
        role for key, role in _PLANNING_AGENT_ROLES.items() if key in _PLANNING_AGENTS
    ]


def _planning_enabled() -> bool:
    """ReAct-style planning on the decision desks.  Batch job → default ON;
    flip ``TRADINGCREW_DECISION_PLANNING=0`` to A/B or if a backend chokes on
    the extra planner calls."""
    raw = os.environ.get("TRADINGCREW_DECISION_PLANNING", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _make_planning_config():
    """Build a ``PlanningConfig`` for the decision desks, or ``None`` if the
    installed CrewAI doesn't expose it (defensive import — never a runtime
    fallback that hides a misconfig)."""
    try:
        from crewai.agent.planning_config import PlanningConfig
    except Exception:  # noqa: BLE001 — older CrewAI without the planner
        return None
    effort = (os.environ.get("TRADINGCREW_PLANNING_EFFORT", "low") or "low").strip().lower()
    if effort not in ("low", "medium", "high"):
        effort = "low"
    kwargs: dict = dict(
        reasoning_effort=effort,
        # Caps keep a single name bounded even though the batch has no time
        # limit: a planner that loops or hangs must not stall the whole collect.
        max_replans=int(os.environ.get("TRADINGCREW_PLANNING_MAX_REPLANS", "2")),
        step_timeout=int(os.environ.get("TRADINGCREW_PLANNING_STEP_TIMEOUT", "900")),
    )
    # Per-step PlannerObserver LLM call.  DEFAULT OFF: our OSS model can't emit
    # CrewAI's nested ``StepObservation`` schema (``suggested_refinements`` must
    # be ``[{step_number, new_description}]`` objects — the model returns bare
    # strings / ``{step, action}`` dicts), so every observation fails schema
    # validation, is retried 2-3×, and falls back to "continue as-is".  That's
    # non-fatal (no tool re-run, no replan) but pure wasted tokens/latency at
    # batch scale.  We keep the useful plan→act loop and drop the broken
    # observe.  Flip ``TRADINGCREW_PLANNING_OBSERVE_STEPS=1`` only on a model
    # that reliably produces structured observations.  Guarded so an older
    # CrewAI without the field still constructs.
    observe = os.environ.get("TRADINGCREW_PLANNING_OBSERVE_STEPS", "0")
    observe_on = observe.strip().lower() not in ("0", "false", "no", "off", "")
    if "observe_steps" in getattr(PlanningConfig, "model_fields", {}):
        kwargs["observe_steps"] = observe_on
    return PlanningConfig(**kwargs)


@CrewBase
class TradingCrew:
    """Multi-agent stock-research crew built from yaml prompts."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ---- ctor (configures non-prompt knobs) --------------------------------

    def __init__(
        self,
        ticker: str = "NTNX",
        debate_rounds: int = 2,
        risk_rounds: int = 1,
        step_callback: Optional[Callable[[Any], None]] = None,
        task_callback: Optional[Callable[[Any], None]] = None,
        memory: bool = False,
        tools_enabled: Optional[Dict[str, List[str]]] = None,
        quality_review_enabled: bool = False,
        compliance_enabled: bool = False,
        run_mode: str = "autonomous",
    ) -> None:
        self.ticker = ticker
        self.debate_rounds = debate_rounds
        self.risk_rounds = risk_rounds
        self._step_callback = step_callback
        self._task_callback = task_callback
        self._use_memory = memory
        mode = (run_mode or "autonomous").strip().lower()
        if mode == "deterministic":
            self.run_mode = "deterministic"
        elif mode == "harness":
            self.run_mode = "harness"
        else:
            self.run_mode = "autonomous"
        # Optional advisory reviewers.  Default OFF so a normal run is
        # lean; when ON they run async and feed the PM (see @crew).
        self._quality_review_enabled = bool(quality_review_enabled)
        self._compliance_enabled = bool(compliance_enabled)
        # Per-agent tool whitelist. Falls back to DEFAULT_AGENT_TOOLS when an
        # agent key is missing — so an empty dict gets default behaviour.
        self._tools_enabled: Dict[str, List[str]] = tools_enabled or {}
        # Exploration temperature for the analyst / debate desks (diverse
        # angles are wanted there).  Decision/synthesis desks (see
        # ``_DECISION_AGENTS``) run at a near-zero temperature so the same
        # evidence reproduces the same call — both env-overridable.
        self._debate_temp = float(os.environ.get("TRADINGCREW_DEBATE_TEMPERATURE", "0.3"))
        self._decision_temp = float(os.environ.get("TRADINGCREW_DECISION_TEMPERATURE", "0.0"))
        # ReAct-style planning on the decision desks (resolved once per crew).
        self._planning_enabled = _planning_enabled()
        self._llm = get_llm(temperature=self._debate_temp)
        self._decision_llm: object | None = None
        self._risk_no_think_llm: object | None = None
        # Phase 2E — per-agent LLM cache.  We look up
        # ``LLM_PER_AGENT[agent_key]`` lazily and cache the result so
        # the same persona doesn't pay LLM-construction cost twice.
        self._llm_cache: Dict[str, object] = {}

    # ---- tool resolution --------------------------------------------------

    def _tools_for(self, agent_key: str) -> list:
        """Return the @tool-decorated callables for an agent, honouring the
        UI's per-agent enable/disable toggles."""
        defaults = DEFAULT_AGENT_TOOLS.get(agent_key, [])
        if agent_key in self._tools_enabled:
            allowed = [t for t in self._tools_enabled[agent_key] if t in defaults]
            # If the user disabled every tool, drop the tools list entirely
            # rather than handing the agent an empty array.
            names = allowed
        else:
            names = defaults
        return [ALL_TOOLS[n] for n in names if n in ALL_TOOLS]

    # ---- shared agent kwargs ----------------------------------------------

    def _llm_for(self, agent_key: str | None):
        """Return the LLM for ``agent_key``, honouring ``LLM_PER_AGENT``.

        Falls back to the global ``self._llm`` when no override is
        configured for this agent — cheap, single-allocation path.
        """
        if not agent_key:
            return self._llm
        # We only build a dedicated LLM when an override exists; the
        # check has to look at the raw env var because ``get_llm`` would
        # otherwise return an instance that's identical to the global.
        from . import _common
        overrides = _common._load_per_agent_overrides().get(agent_key)
        if overrides:
            if agent_key in self._llm_cache:
                return self._llm_cache[agent_key]
            # An explicit per-agent override wins outright (it may set its own
            # temperature); the debate temp is only the starting default.
            llm = get_llm(
                temperature=self._debate_temp,
                agent_key=agent_key,
                disable_thinking=agent_key in _SYNTHESIS_NO_THINK,
            )
            self._llm_cache[agent_key] = llm
            return llm
        # No explicit override: decision/synthesis desks run near-deterministic;
        # everyone else keeps the exploration temperature.
        if agent_key in _DECISION_AGENTS:
            if self._decision_llm is None:
                self._decision_llm = get_llm(
                    temperature=self._decision_temp,
                    disable_thinking=True,
                )
            return self._decision_llm
        if agent_key in _SYNTHESIS_NO_THINK:
            if self._risk_no_think_llm is None:
                self._risk_no_think_llm = get_llm(
                    temperature=self._debate_temp,
                    disable_thinking=True,
                )
            return self._risk_no_think_llm
        return self._llm

    def _agent_kwargs(self, max_iter: int = 6, agent_key: str | None = None) -> dict:
        kwargs = dict(
            llm=self._llm_for(agent_key),
            allow_delegation=False,
            max_iter=max_iter,
            # Auto-summarize prior messages if the prompt would otherwise
            # exceed the LLM's context window. Without this, CrewAI passes
            # the full chain forward and either gets truncated silently or
            # the server hangs / 400s on a too-long request.
            respect_context_window=True,
            verbose=True,
        )
        # ReAct-style plan/observe loop on the synthesis + decision desks only
        # (internal to the agent — does not enlarge downstream context).
        if agent_key in _PLANNING_AGENTS and self._planning_enabled:
            pc = _make_planning_config()
            if pc is not None:
                kwargs["planning"] = True
                kwargs["planning_config"] = pc
        return kwargs

    def _apply_harness(self, agents) -> None:
        """Prepend the Hermes harness (strategy, open experiment, optional desk addendum)."""
        from trading_crew.agentic.harness import apply_harness

        apply_harness(agents, book="equity", run_mode=self.run_mode)

    def materialize_agents(self) -> Dict[str, Agent]:
        """Build stock-desk agents for the LangGraph scheduler (no sequential crew)."""
        agents: Dict[str, Agent] = {
            "market_analyst": self.market_analyst(),
            "news_analyst": self.news_analyst(),
            "fundamentals_analyst": self.fundamentals_analyst(),
            "reports_analyst": self.reports_analyst(),
            "macro_analyst": self.macro_analyst(),
            "bull_researcher": self.bull_researcher(),
            "bear_researcher": self.bear_researcher(),
            "research_manager": self.research_manager(),
            "trader": self.trader(),
            "risk_aggressive": self.risk_aggressive(),
            "risk_conservative": self.risk_conservative(),
            "risk_neutral": self.risk_neutral(),
            "portfolio_manager": self.portfolio_manager(),
        }
        if self._quality_review_enabled:
            agents["quality_reviewer"] = self.quality_reviewer()
        if self._compliance_enabled:
            agents["compliance_officer"] = self.compliance_officer()
        self._apply_harness(agents)
        pm = Task(
            config=self.tasks_config["pm_decision"],
            agent=agents["portfolio_manager"],
            output_pydantic=PortfolioDecision,
            guardrail=confidence_guardrail,
            guardrail_max_retries=2,
        )
        self._pm_task = pm
        self._materialized = agents
        return agents

    # =======================================================================
    # Agents — one @agent per persona, all read role/goal/backstory from yaml
    # =======================================================================

    # ---- Analyst Team -----------------------------------------------------

    @agent
    def market_analyst(self) -> Agent:
        # Merged market + quant desk: price/technicals + options/positioning.
        return Agent(
            config=self.agents_config["market_analyst"],
            tools=self._tools_for("market_analyst"),
            **self._agent_kwargs(max_iter=10, agent_key="market_analyst"),
        )

    @agent
    def news_analyst(self) -> Agent:
        # Merged news + social desk.  max_iter sized for the 5-tool budget
        # in news_task (each tool call burns roughly 2 reasoning
        # iterations).  Keep just enough headroom for one retry; tighter
        # than that and OSS-120B can't finish; looser and it gets stuck
        # in tool-call loops on Indian tickers (the LT.NS / BHARATFORG.NS
        # / BAJAJ-AUTO.NS degraded-output pattern).
        return Agent(
            config=self.agents_config["news_analyst"],
            tools=self._tools_for("news_analyst"),
            # 6-tool budget now (added research_web); each tool call burns
            # ~2 reasoning iterations, so give one extra pair of headroom.
            **self._agent_kwargs(max_iter=14, agent_key="news_analyst"),
        )

    @agent
    def fundamentals_analyst(self) -> Agent:
        # Merged fundamentals + sector desk: financials + long-term thesis
        # + peer relative strength (7 tools).
        return Agent(
            config=self.agents_config["fundamentals_analyst"],
            tools=self._tools_for("fundamentals_analyst"),
            **self._agent_kwargs(max_iter=12, agent_key="fundamentals_analyst"),
        )

    @agent
    def reports_analyst(self) -> Agent:
        # Primary-filing desk: reads the company's own concall transcripts /
        # investor decks / annual-report MD&A / credit ratings for management
        # guidance + target-vs-actual (3-tool budget; PDFs pre-summarised).
        return Agent(
            config=self.agents_config["reports_analyst"],
            tools=self._tools_for("reports_analyst"),
            **self._agent_kwargs(max_iter=8, agent_key="reports_analyst"),
        )

    @agent
    def macro_analyst(self) -> Agent:
        # Merged macro + geopolitical desk: macro regime + geopolitical +
        # supply-chain (5 tools).
        return Agent(
            config=self.agents_config["macro_analyst"],
            tools=self._tools_for("macro_analyst"),
            **self._agent_kwargs(max_iter=10, agent_key="macro_analyst"),
        )

    # ---- Researcher Team --------------------------------------------------

    @agent
    def bull_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["bull_researcher"],
            tools=self._tools_for("bull_researcher"),
            **self._agent_kwargs(max_iter=12, agent_key="bull_researcher"),
        )

    @agent
    def bear_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["bear_researcher"],
            tools=self._tools_for("bear_researcher"),
            **self._agent_kwargs(max_iter=12, agent_key="bear_researcher"),
        )

    @agent
    def research_manager(self) -> Agent:
        return Agent(
            config=self.agents_config["research_manager"],
            tools=self._tools_for("research_manager"),
            **self._agent_kwargs(agent_key="research_manager"),
        )

    # ---- Quality Reviewer -------------------------------------------------

    @agent
    def quality_reviewer(self) -> Agent:
        return Agent(config=self.agents_config["quality_reviewer"], **self._agent_kwargs(agent_key="quality_reviewer"))

    # ---- Trader -----------------------------------------------------------

    @agent
    def trader(self) -> Agent:
        return Agent(
            config=self.agents_config["trader"],
            tools=self._tools_for("trader"),
            **self._agent_kwargs(max_iter=10, agent_key="trader"),
        )

    # ---- Risk Team --------------------------------------------------------

    @agent
    def risk_aggressive(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_aggressive"],
            tools=self._tools_for("risk_aggressive"),
            **self._agent_kwargs(max_iter=12, agent_key="risk_aggressive"),
        )

    @agent
    def risk_neutral(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_neutral"],
            tools=self._tools_for("risk_neutral"),
            **self._agent_kwargs(max_iter=12, agent_key="risk_neutral"),
        )

    @agent
    def risk_conservative(self) -> Agent:
        return Agent(
            config=self.agents_config["risk_conservative"],
            tools=self._tools_for("risk_conservative"),
            **self._agent_kwargs(max_iter=12, agent_key="risk_conservative"),
        )

    # ---- Compliance + PM --------------------------------------------------

    @agent
    def compliance_officer(self) -> Agent:
        return Agent(config=self.agents_config["compliance_officer"], **self._agent_kwargs(agent_key="compliance_officer"))

    @agent
    def portfolio_manager(self) -> Agent:
        return Agent(config=self.agents_config["portfolio_manager"], **self._agent_kwargs(agent_key="portfolio_manager"))

    # =======================================================================
    # Tasks — analyst phase is fixed, debate/risk phases are generated
    # programmatically in the @crew method below (variable round counts).
    # =======================================================================

    @task
    def market_task(self) -> Task:
        return Task(
            config=self.tasks_config["market_task"],
            agent=self.market_analyst(),
            async_execution=True,
        )

    @task
    def news_task(self) -> Task:
        return Task(
            config=self.tasks_config["news_task"],
            agent=self.news_analyst(),
            async_execution=True,
        )

    @task
    def fundamentals_task(self) -> Task:
        return Task(
            config=self.tasks_config["fundamentals_task"],
            agent=self.fundamentals_analyst(),
            async_execution=True,
        )

    @task
    def reports_task(self) -> Task:
        return Task(
            config=self.tasks_config["reports_task"],
            agent=self.reports_analyst(),
            async_execution=True,
        )

    @task
    def macro_task(self) -> Task:
        return Task(
            config=self.tasks_config["macro_task"],
            agent=self.macro_analyst(),
            async_execution=True,
        )

    # =======================================================================
    # Crew — wires everything together, including the dynamic debate rounds.
    # =======================================================================

    @crew
    def crew(self) -> Crew:
        analyst_tasks = [
            self.market_task(),
            self.news_task(),
            self.fundamentals_task(),
            self.reports_task(),
            self.macro_task(),
        ]

        # ---- Bull / Bear debate ------------------------------------------
        # Same as v2: sequential process drains the async analyst batch
        # at the first sync debate task. Debate prompts do NOT receive
        # those reports (or prior rounds) as CrewAI context — they share
        # a compact scratch pad and fetch new evidence with tools.
        bull_cfg = self.tasks_config["bull_round"]
        bear_cfg = self.tasks_config["bear_round"]
        debate_tasks: List[Task] = []
        last_bull: Optional[Task] = None
        last_bear: Optional[Task] = None
        for r in range(1, self.debate_rounds + 1):
            # Round-1 prompts are independent openings (not a bear
            # rebuttal). CrewAI Process.sequential will not let an async
            # task take other async tasks as context unless a sync task
            # sits between them. The five analysts are already an async
            # batch, so marking bull/bear async (simultaneous openings)
            # raises ValidationError at Crew() — that is what broke the
            # NTNX run. Autonomous LangGraph runs the openings in
            # parallel; this sequential crew keeps them sync.
            bull_t = Task(
                description=bull_cfg["description"].format(
                    ticker=self.ticker, round_num=r, total_rounds=self.debate_rounds
                ),
                expected_output=bull_cfg["expected_output"].format(
                    ticker=self.ticker, round_num=r, total_rounds=self.debate_rounds
                ),
                agent=self.bull_researcher(),
                context=[],
            )
            bear_t = Task(
                description=bear_cfg["description"].format(
                    ticker=self.ticker, round_num=r, total_rounds=self.debate_rounds
                ),
                expected_output=bear_cfg["expected_output"].format(
                    ticker=self.ticker, round_num=r, total_rounds=self.debate_rounds
                ),
                agent=self.bear_researcher(),
                context=[],
            )
            debate_tasks += [bull_t, bear_t]
            last_bull, last_bear = bull_t, bear_t

        # ---- Research Manager synthesis ----------------------------------
        synthesis_ctx = [t for t in (last_bull, last_bear) if t is not None]
        research_synthesis = Task(
            config=self.tasks_config["research_synthesis"],
            agent=self.research_manager(),
            context=synthesis_ctx,
        )

        # ---- Quality Review (optional, ASYNC advisory to the PM) ---------
        # When enabled we mark it ``async_execution=True`` and DO NOT wire
        # it into the trader's context.  It only depends on the research
        # synthesis, so it fans out in parallel with the trader/risk
        # stretch and is awaited only by the PM (which lists it in
        # context).  This keeps it off the critical path.
        quality_review: Optional[Task] = None
        if self._quality_review_enabled:
            quality_review = Task(
                config=self.tasks_config["quality_review"],
                agent=self.quality_reviewer(),
                context=[research_synthesis],
                async_execution=True,
            )

        # ---- Trader ------------------------------------------------------
        trader_t = Task(
            config=self.tasks_config["trader_task"],
            agent=self.trader(),
            context=[research_synthesis],
        )

        # ---- Risk debate -------------------------------------------------
        aggr_cfg = self.tasks_config["risk_aggressive_round"]
        cons_cfg = self.tasks_config["risk_conservative_round"]
        neut_cfg = self.tasks_config["risk_neutral_round"]
        risk_tasks: List[Task] = []
        risk_ctx: List[Task] = [trader_t, research_synthesis]
        for r in range(1, self.risk_rounds + 1):
            # Round 1: both wings are independent first thoughts (parallel).
            # Round 2+: Conservative waits on this-round Aggressive so it
            # can rebut — same contract as later bull→bear turns.
            opening = r == 1
            aggr = Task(
                description=aggr_cfg["description"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                expected_output=aggr_cfg["expected_output"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                agent=self.risk_aggressive(),
                context=list(risk_ctx),
                async_execution=opening,
            )
            cons = Task(
                description=cons_cfg["description"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                expected_output=cons_cfg["expected_output"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                agent=self.risk_conservative(),
                context=list(risk_ctx) if opening else list(risk_ctx) + [aggr],
                async_execution=opening,
            )
            neut = Task(
                description=neut_cfg["description"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                expected_output=neut_cfg["expected_output"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                agent=self.risk_neutral(),
                context=list(risk_ctx) + [aggr, cons],
            )
            risk_tasks += [aggr, cons, neut]
            risk_ctx += [aggr, cons, neut]

        # ---- Compliance (optional, advisory to the PM) ------------------
        compliance: Optional[Task] = None
        if self._compliance_enabled:
            compliance = Task(
                config=self.tasks_config["compliance_review"],
                agent=self.compliance_officer(),
                context=[trader_t, research_synthesis] + list(risk_tasks),
                # Async so it overlaps the (also-async) Quality Review
                # when both are on; the PM awaits it.
                async_execution=True,
            )

        # ---- PM ----------------------------------------------------------
        # Give the PM DIRECT sight of the raw analyst reports (TLDRs +
        # bodies) on top of the manager's 3-driver/3-risk synthesis and
        # the risk chain.  Without this the final decision only sees the
        # manager's heavily-compressed thesis, so the non-yfinance lenses
        # (news / geopolitical / macro / sector) get diluted out before
        # the decision is made.  The hosted model's 131K window
        # comfortably fits the extra context.  The optional Quality
        # Review + Compliance advisory tasks (async) are appended when
        # enabled — the PM awaits them here without them gating the
        # trader / risk chain.
        pm_context: List[Task] = list(analyst_tasks) + list(risk_ctx)
        if quality_review is not None:
            pm_context.append(quality_review)
        if compliance is not None:
            pm_context.append(compliance)
        pm = Task(
            config=self.tasks_config["pm_decision"],
            agent=self.portfolio_manager(),
            context=pm_context,
            output_pydantic=PortfolioDecision,
            guardrail=confidence_guardrail,
            guardrail_max_retries=2,
        )
        # Expose the PM task so the driver (runner) can append the
        # deterministically-parsed reports-desk guidance ledger to its prompt
        # once the reports tool has run — guaranteeing that signal reaches the
        # decision even if the analyst's prose dropped it.  CrewAI reads
        # ``task.description`` at execution time (PM runs last), so a mutation
        # made mid-kickoff (before the PM node) takes effect.
        self._pm_task = pm

        # Assemble the task list.  CrewAI's sequential process drains all
        # pending async futures at the next SYNC task, so an async task
        # only truly overlaps its async siblings (not the sync tasks that
        # follow it).  We therefore place the optional Quality Review +
        # Compliance reviewers as a CONSECUTIVE async pair immediately
        # before the PM: when BOTH are enabled they run concurrently with
        # each other and the PM (sync) awaits both — so the critical-path
        # cost of the reviewers is max(QR, compliance) instead of their
        # sum.  The PM stays sync + last, satisfying CrewAI's
        # "end with at most one async task" rule.
        all_tasks: List[Task] = (
            list(analyst_tasks)
            + list(debate_tasks)
            + [research_synthesis, trader_t]
            + list(risk_tasks)
        )
        if quality_review is not None:
            all_tasks.append(quality_review)
        if compliance is not None:
            all_tasks.append(compliance)
        all_tasks.append(pm)

        agents = [
            self.market_analyst(),
            self.news_analyst(),
            self.fundamentals_analyst(),
            self.reports_analyst(),
            self.macro_analyst(),
            self.bull_researcher(), self.bear_researcher(),
            self.research_manager(),
        ]
        if quality_review is not None:
            agents.append(self.quality_reviewer())
        agents.append(self.trader())
        agents += [self.risk_aggressive(), self.risk_neutral(), self.risk_conservative()]
        if compliance is not None:
            agents.append(self.compliance_officer())
        agents.append(self.portfolio_manager())

        self._apply_harness(agents)

        crew_kwargs: dict = dict(
            agents=agents,
            tasks=all_tasks,
            process=Process.sequential,
            verbose=True,
        )

        if self._step_callback is not None:
            crew_kwargs["step_callback"] = self._step_callback
        if self._task_callback is not None:
            crew_kwargs["task_callback"] = self._task_callback
        if self._use_memory:
            # Build a Memory instance ourselves (rather than memory=True)
            # so we can wire BOTH our custom embedder AND the local
            # ``get_llm()`` into CrewAI's analyze subsystem (which runs
            # ``analyze_for_save`` + ``extract_memories_from_content`` +
            # ``analyze_for_consolidation`` on every save and
            # ``analyze_query`` on every ``search_memory`` call).
            #
            # WHY ALSO OVERRIDE ``_save_pool`` — CrewAI's default
            # ``Memory._save_pool`` is a ``ThreadPoolExecutor(max_workers=1)``,
            # which serializes every ``remember()`` call.  Combined with
            # ~5-15 s analyze latency on the local LLM, that turned a
            # 5-minute run into a 30-minute one (~100 sequential analyze
            # calls).  The local LLM endpoint can sustain ~60 req/min;
            # bumping the pool to ``max_workers=8`` lets concurrent
            # ``remember()`` calls fan out so we can actually saturate
            # that 60-req/min budget instead of leaving 90 % of the
            # latency on the table.  EncodingFlow already has an INNER
            # pool of ``max_workers=10`` for items inside one batch — the
            # outer pool override is what lets CONCURRENT batches go
            # through too.  See README "Memory embedder" for the math.
            from concurrent.futures import ThreadPoolExecutor

            from crewai.memory.unified_memory import Memory
            from crewai.memory.utils import sanitize_scope_name

            from . import embedding_presets

            crew_name = f"trading-crew-{self.ticker}"
            crew_kwargs["name"] = crew_name
            crew_kwargs["embedder"] = get_embedder_config()
            # Fold the active embedding-preset id into the root scope.
            # The persisted LanceDB schema (vector dim) is fixed the
            # first time a store is written under a given scope; mixing
            # 768-d and 1536-d embeddings into the same scope crashes
            # the next ``search_memory`` with "query dim doesn't match
            # the column vector dim".  Partitioning by preset id lets
            # the user switch embedders without nuking their entire
            # memory directory.
            preset_id = embedding_presets.get_active_preset_id() or "default"
            mem = Memory(
                # get_memory_llm() returns the same LLM as get_llm() for
                # non-thinking models, but bakes ``enable_thinking=false``
                # into the chat-template kwargs for Qwen3-family models
                # so CrewAI's four internal memory-analyze helpers
                # (analyze_for_save / _consolidation / _query,
                # extract_memories_from_content) don't spend 20-30 k
                # tokens each on chain-of-thought for a two-line JSON
                # classifier response.  See ``trading_crew._common.
                # get_memory_llm`` for the full rationale — this fix
                # avoids the KV-cache saturation that surfaced as
                # ``Memory save analysis failed, using defaults`` after
                # a 10-min OpenAI SDK read timeout in the ZENTEC.NS +
                # Qwen3-32B run.  Main-agent / debate / PM LLMs are
                # unchanged and still get full Qwen3-Thinking quality.
                llm=get_memory_llm(),
                embedder=get_embedder_config(),
                root_scope=(
                    f"/crew/{sanitize_scope_name(crew_name)}"
                    f"/emb/{sanitize_scope_name(preset_id)}"
                ),
                # Disable the LLM-based consolidation pass on save.
                # CrewAI's default ``consolidation_threshold=0.85`` fires
                # an ``analyze_for_consolidation`` LLM call whenever a
                # newly saved memory is semantically similar to an
                # existing record — the LLM is asked to decide whether
                # to merge / update / drop.  For our workflow this is
                # almost always wasted budget: every analyst writes a
                # fresh structured TLDR + body, ticker scope already
                # partitions records by symbol, and we WANT successive
                # runs to keep their distinct snapshots for trend audit
                # rather than be collapsed by the consolidator.
                #
                # Setting the threshold to 1.0 falls into the encoding
                # flow's Group A / C "no consolidation" path — saves
                # one LLM call per ``remember()`` invocation (CrewAI
                # silently fires one per task on memory=True) with
                # zero impact on retrieval quality.  Search behaviour
                # is unchanged; only the redundant merge-pass is gone.
                # Confirmed by reading ``crewai.memory.encoding_flow``
                # decision matrix at the time of this writing.
                consolidation_threshold=1.0,
            )
            # Replace the default 1-worker save pool with a parallel one.
            # We shut the old pool down first (it has no in-flight work
            # because we just constructed Memory) so we don't leak a
            # thread.
            try:
                mem._save_pool.shutdown(wait=False)
            except Exception:  # pragma: no cover - defensive
                pass
            mem._save_pool = ThreadPoolExecutor(
                max_workers=int(os.environ.get("TRADINGCREW_MEMORY_SAVE_WORKERS", "8")),
                thread_name_prefix="memory-save",
            )
            crew_kwargs["memory"] = mem

        return Crew(**crew_kwargs)
