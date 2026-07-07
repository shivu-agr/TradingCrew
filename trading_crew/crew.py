"""TradingCrew — the lean multi-agent workflow assembled from yaml.

Layout note (post-consolidation)
--------------------------------
The analyst team was consolidated from eight narrow desks to FOUR wider
ones (market+quant, news+social, fundamentals+sector, macro+geopolitical)
to cut redundant LLM tool-loops and reduce information dilution.  The
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


def _rlmf_prefix_enabled() -> bool:
    """Return True iff ``TRADINGCREW_RLMF_ENABLED`` is truthy.

    The RLMF (Reinforcement-Learning from Market Feedback) few-shot
    prefix is an experimental hook — we default it to OFF so a fresh
    install with no resolved episodes doesn't quietly change PM
    behaviour.  Operators can flip it on with
    ``TRADINGCREW_RLMF_ENABLED=1`` once they have a case-book of
    resolved decisions worth learning from.
    """
    raw = os.environ.get("TRADINGCREW_RLMF_ENABLED", "0")
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


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
    ) -> None:
        self.ticker = ticker
        self.debate_rounds = debate_rounds
        self.risk_rounds = risk_rounds
        self._step_callback = step_callback
        self._task_callback = task_callback
        self._use_memory = memory
        # Optional advisory reviewers.  Default OFF so a normal run is
        # lean; when ON they run async and feed the PM (see @crew).
        self._quality_review_enabled = bool(quality_review_enabled)
        self._compliance_enabled = bool(compliance_enabled)
        # Per-agent tool whitelist. Falls back to DEFAULT_AGENT_TOOLS when an
        # agent key is missing — so an empty dict gets default behaviour.
        self._tools_enabled: Dict[str, List[str]] = tools_enabled or {}
        self._llm = get_llm(temperature=0.3)
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
        if not overrides:
            return self._llm
        if agent_key in self._llm_cache:
            return self._llm_cache[agent_key]
        llm = get_llm(temperature=0.3, agent_key=agent_key)
        self._llm_cache[agent_key] = llm
        return llm

    def _agent_kwargs(self, max_iter: int = 6, agent_key: str | None = None) -> dict:
        return dict(
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
            **self._agent_kwargs(max_iter=12, agent_key="news_analyst"),
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
        return Agent(config=self.agents_config["bull_researcher"], **self._agent_kwargs(agent_key="bull_researcher"))

    @agent
    def bear_researcher(self) -> Agent:
        return Agent(config=self.agents_config["bear_researcher"], **self._agent_kwargs(agent_key="bear_researcher"))

    @agent
    def research_manager(self) -> Agent:
        return Agent(config=self.agents_config["research_manager"], **self._agent_kwargs(agent_key="research_manager"))

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
        return Agent(config=self.agents_config["risk_aggressive"], **self._agent_kwargs(agent_key="risk_aggressive"))

    @agent
    def risk_neutral(self) -> Agent:
        return Agent(config=self.agents_config["risk_neutral"], **self._agent_kwargs(agent_key="risk_neutral"))

    @agent
    def risk_conservative(self) -> Agent:
        return Agent(config=self.agents_config["risk_conservative"], **self._agent_kwargs(agent_key="risk_conservative"))

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
            self.macro_task(),
        ]

        # ---- Bull / Bear debate ------------------------------------------
        bull_cfg = self.tasks_config["bull_round"]
        bear_cfg = self.tasks_config["bear_round"]
        debate_tasks: List[Task] = []
        debate_ctx: List[Task] = list(analyst_tasks)
        for r in range(1, self.debate_rounds + 1):
            bull_t = Task(
                description=bull_cfg["description"].format(
                    ticker=self.ticker, round_num=r, total_rounds=self.debate_rounds
                ),
                expected_output=bull_cfg["expected_output"].format(
                    ticker=self.ticker, round_num=r, total_rounds=self.debate_rounds
                ),
                agent=self.bull_researcher(),
                context=list(debate_ctx),
            )
            bear_t = Task(
                description=bear_cfg["description"].format(
                    ticker=self.ticker, round_num=r, total_rounds=self.debate_rounds
                ),
                expected_output=bear_cfg["expected_output"].format(
                    ticker=self.ticker, round_num=r, total_rounds=self.debate_rounds
                ),
                agent=self.bear_researcher(),
                context=list(debate_ctx) + [bull_t],
            )
            debate_tasks += [bull_t, bear_t]
            debate_ctx += [bull_t, bear_t]

        # ---- Research Manager synthesis ----------------------------------
        research_synthesis = Task(
            config=self.tasks_config["research_synthesis"],
            agent=self.research_manager(),
            context=list(debate_ctx),
        )

        # ---- Quality Review (optional, ASYNC advisory to the PM) ---------
        # When enabled we mark it ``async_execution=True`` and DO NOT wire
        # it into the trader's context.  It only depends on the research
        # synthesis + debate, so it fans out in parallel with the
        # trader/risk stretch and is awaited only by the PM (which lists
        # it in context).  This keeps it off the critical path.
        quality_review: Optional[Task] = None
        if self._quality_review_enabled:
            quality_review = Task(
                config=self.tasks_config["quality_review"],
                agent=self.quality_reviewer(),
                context=[research_synthesis] + list(debate_ctx),
                async_execution=True,
            )

        # ---- Trader ------------------------------------------------------
        trader_t = Task(
            config=self.tasks_config["trader_task"],
            agent=self.trader(),
            context=[research_synthesis] + list(debate_ctx),
        )

        # ---- Risk debate -------------------------------------------------
        aggr_cfg = self.tasks_config["risk_aggressive_round"]
        cons_cfg = self.tasks_config["risk_conservative_round"]
        neut_cfg = self.tasks_config["risk_neutral_round"]
        risk_tasks: List[Task] = []
        risk_ctx: List[Task] = [trader_t, research_synthesis]
        for r in range(1, self.risk_rounds + 1):
            # Phase 2E — Aggressive and Conservative argue *independently*
            # from the same prior context.  Marking both
            # ``async_execution=True`` lets CrewAI dispatch them in
            # parallel; the Neutral task lists both in its ``context`` so
            # it still waits for both to complete before synthesising.
            aggr = Task(
                description=aggr_cfg["description"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                expected_output=aggr_cfg["expected_output"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                agent=self.risk_aggressive(),
                context=list(risk_ctx),
                async_execution=True,
            )
            cons = Task(
                description=cons_cfg["description"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                expected_output=cons_cfg["expected_output"].format(
                    round_num=r, total_rounds=self.risk_rounds
                ),
                agent=self.risk_conservative(),
                # Independent from ``aggr`` — same prior context, no
                # rebuttal dependency.  Saves one round-trip-time per
                # risk round.
                context=list(risk_ctx),
                async_execution=True,
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
        # RLMF prefix — inject a compact block of past wins-to-emulate
        # and losses-to-avoid, ranked by realised market-alignment
        # reward.  Gated by ``TRADINGCREW_RLMF_ENABLED`` so users on a
        # cold cache (no resolved episodes yet) don't waste tokens on
        # an empty prefix, and so a debugging user can A/B the effect
        # by flipping it off.  Wired here (not in tasks.yaml) because
        # the exemplar pool changes every run — a YAML string can't
        # capture that.
        if _rlmf_prefix_enabled():
            try:
                from trading_crew.agentic.rlmf import build_few_shot_pool
                prefix = build_few_shot_pool(
                    symbol=self.ticker,
                    k_wins=int(os.environ.get("TRADINGCREW_RLMF_K_WINS", "3")),
                    k_losses=int(os.environ.get("TRADINGCREW_RLMF_K_LOSSES", "2")),
                )
                if prefix:
                    pm.description = f"{prefix}\n\n{pm.description}"
            except Exception:
                # RLMF is an experimental hook — never crash a workflow
                # on a bad prefix build.  The regular prompt still runs.
                import logging
                logging.getLogger(__name__).warning(
                    "RLMF prefix build failed; falling back to plain prompt",
                    exc_info=True,
                )

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
