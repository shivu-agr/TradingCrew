"""TradingCrew — the full 18-agent / 20-task workflow assembled from yaml.

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
from ._common import get_embedder_config, get_llm
from .guardrails import confidence_guardrail
from .schemas import PortfolioDecision
from .tools import ALL_TOOLS, DEFAULT_AGENT_TOOLS


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
        memory: bool = True,
        tools_enabled: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.ticker = ticker
        self.debate_rounds = debate_rounds
        self.risk_rounds = risk_rounds
        self._step_callback = step_callback
        self._task_callback = task_callback
        self._use_memory = memory
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
        return Agent(
            config=self.agents_config["market_analyst"],
            tools=self._tools_for("market_analyst"),
            **self._agent_kwargs(max_iter=8, agent_key="market_analyst"),
        )

    @agent
    def social_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["social_analyst"],
            tools=self._tools_for("social_analyst"),
            **self._agent_kwargs(agent_key="social_analyst"),
        )

    @agent
    def news_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["news_analyst"],
            tools=self._tools_for("news_analyst"),
            **self._agent_kwargs(max_iter=8, agent_key="news_analyst"),
        )

    @agent
    def fundamentals_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["fundamentals_analyst"],
            tools=self._tools_for("fundamentals_analyst"),
            **self._agent_kwargs(max_iter=10, agent_key="fundamentals_analyst"),
        )

    @agent
    def macro_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["macro_analyst"],
            tools=self._tools_for("macro_analyst"),
            **self._agent_kwargs(agent_key="macro_analyst"),
        )

    @agent
    def geopolitical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["geopolitical_analyst"],
            tools=self._tools_for("geopolitical_analyst"),
            **self._agent_kwargs(max_iter=8, agent_key="geopolitical_analyst"),
        )

    @agent
    def sector_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["sector_analyst"],
            tools=self._tools_for("sector_analyst"),
            **self._agent_kwargs(agent_key="sector_analyst"),
        )

    @agent
    def quant_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["quant_analyst"],
            tools=self._tools_for("quant_analyst"),
            **self._agent_kwargs(agent_key="quant_analyst"),
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
    def social_task(self) -> Task:
        return Task(
            config=self.tasks_config["social_task"],
            agent=self.social_analyst(),
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

    @task
    def geopolitical_task(self) -> Task:
        return Task(
            config=self.tasks_config["geopolitical_task"],
            agent=self.geopolitical_analyst(),
            async_execution=True,
        )

    @task
    def sector_task(self) -> Task:
        return Task(
            config=self.tasks_config["sector_task"],
            agent=self.sector_analyst(),
            async_execution=True,
        )

    @task
    def quant_task(self) -> Task:
        return Task(
            config=self.tasks_config["quant_task"],
            agent=self.quant_analyst(),
            async_execution=True,
        )

    # =======================================================================
    # Crew — wires everything together, including the dynamic debate rounds.
    # =======================================================================

    @crew
    def crew(self) -> Crew:
        analyst_tasks = [
            self.market_task(),
            self.social_task(),
            self.news_task(),
            self.fundamentals_task(),
            self.macro_task(),
            self.geopolitical_task(),
            self.sector_task(),
            self.quant_task(),
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

        # ---- Research Manager synthesis + Quality Review -----------------
        research_synthesis = Task(
            config=self.tasks_config["research_synthesis"],
            agent=self.research_manager(),
            context=list(debate_ctx),
        )
        quality_review = Task(
            config=self.tasks_config["quality_review"],
            agent=self.quality_reviewer(),
            context=[research_synthesis] + list(debate_ctx),
        )

        # ---- Trader ------------------------------------------------------
        trader_t = Task(
            config=self.tasks_config["trader_task"],
            agent=self.trader(),
            context=[research_synthesis, quality_review] + list(debate_ctx),
        )

        # ---- Risk debate -------------------------------------------------
        aggr_cfg = self.tasks_config["risk_aggressive_round"]
        cons_cfg = self.tasks_config["risk_conservative_round"]
        neut_cfg = self.tasks_config["risk_neutral_round"]
        risk_tasks: List[Task] = []
        risk_ctx: List[Task] = [trader_t, research_synthesis, quality_review]
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

        # ---- Compliance + PM --------------------------------------------
        compliance = Task(
            config=self.tasks_config["compliance_review"],
            agent=self.compliance_officer(),
            context=[trader_t, research_synthesis] + list(risk_tasks),
        )
        pm = Task(
            config=self.tasks_config["pm_decision"],
            agent=self.portfolio_manager(),
            context=list(risk_ctx) + [compliance],
            output_pydantic=PortfolioDecision,
            guardrail=confidence_guardrail,
            guardrail_max_retries=2,
        )

        all_tasks = (
            analyst_tasks
            + debate_tasks
            + [research_synthesis, quality_review, trader_t]
            + risk_tasks
            + [compliance, pm]
        )

        crew_kwargs: dict = dict(
            agents=[
                self.market_analyst(), self.social_analyst(),
                self.news_analyst(), self.fundamentals_analyst(),
                self.macro_analyst(), self.geopolitical_analyst(),
                self.sector_analyst(), self.quant_analyst(),
                self.bull_researcher(), self.bear_researcher(),
                self.research_manager(), self.quality_reviewer(),
                self.trader(),
                self.risk_aggressive(), self.risk_neutral(), self.risk_conservative(),
                self.compliance_officer(), self.portfolio_manager(),
            ],
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
                llm=get_llm(),
                embedder=get_embedder_config(),
                root_scope=(
                    f"/crew/{sanitize_scope_name(crew_name)}"
                    f"/emb/{sanitize_scope_name(preset_id)}"
                ),
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
