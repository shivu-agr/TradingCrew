"""CommodityCrew — 17-agent futures research workflow.

This is a sibling of ``trading_crew.TradingCrew`` and reuses the same
deterministic backbone (``trading_crew.agentic``, the M1 bridge,
``ReflectiveCritic``, run persistence). The topology mirrors the equity
crew's debate-driven pipeline but with futures-aware personas:

    7 analysts (parallel)
      -> DEBATE_ROUNDS x (Bull -> Bear)
        -> Research Manager synthesis
          -> Quality Reviewer audit
            -> Senior Futures Trader (trade plan)
              -> RISK_ROUNDS x (Aggressive -> Conservative -> Neutral)
                -> Compliance Officer
                  -> Portfolio Manager (typed FuturesDecision)

Only the PM task carries ``output_pydantic=FuturesDecision`` — the
trader produces a plan, the PM commits to the final sized action.

Public API::

    CommodityCrew(
        ticker="CL=F",
        debate_rounds=2,
        risk_rounds=1,
        step_callback=None,
        task_callback=None,
    ).crew().kickoff(inputs={"ticker": "CL=F"})
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from trading_crew._common import get_embedder_config, get_llm

from . import tools as T
from .schemas import FuturesDecision
from .tools import ALL_TOOLS, DEFAULT_AGENT_TOOLS


@CrewBase
class CommodityCrew:
    """8-agent futures research crew."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(
        self,
        ticker: str = "CL=F",
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
        self._tools_enabled: Dict[str, List[str]] = tools_enabled or {}
        self._llm = get_llm(temperature=0.3)

    # ---- tool resolution --------------------------------------------------

    def _tools_for(self, agent_key: str) -> list:
        defaults = DEFAULT_AGENT_TOOLS.get(agent_key, [])
        if agent_key in self._tools_enabled:
            allowed = [t for t in self._tools_enabled[agent_key] if t in defaults]
            names = allowed
        else:
            names = defaults
        return [ALL_TOOLS[n] for n in names if n in ALL_TOOLS]

    def _agent_kwargs(self, max_iter: int = 6) -> dict:
        return dict(
            llm=self._llm,
            allow_delegation=False,
            max_iter=max_iter,
            respect_context_window=True,
            verbose=True,
        )

    # =======================================================================
    # Agents
    # =======================================================================

    @agent
    def market_analyst(self) -> Agent:
        return Agent(config=self.agents_config["market_analyst"],
                     tools=self._tools_for("market_analyst"), **self._agent_kwargs(max_iter=8))

    @agent
    def curve_analyst(self) -> Agent:
        return Agent(config=self.agents_config["curve_analyst"],
                     tools=self._tools_for("curve_analyst"), **self._agent_kwargs())

    @agent
    def inventories_analyst(self) -> Agent:
        return Agent(config=self.agents_config["inventories_analyst"],
                     tools=self._tools_for("inventories_analyst"), **self._agent_kwargs())

    @agent
    def supply_demand_analyst(self) -> Agent:
        return Agent(config=self.agents_config["supply_demand_analyst"],
                     tools=self._tools_for("supply_demand_analyst"), **self._agent_kwargs())

    @agent
    def macro_analyst(self) -> Agent:
        return Agent(config=self.agents_config["macro_analyst"],
                     tools=self._tools_for("macro_analyst"), **self._agent_kwargs())

    @agent
    def geopolitical_analyst(self) -> Agent:
        return Agent(config=self.agents_config["geopolitical_analyst"],
                     tools=self._tools_for("geopolitical_analyst"), **self._agent_kwargs())

    @agent
    def quant_analyst(self) -> Agent:
        return Agent(config=self.agents_config["quant_analyst"],
                     tools=self._tools_for("quant_analyst"), **self._agent_kwargs())

    # ---- Researcher Team --------------------------------------------------

    @agent
    def bull_researcher(self) -> Agent:
        return Agent(config=self.agents_config["bull_researcher"], **self._agent_kwargs())

    @agent
    def bear_researcher(self) -> Agent:
        return Agent(config=self.agents_config["bear_researcher"], **self._agent_kwargs())

    @agent
    def research_manager(self) -> Agent:
        return Agent(config=self.agents_config["research_manager"], **self._agent_kwargs())

    # ---- Quality Reviewer -------------------------------------------------

    @agent
    def quality_reviewer(self) -> Agent:
        return Agent(config=self.agents_config["quality_reviewer"], **self._agent_kwargs())

    # ---- Trader -----------------------------------------------------------

    @agent
    def trader(self) -> Agent:
        return Agent(config=self.agents_config["trader"],
                     tools=self._tools_for("trader"), **self._agent_kwargs(max_iter=10))

    # ---- Risk Team --------------------------------------------------------

    @agent
    def risk_aggressive(self) -> Agent:
        return Agent(config=self.agents_config["risk_aggressive"], **self._agent_kwargs())

    @agent
    def risk_neutral(self) -> Agent:
        return Agent(config=self.agents_config["risk_neutral"], **self._agent_kwargs())

    @agent
    def risk_conservative(self) -> Agent:
        return Agent(config=self.agents_config["risk_conservative"], **self._agent_kwargs())

    # ---- Compliance + PM --------------------------------------------------

    @agent
    def compliance_officer(self) -> Agent:
        return Agent(config=self.agents_config["compliance_officer"],
                     tools=self._tools_for("compliance_officer"), **self._agent_kwargs())

    @agent
    def portfolio_manager(self) -> Agent:
        return Agent(config=self.agents_config["portfolio_manager"], **self._agent_kwargs())

    # =======================================================================
    # Tasks — 7 analyst tasks fan out in parallel, then the @crew method
    # builds the debate / risk / compliance / PM chain programmatically.
    # =======================================================================

    @task
    def market_task(self) -> Task:
        return Task(
            config=self.tasks_config["market_task"],
            agent=self.market_analyst(),
            async_execution=True,
        )

    @task
    def curve_task(self) -> Task:
        return Task(
            config=self.tasks_config["curve_task"],
            agent=self.curve_analyst(),
            async_execution=True,
        )

    @task
    def inventories_task(self) -> Task:
        return Task(
            config=self.tasks_config["inventories_task"],
            agent=self.inventories_analyst(),
            async_execution=True,
        )

    @task
    def supply_demand_task(self) -> Task:
        return Task(
            config=self.tasks_config["supply_demand_task"],
            agent=self.supply_demand_analyst(),
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
            self.curve_task(),
            self.inventories_task(),
            self.supply_demand_task(),
            self.macro_task(),
            self.geopolitical_task(),
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

        # ---- Trader plan (NOT final decision) ----------------------------
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
            aggr = Task(
                description=aggr_cfg["description"].format(
                    round_num=r, total_rounds=self.risk_rounds, ticker=self.ticker
                ),
                expected_output=aggr_cfg["expected_output"].format(
                    round_num=r, total_rounds=self.risk_rounds, ticker=self.ticker
                ),
                agent=self.risk_aggressive(),
                context=list(risk_ctx),
            )
            cons = Task(
                description=cons_cfg["description"].format(
                    round_num=r, total_rounds=self.risk_rounds, ticker=self.ticker
                ),
                expected_output=cons_cfg["expected_output"].format(
                    round_num=r, total_rounds=self.risk_rounds, ticker=self.ticker
                ),
                agent=self.risk_conservative(),
                context=list(risk_ctx) + [aggr],
            )
            neut = Task(
                description=neut_cfg["description"].format(
                    round_num=r, total_rounds=self.risk_rounds, ticker=self.ticker
                ),
                expected_output=neut_cfg["expected_output"].format(
                    round_num=r, total_rounds=self.risk_rounds, ticker=self.ticker
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
            output_pydantic=FuturesDecision,
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
                self.market_analyst(), self.curve_analyst(),
                self.inventories_analyst(), self.supply_demand_analyst(),
                self.macro_analyst(), self.geopolitical_analyst(),
                self.quant_analyst(),
                self.bull_researcher(), self.bear_researcher(),
                self.research_manager(), self.quality_reviewer(),
                self.trader(),
                self.risk_aggressive(), self.risk_neutral(), self.risk_conservative(),
                self.compliance_officer(), self.portfolio_manager(),
            ],
            tasks=all_tasks,
            process=Process.sequential,
            verbose=True,
            step_callback=self._step_callback,
            task_callback=self._task_callback,
        )
        if self._use_memory:
            # Native CrewAI memory: memory=True + embedder spec routed at our
            # vLLM-hosted embedding endpoint (see get_embedder_config). CrewAI
            # builds a unified short-term / long-term / entity memory whose
            # root_scope is derived from ``name``.
            crew_kwargs["memory"] = True
            crew_kwargs["embedder"] = get_embedder_config()
            crew_kwargs["name"] = f"commodity-crew-{self.ticker}"
        return Crew(**crew_kwargs)


# ---------------------------------------------------------------------------
# Catalog used by the UI bootstrap (mirrors trading_crew runner's helper)
# ---------------------------------------------------------------------------


_AGENT_LAYOUT = [
    # analysts
    ("market_analyst",         "Market Analyst",          "analyst"),
    ("curve_analyst",          "Curve Analyst",           "analyst"),
    ("inventories_analyst",    "Inventories Analyst",     "analyst"),
    ("supply_demand_analyst",  "Supply/Demand Analyst",   "analyst"),
    ("macro_analyst",          "Macro Analyst",           "analyst"),
    ("geopolitical_analyst",   "Geopolitical Analyst",    "analyst"),
    ("quant_analyst",          "Positioning Quant",       "analyst"),
    # researchers
    ("bull_researcher",        "Bullish Researcher",      "bull"),
    ("bear_researcher",        "Bearish Researcher",      "bear"),
    ("research_manager",       "Research Manager",        "manager"),
    # quality review
    ("quality_reviewer",       "Quality Reviewer",        "reviewer"),
    # trader
    ("trader",                 "Futures Trader",          "trader"),
    # risk debate
    ("risk_aggressive",        "Aggressive Risk Analyst", "risk_a"),
    ("risk_neutral",           "Neutral Risk Analyst",    "risk_n"),
    ("risk_conservative",      "Conservative Risk Analyst","risk_c"),
    # compliance + PM
    ("compliance_officer",     "Compliance Officer",      "reviewer"),
    ("portfolio_manager",      "Portfolio Manager",       "manager"),
]


def get_agent_catalog() -> List[Dict[str, Any]]:
    """Agent layout for the workflow diagram + tool checkboxes."""
    out: List[Dict[str, Any]] = []
    for key, role, kind in _AGENT_LAYOUT:
        out.append({
            "key": key,
            "role": role,
            "kind": kind,
            "tools": list(DEFAULT_AGENT_TOOLS.get(key, [])),
        })
    return out
