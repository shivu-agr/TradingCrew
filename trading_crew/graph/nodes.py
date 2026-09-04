"""CrewAI Agent nodes scheduled by the v3 graph."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from crewai import Crew, Process, Task
from crewai.utilities.string_utils import interpolate_only

from trading_crew.llm_presets import submit_in_active_context
from trading_crew.schemas import PortfolioDecision

logger = logging.getLogger(__name__)

ANALYST_KEYS = (
    "market_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "reports_analyst",
    "macro_analyst",
)

_TASK_FOR_ANALYST = {
    "market_analyst": "market_task",
    "news_analyst": "news_task",
    "fundamentals_analyst": "fundamentals_task",
    "reports_analyst": "reports_task",
    "macro_analyst": "macro_task",
}

_ANALYST_TITLES = {
    "market_analyst": "Market Analyst report",
    "news_analyst": "News Analyst report",
    "fundamentals_analyst": "Fundamentals Analyst report",
    "reports_analyst": "Reports Analyst report",
    "macro_analyst": "Macro / Geopolitical Analyst report",
}

# CrewAI kickoff interpolates ``{identifier}`` in the task description.
# Prior desk output can contain those shapes (JSON keys, ``{ticker}``
# copied from a prompt). Neutralize them so interpolation cannot
# KeyError on a report body.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_\-]*)\}")
_SOURCE_TAG_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.I)


def neutralize_placeholders(text: str) -> str:
    return _PLACEHOLDER_RE.sub(r"{ \1}", text or "")


def merge_spawn_instruction(standing: str, instruction: str) -> str:
    """Lead with the planner contract; keep the YAML remit as standing text."""
    packed = neutralize_placeholders(instruction or "").strip()
    remit = (standing or "").strip()
    if not packed:
        return remit
    if not remit:
        return packed
    return f"{packed}\n\n## Standing desk remit\n\n{remit}"


def extract_source_tags(*texts: str) -> List[str]:
    """Unique ``[source: …]`` identifiers, first-seen order."""
    seen: List[str] = []
    keys: set = set()
    for text in texts:
        for match in _SOURCE_TAG_RE.finditer(text or ""):
            ident = " ".join(match.group(1).split())
            key = ident.lower()
            if key and key not in keys:
                keys.add(key)
                seen.append(ident)
    return seen


def format_citation_index(tags: Sequence[str]) -> str:
    """Make tagged sources visible so Quality Review cannot miss them."""
    if not tags:
        return ""
    lines = "\n".join(f"- {tag}" for tag in tags)
    return (
        "## Citation index\n"
        "These [source: …] identifiers already appear on claims in this "
        "run. A claim that carries one of them is SUPPORTED. Do not "
        "require the same URL to also appear as prose in another desk "
        "report, and do not tell the PM to drop that claim.\n"
        f"{lines}"
    )


def pack_prior_outputs(sections: Sequence[Tuple[str, str]]) -> str:
    """Stuff prior desk text the way v2 CrewAI ``Task.context`` did.

    Isolated one-task Crews have no implicit memory. Without this the
    PM only sees runner-injected ledgers (price-structure + guidance)
    and the decision collapses onto charts / filings.
    """
    blocks: List[str] = []
    for title, body in sections:
        text = (body or "").strip()
        if not text:
            continue
        blocks.append(f"## {title}\n{text}")
    if not blocks:
        return ""
    return (
        "# PRIOR OUTPUTS FROM THIS RUN\n"
        "These are the actual desk outputs for this ticker. Cite them "
        "with their inline [source: …] tags. Do not invent figures that "
        "are not here.\n\n"
        + "\n\n".join(blocks)
    )


class GraphRuntime:
    """Holds materialized agents and runs one-task Crews so existing callbacks fire."""

    def __init__(
        self,
        trading_crew: Any,
        agents: Dict[str, Any],
        *,
        step_callback: Optional[Callable] = None,
        task_callback: Optional[Callable] = None,
        on_edge: Optional[Callable[[str, str, str], None]] = None,
    ) -> None:
        self.tc = trading_crew
        self.agents = agents
        self.step_callback = step_callback
        self.task_callback = task_callback
        self.on_edge = on_edge
        self.last_pydantic: Any = None
        self.last_raw: str = ""

    def emit_edge(self, src: str, dst: str, kind: str = "activated") -> None:
        if self.on_edge:
            self.on_edge(src, dst, kind)

    def _task_cfg(self, name: str) -> Dict[str, str]:
        return self.tc.tasks_config[name]

    def _run_agent(
        self,
        agent_key: str,
        description: str,
        expected_output: str,
        *,
        output_pydantic: Any = None,
        prior: str = "",
    ) -> str:
        if prior:
            description = f"{description}\n\n{neutralize_placeholders(prior)}"
        agent = self.agents[agent_key]
        kwargs: Dict[str, Any] = dict(
            description=description,
            expected_output=expected_output,
            agent=agent,
        )
        if output_pydantic is not None:
            kwargs["output_pydantic"] = output_pydantic
        task = Task(**kwargs)
        crew_kwargs: Dict[str, Any] = dict(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        if self.step_callback is not None:
            crew_kwargs["step_callback"] = self.step_callback
        if self.task_callback is not None:
            crew_kwargs["task_callback"] = self.task_callback
        mini = Crew(**crew_kwargs)
        result = mini.kickoff(inputs={"ticker": self.tc.ticker})
        raw = ""
        if result is not None:
            raw = getattr(result, "raw", None) or str(result)
            if getattr(result, "pydantic", None) is not None:
                self.last_pydantic = result.pydantic
        self.last_raw = raw or ""
        return self.last_raw

    def _format(self, name: str, **fmt: Any) -> tuple[str, str]:
        # Same interpolator CrewAI uses on kickoff: only ``{identifier}``
        # placeholders are replaced. ``str.format`` would treat set
        # notation in the PM prompt (``{SHORT, MEDIUM}``) as a field.
        cfg = self._task_cfg(name)
        from trading_crew.agentic.run_mode_prompts import task_addendum, task_override

        mode = getattr(self.tc, "run_mode", "autonomous")
        override = task_override(name, book="equity", run_mode=mode)
        if str(mode).strip().lower() == "harness":
            if not override:
                raise ValueError(f"harness run is missing a task replacement for {name}")
            desc = interpolate_only(override["description"], fmt)
            src_out = override.get("expected_output") or cfg["expected_output"]
            return desc, interpolate_only(src_out, fmt)
        desc = interpolate_only(cfg["description"], fmt)
        out = interpolate_only(cfg["expected_output"], fmt)
        extra = task_addendum(name, book="equity", run_mode=mode)
        if extra:
            desc = f"{desc}\n\n{extra}"
        return desc, out

    def run_spawned(self, agent_key: str, task_name: str, prior: str = "") -> str:
        """One-shot desk for the harness conductor — no round_num / total_rounds."""
        desc, out = self._format(task_name, ticker=self.tc.ticker)
        desc = merge_spawn_instruction(desc, prior or "")
        return self._run_agent(agent_key, desc, out)

    def run_analyst(self, agent_key: str) -> str:
        task_name = _TASK_FOR_ANALYST[agent_key]
        desc, out = self._format(task_name, ticker=self.tc.ticker)
        return self._run_agent(agent_key, desc, out)

    def run_analysts(self, skip_reports: bool) -> Dict[str, str]:
        keys = [k for k in ANALYST_KEYS if not (skip_reports and k == "reports_analyst")]
        reports: Dict[str, str] = {}
        if not keys:
            return reports
        with ThreadPoolExecutor(max_workers=len(keys)) as pool:
            futs = {submit_in_active_context(pool, self.run_analyst, k): k for k in keys}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    reports[key] = fut.result() or ""
                except Exception:
                    logger.exception("analyst %s failed", key)
                    raise
        for key in keys:
            role = self.agents[key].role
            self.emit_edge("Playbook Router", role, "activated")
            self.emit_edge(role, "Shared State", "activated")
        self.emit_edge("Shared State", "Bullish Researcher", "activated")
        self.emit_edge("Shared State", "Bearish Researcher", "activated")
        return reports

    def _debate_side(self, side: str, round_num: int, total_rounds: int) -> str:
        name = "bull_round" if side == "bull" else "bear_round"
        key = "bull_researcher" if side == "bull" else "bear_researcher"
        desc, out = self._format(
            name,
            ticker=self.tc.ticker,
            round_num=round_num,
            total_rounds=total_rounds,
        )
        return self._run_agent(key, desc, out)

    def _task_text(self, task_out: Any) -> str:
        if task_out is None:
            return ""
        return (
            getattr(task_out, "raw", None)
            or getattr(task_out, "exported_output", None)
            or str(task_out)
            or ""
        )

    def run_bundled_debate(self, prior: str = "") -> Tuple[str, str]:
        """One crew: bull opens (tools), bear rebuts (tools), bull closes (tools).

        Separate ``bull_researcher`` / ``bear_researcher`` one-shots remain
        available for extra turns. This bundle is the live exchange.
        """
        bull_agent = self.agents["bull_researcher"]
        bear_agent = self.agents["bear_researcher"]
        ticker = self.tc.ticker
        bull_open_desc, bull_open_out = self._format(
            "debate_bull_open", ticker=ticker
        )
        bear_desc, bear_out = self._format("debate_bear_rebut", ticker=ticker)
        bull_close_desc, bull_close_out = self._format(
            "debate_bull_close", ticker=ticker
        )
        if prior:
            bull_open_desc = merge_spawn_instruction(bull_open_desc, prior)
        open_task = Task(
            description=bull_open_desc,
            expected_output=bull_open_out,
            agent=bull_agent,
        )
        rebut_task = Task(
            description=bear_desc,
            expected_output=bear_out,
            agent=bear_agent,
            context=[open_task],
        )
        close_task = Task(
            description=bull_close_desc,
            expected_output=bull_close_out,
            agent=bull_agent,
            context=[open_task, rebut_task],
        )
        crew_kwargs: Dict[str, Any] = dict(
            agents=[bull_agent, bear_agent],
            tasks=[open_task, rebut_task, close_task],
            process=Process.sequential,
            verbose=True,
        )
        if self.step_callback is not None:
            crew_kwargs["step_callback"] = self.step_callback
        if self.task_callback is not None:
            crew_kwargs["task_callback"] = self.task_callback
        crew = Crew(**crew_kwargs)
        result = crew.kickoff(inputs={"ticker": ticker})
        outputs = list(getattr(result, "tasks_output", None) or [])
        bull_open = self._task_text(outputs[0]) if len(outputs) > 0 else ""
        bear_text = self._task_text(outputs[1]) if len(outputs) > 1 else ""
        bull_close = self._task_text(outputs[2]) if len(outputs) > 2 else ""
        if not bear_text and result is not None:
            bear_text = getattr(result, "raw", None) or ""
        bull_text = "\n\n".join(
            part for part in (bull_open, bull_close) if (part or "").strip()
        )
        self.last_raw = (
            f"## Bundled debate — Bull open\n{bull_open}\n\n"
            f"## Bundled debate — Bear rebuttal\n{bear_text}\n\n"
            f"## Bundled debate — Bull close\n{bull_close}"
        )
        self.emit_edge("Bullish Researcher", "Bearish Researcher", "activated")
        self.emit_edge("Bearish Researcher", "Bullish Researcher", "cycled")
        self.emit_edge("Bullish Researcher", "Research Manager", "activated")
        self.emit_edge("Bearish Researcher", "Research Manager", "activated")
        return bull_text or bull_open, bear_text

    def run_opening_debate(self, round_num: int, total_rounds: int) -> Tuple[str, str]:
        """Opening exchange: one bundled crew (bull → bear → bull close)."""
        return self.run_bundled_debate()

    def run_bundled_risk(self, prior: str = "") -> Tuple[str, str, str]:
        """One crew, two rounds: each wing writes an opinion, then they debate.

        Separate ``risk_aggressive`` / ``risk_conservative`` / ``risk_neutral``
        one-shots remain available after this bundle.
        """
        aggr_agent = self.agents["risk_aggressive"]
        cons_agent = self.agents["risk_conservative"]
        neut_agent = self.agents["risk_neutral"]
        ticker = self.tc.ticker

        def _task(name: str, agent: Any, context: Optional[List[Any]] = None) -> Task:
            desc, out = self._format(name, ticker=ticker)
            if prior and name.endswith("_open"):
                desc = merge_spawn_instruction(desc, prior)
            kwargs: Dict[str, Any] = dict(
                description=desc,
                expected_output=out,
                agent=agent,
            )
            if context:
                kwargs["context"] = context
            return Task(**kwargs)

        aggr_open = _task("risk_aggressive_open", aggr_agent)
        cons_open = _task("risk_conservative_open", cons_agent)
        neut_open = _task("risk_neutral_open", neut_agent)
        opens = [aggr_open, cons_open, neut_open]
        aggr_rebut = _task("risk_aggressive_rebut", aggr_agent, opens)
        cons_rebut = _task("risk_conservative_rebut", cons_agent, opens)
        neut_rebut = _task(
            "risk_neutral_rebut",
            neut_agent,
            opens + [aggr_rebut, cons_rebut],
        )
        crew_kwargs: Dict[str, Any] = dict(
            agents=[aggr_agent, cons_agent, neut_agent],
            tasks=[
                aggr_open,
                cons_open,
                neut_open,
                aggr_rebut,
                cons_rebut,
                neut_rebut,
            ],
            process=Process.sequential,
            verbose=True,
        )
        if self.step_callback is not None:
            crew_kwargs["step_callback"] = self.step_callback
        if self.task_callback is not None:
            crew_kwargs["task_callback"] = self.task_callback
        crew = Crew(**crew_kwargs)
        result = crew.kickoff(inputs={"ticker": ticker})
        outputs = list(getattr(result, "tasks_output", None) or [])
        texts = [self._task_text(outputs[i]) if len(outputs) > i else "" for i in range(6)]
        aggr_text = "\n\n".join(p for p in (texts[0], texts[3]) if (p or "").strip())
        cons_text = "\n\n".join(p for p in (texts[1], texts[4]) if (p or "").strip())
        neut_text = "\n\n".join(p for p in (texts[2], texts[5]) if (p or "").strip())
        self.last_raw = (
            f"## Bundled risk — Aggressive open\n{texts[0]}\n\n"
            f"## Bundled risk — Conservative open\n{texts[1]}\n\n"
            f"## Bundled risk — Neutral open\n{texts[2]}\n\n"
            f"## Bundled risk — Aggressive rebuttal\n{texts[3]}\n\n"
            f"## Bundled risk — Conservative rebuttal\n{texts[4]}\n\n"
            f"## Bundled risk — Neutral rebuttal\n{texts[5]}"
        )
        self.emit_edge("Aggressive Risk Analyst", "Conservative Risk Analyst", "activated")
        self.emit_edge("Conservative Risk Analyst", "Neutral Risk Analyst", "activated")
        self.emit_edge("Neutral Risk Analyst", "Aggressive Risk Analyst", "cycled")
        self.emit_edge("Aggressive Risk Analyst", "Portfolio Manager", "activated")
        self.emit_edge("Conservative Risk Analyst", "Portfolio Manager", "activated")
        self.emit_edge("Neutral Risk Analyst", "Portfolio Manager", "activated")
        return aggr_text or texts[0], cons_text or texts[1], neut_text or texts[2]

    def run_debate_turn(self, side: str, round_num: int, total_rounds: int) -> str:
        text = self._debate_side(side, round_num, total_rounds)
        if side == "bull":
            self.emit_edge("Bullish Researcher", "Bearish Researcher", "activated")
        else:
            self.emit_edge("Bearish Researcher", "Research Manager", "activated")
        return text

    def drain_evidence(self) -> int:
        from trading_crew.agentic.graph.blackboard import current, drain_queued_requests

        bb = current()
        if bb is None:
            return 0
        n = drain_queued_requests(bb.ticker)
        self.emit_edge("Bearish Researcher", "Shared State", "cycled")
        self.emit_edge("Shared State", "Market Analyst", "cycled")
        self.emit_edge("Shared State", "News Analyst", "cycled")
        self.emit_edge("Shared State", "Bullish Researcher", "cycled")
        return n

    def evidence_is_queued(self) -> bool:
        from trading_crew.agentic.graph.blackboard import current

        bb = current()
        if bb is None:
            return False
        return any(r.status == "queued" for r in bb.requests)

    def run_research_manager(self, last_bull: str = "", last_bear: str = "") -> str:
        desc, out = self._format("research_synthesis", ticker=self.tc.ticker)
        prior = pack_prior_outputs([
            ("Last Bullish Researcher turn", last_bull),
            ("Last Bearish Researcher turn", last_bear),
        ])
        text = self._run_agent("research_manager", desc, out, prior=prior)
        self.emit_edge("Research Manager", "Trader", "activated")
        return text

    def run_trader(self, thesis: str = "") -> str:
        desc, out = self._format("trader_task", ticker=self.tc.ticker)
        prior = pack_prior_outputs([("Research Manager thesis", thesis)])
        text = self._run_agent("trader", desc, out, prior=prior)
        self.emit_edge("Trader", "Aggressive Risk Analyst", "activated")
        self.emit_edge("Trader", "Conservative Risk Analyst", "activated")
        self.emit_edge("Trader", "Neutral Risk Analyst", "activated")
        return text

    def run_risk_wings(
        self,
        round_num: int = 1,
        total_rounds: int = 1,
        *,
        thesis: str = "",
        trade_plan: str = "",
        prior_wings: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        notes: Dict[str, str] = {}
        extras = [
            ("Research Manager thesis", thesis),
            ("Trader plan", trade_plan),
        ]
        if round_num > 1:
            extras.extend([
                (
                    "Prior Aggressive Risk turn",
                    (prior_wings or {}).get("Aggressive Risk Analyst", ""),
                ),
                (
                    "Prior Conservative Risk turn",
                    (prior_wings or {}).get("Conservative Risk Analyst", ""),
                ),
            ])

        def _one(key: str, task_name: str, extra_prior: str = "") -> str:
            desc, out = self._format(
                task_name,
                ticker=self.tc.ticker,
                round_num=round_num,
                total_rounds=total_rounds,
            )
            packed = list(extras)
            if extra_prior:
                packed.append(("This-round Aggressive Risk turn", extra_prior))
            return self._run_agent(
                key, desc, out, prior=pack_prior_outputs(packed)
            )

        if round_num == 1:
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_a = submit_in_active_context(
                    pool, _one, "risk_aggressive", "risk_aggressive_round"
                )
                f_c = submit_in_active_context(
                    pool, _one, "risk_conservative", "risk_conservative_round"
                )
                notes["Aggressive Risk Analyst"] = f_a.result() or ""
                notes["Conservative Risk Analyst"] = f_c.result() or ""
        else:
            notes["Aggressive Risk Analyst"] = _one(
                "risk_aggressive", "risk_aggressive_round"
            )
            self.emit_edge(
                "Aggressive Risk Analyst", "Conservative Risk Analyst", "activated"
            )
            notes["Conservative Risk Analyst"] = _one(
                "risk_conservative",
                "risk_conservative_round",
                extra_prior=notes["Aggressive Risk Analyst"],
            )
        self.emit_edge("Aggressive Risk Analyst", "Neutral Risk Analyst", "activated")
        self.emit_edge("Conservative Risk Analyst", "Neutral Risk Analyst", "activated")
        return notes

    def run_risk_neutral(
        self,
        round_num: int = 1,
        total_rounds: int = 1,
        *,
        thesis: str = "",
        trade_plan: str = "",
        wing_notes: Optional[Dict[str, str]] = None,
    ) -> str:
        desc, out = self._format(
            "risk_neutral_round",
            ticker=self.tc.ticker,
            round_num=round_num,
            total_rounds=total_rounds,
        )
        extra = (
            "\n\nIf the trader's size is wrong given the risk debate, emit a "
            "line containing exactly ``REOPEN_TRADER`` (once) so the graph "
            "can send the plan back for a re-size. Do not emit it otherwise."
        )
        desc = desc + extra
        wings = wing_notes or {}
        prior = pack_prior_outputs([
            ("Research Manager thesis", thesis),
            ("Trader plan", trade_plan),
            ("Aggressive Risk Analyst", wings.get("Aggressive Risk Analyst", "")),
            ("Conservative Risk Analyst", wings.get("Conservative Risk Analyst", "")),
        ])
        text = self._run_agent("risk_neutral", desc, out, prior=prior)
        return text

    def run_reviewers(
        self,
        *,
        quality_enabled: bool,
        compliance_enabled: bool,
        reports: Optional[Dict[str, str]] = None,
        thesis: str = "",
        trade_plan: str = "",
        risk_notes: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Advisory Quality Review / Compliance after risk, before PM."""
        notes: Dict[str, str] = {}
        analyst_sections = [
            (_ANALYST_TITLES[key], (reports or {}).get(key, ""))
            for key in ANALYST_KEYS
        ]
        qr_sections = analyst_sections + [("Research Manager thesis", thesis)]
        qr_prior = pack_prior_outputs(qr_sections)
        qr_index = format_citation_index(
            extract_source_tags(*[body for _, body in qr_sections])
        )
        if qr_index:
            qr_prior = f"{qr_prior}\n\n{qr_index}" if qr_prior else qr_index
        co_prior = pack_prior_outputs(
            analyst_sections
            + [
                ("Research Manager thesis", thesis),
                ("Trader plan", trade_plan),
            ]
            + [(str(role), text) for role, text in (risk_notes or {}).items()]
        )

        def _qr() -> str:
            desc, out = self._format("quality_review", ticker=self.tc.ticker)
            return self._run_agent("quality_reviewer", desc, out, prior=qr_prior)

        def _co() -> str:
            desc, out = self._format("compliance_review", ticker=self.tc.ticker)
            return self._run_agent("compliance_officer", desc, out, prior=co_prior)

        jobs: List[Tuple[str, Callable[[], str]]] = []
        if quality_enabled:
            jobs.append(("quality_notes", _qr))
        if compliance_enabled:
            jobs.append(("compliance_notes", _co))
        if not jobs:
            return notes
        if len(jobs) == 1:
            key, fn = jobs[0]
            notes[key] = fn() or ""
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futs = {submit_in_active_context(pool, fn): key for key, fn in jobs}
                for fut in as_completed(futs):
                    key = futs[fut]
                    notes[key] = fut.result() or ""
        if quality_enabled:
            self.emit_edge("Neutral Risk Analyst", "Quality Reviewer", "activated")
            self.emit_edge("Quality Reviewer", "Portfolio Manager", "activated")
        if compliance_enabled:
            self.emit_edge("Neutral Risk Analyst", "Compliance Officer", "activated")
            self.emit_edge("Compliance Officer", "Portfolio Manager", "activated")
        return notes

    def run_pm(
        self,
        *,
        reports: Optional[Dict[str, str]] = None,
        thesis: str = "",
        trade_plan: str = "",
        risk_notes: Optional[Dict[str, str]] = None,
        quality_notes: str = "",
        compliance_notes: str = "",
    ) -> str:
        desc, out = self._format("pm_decision", ticker=self.tc.ticker)
        pm_task = getattr(self.tc, "_pm_task", None)
        if (
            str(getattr(self.tc, "run_mode", "")).strip().lower() != "harness"
            and pm_task is not None
            and getattr(pm_task, "description", None)
        ):
            # Bind {ticker} only. Neutralize any other {ident} the
            # runner/RLMF prefix may have left so interpolate_only
            # cannot KeyError before we append prior reports.
            raw = str(pm_task.description)
            raw = _PLACEHOLDER_RE.sub(
                lambda m: m.group(0) if m.group(1) == "ticker" else "{ " + m.group(1) + "}",
                raw,
            )
            desc = interpolate_only(raw, {"ticker": self.tc.ticker})
        sections: List[Tuple[str, str]] = [
            (_ANALYST_TITLES[key], (reports or {}).get(key, ""))
            for key in ANALYST_KEYS
        ]
        sections.append(("Research Manager thesis", thesis))
        sections.append(("Trader plan", trade_plan))
        for role, text in (risk_notes or {}).items():
            sections.append((str(role), text))
        if quality_notes:
            sections.append(("Quality Review", quality_notes))
        if compliance_notes:
            sections.append(("Compliance review", compliance_notes))
        prior = pack_prior_outputs(sections)
        text = self._run_agent(
            "portfolio_manager",
            desc,
            out,
            output_pydantic=PortfolioDecision,
            prior=prior,
        )
        self.emit_edge("Portfolio Manager", "decision", "activated")
        return text
