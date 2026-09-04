"""Gate every conductor dispatch."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

QUERY_FAMILIES: Dict[str, Set[str]] = {
    "price": {
        "get_stock_data",
        "get_indicators",
        "get_options_summary",
        "get_analyst_recommendations",
        "get_crowd_positioning",
        "get_event_proximity",
    },
    "fundamental": {
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_shareholding_pattern",
        "get_long_term_thesis",
        "get_forward_estimates",
        "get_sector_peers",
        "get_report_insights",
    },
    "policy": {
        "get_news",
        "get_global_news",
        "get_corporate_actions",
        "get_insider_transactions",
        "research_web",
        "get_macro_data",
        "get_geopolitical_news",
        "get_supply_chain_risk",
        "query_knowledge",
        "get_market_context",
    },
}

SKEPTICAL_TOOLS = {
    "get_geopolitical_news",
    "get_supply_chain_risk",
    "get_insider_transactions",
    "get_shareholding_pattern",
    "get_crowd_positioning",
}

DEBATE_DESKS = {"bull_researcher", "bear_researcher"}
BUNDLED_DEBATE = "debate"
RISK_DESKS = {"risk_aggressive", "risk_conservative", "risk_neutral"}
BUNDLED_RISK = "risk"
BLOCKED_SPAWN = {"portfolio_manager", "quality_reviewer", "compliance_officer"}


class HarnessGateError(ValueError):
    """A dispatch was refused by the safety layer."""


def tool_family(name: str) -> Optional[str]:
    for family, names in QUERY_FAMILIES.items():
        if name in names:
            return family
    return None


def check_tool(
    name: str,
    args: Dict[str, Any],
    cfg: Dict[str, Any],
) -> None:
    if name in cfg["deny_tools"]:
        raise HarnessGateError(f"tool denied: {name}")
    if name not in cfg["tool_allowlist"] and name != "query_knowledge":
        raise HarnessGateError(f"tool not on allowlist: {name}")
    if name in cfg["query_required_tools"]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise HarnessGateError(
                f"{name} requires an agent-written query; canned seeds are not used"
            )


def check_spawn(
    desk: str,
    *,
    depth: int,
    cfg: Dict[str, Any],
    spawn_used: int,
) -> None:
    if depth >= int(cfg["max_spawn_depth"]):
        raise HarnessGateError("spawn depth exhausted; children cannot spawn")
    if spawn_used >= int(cfg["max_spawn"]):
        raise HarnessGateError("max_spawn exhausted")
    key = (desk or "").strip()
    if key in BLOCKED_SPAWN:
        raise HarnessGateError(f"desk cannot be spawned: {key}")
    if key != "generic_investigator" and key not in cfg["catalog_desks"]:
        raise HarnessGateError(f"unknown catalog desk: {key}")


def is_skeptical_spawn(kind: str, desk: str, brief: str) -> bool:
    if desk in ("bear_researcher", "debate"):
        return True
    if desk in RISK_DESKS or desk == BUNDLED_RISK:
        return True
    text = f"{kind} {desk} {brief}".lower()
    needles = (
        "skeptic",
        "counter",
        "bear",
        "risk",
        "falsif",
        "ownership",
        "circular",
        "related-party",
        "related party",
        "policy shock",
        "what would make this wrong",
    )
    return any(n in text for n in needles)


def diversity_gaps(
    *,
    families: Iterable[str],
    skeptical: bool,
    cfg: Dict[str, Any],
) -> List[str]:
    used = {f for f in families if f}
    need = int(cfg["min_tool_families"])
    gaps: List[str] = []
    if len(used) < need:
        gaps.append(
            f"need {need} tool families, have {sorted(used) or 'none'} "
            f"(price, fundamental, policy)"
        )
    if cfg["require_skeptical_lens"] and not skeptical:
        gaps.append(
            "need a skeptical lens: bear desk, risk desk, a counter-brief "
            "investigator, or a risk/ownership/policy tool"
        )
    return gaps


def coverage_gaps(
    *,
    families: Iterable[str],
    skeptical: bool,
    spawned: Iterable[str],
    reports: Dict[str, str],
    tool_names: Iterable[str],
    steps: int,
    cfg: Dict[str, Any],
) -> List[str]:
    """Finish gates: tool diversity plus the required long-horizon lenses."""
    gaps = diversity_gaps(families=families, skeptical=skeptical, cfg=cfg)
    desks = {str(k).strip() for k in spawned if k}
    tools = {str(n).strip() for n in tool_names if n}
    filled = {
        key for key, text in (reports or {}).items() if str(text or "").strip()
    }
    if cfg.get("require_market_lens") and "market_analyst" not in desks and "market_analyst" not in filled:
        gaps.append("need market_analyst (price tape)")
    if cfg.get("require_news_lens") and "news_analyst" not in desks and "news_analyst" not in filled:
        gaps.append("need news_analyst")
    if (
        cfg.get("require_fundamentals_lens")
        and "fundamentals_analyst" not in desks
        and "fundamentals_analyst" not in filled
    ):
        gaps.append("need fundamentals_analyst")
    if cfg.get("require_macro_lens") and "macro_analyst" not in desks and "macro_analyst" not in filled:
        gaps.append("need macro_analyst")
    if cfg.get("require_reports_lens"):
        filings = (
            "reports_analyst" in desks
            or "reports_analyst" in filled
            or "get_report_insights" in tools
        )
        if not filings:
            gaps.append("need reports_analyst or get_report_insights")
    if cfg.get("require_debate_pair"):
        if "bull_researcher" not in filled:
            gaps.append("need bull_researcher so a debate can happen")
        if "bear_researcher" not in filled:
            gaps.append("need bear_researcher so a debate can happen")
    if cfg.get("require_bundled_debate"):
        bundled = BUNDLED_DEBATE in desks or BUNDLED_DEBATE in filled
        if not bundled:
            gaps.append(
                "need a bundled debate (desk=debate) so both sides argue "
                "in one crew and can fetch evidence mid-exchange"
            )
    if cfg.get("require_bundled_risk"):
        bundled_risk = BUNDLED_RISK in desks or BUNDLED_RISK in filled
        if not bundled_risk:
            gaps.append(
                "need a bundled risk debate (desk=risk) so aggressive, "
                "conservative, and neutral each write an opinion then "
                "debate each other"
            )
    if cfg.get("require_risk_lens"):
        if not any(k in filled or k in desks for k in RISK_DESKS):
            gaps.append(
                "need a risk desk (risk_aggressive, risk_conservative, or risk_neutral)"
            )
    floor = int(cfg.get("min_steps_before_finish") or 0)
    if floor and steps < floor:
        gaps.append(
            f"long-horizon budget: {floor} steps before finish, now {steps}"
        )
    return gaps
