"""Stock workflow graph spec — single source of truth for nodes, edges, tools.

The UI diagram currently hardcodes the same layout in
``web/frontend/diagram.js``.  Node ``id`` values MUST stay equal to the
CrewAI agent ``role`` strings the runner emits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Display role (runner / diagram node id) → crew agent_key.
ROLE_TO_AGENT_KEY: Dict[str, str] = {
    "Market Analyst": "market_analyst",
    "News Analyst": "news_analyst",
    "Fundamentals Analyst": "fundamentals_analyst",
    "Reports Analyst": "reports_analyst",
    "Macro Analyst": "macro_analyst",
    "Bullish Researcher": "bull_researcher",
    "Bearish Researcher": "bear_researcher",
    "Research Manager": "research_manager",
    "Quality Reviewer": "quality_reviewer",
    "Trader": "trader",
    "Aggressive Risk Analyst": "risk_aggressive",
    "Neutral Risk Analyst": "risk_neutral",
    "Conservative Risk Analyst": "risk_conservative",
    "Compliance Officer": "compliance_officer",
    "Portfolio Manager": "portfolio_manager",
}

AGENT_KEY_TO_ROLE: Dict[str, str] = {v: k for k, v in ROLE_TO_AGENT_KEY.items()}

ANALYST_AGENT_KEYS: Tuple[str, ...] = (
    "market_analyst",
    "news_analyst",
    "fundamentals_analyst",
    "reports_analyst",
    "macro_analyst",
)

# Researcher desks may request a re-query of these tools (owning desk).
RESEARCHER_EVIDENCE_TOOLS: Dict[str, List[str]] = {
    "market_analyst": [
        "get_stock_data", "get_indicators",
        "get_options_summary", "get_analyst_recommendations",
    ],
    "news_analyst": [
        "get_news", "get_global_news",
        "get_corporate_actions", "get_insider_transactions",
    ],
    "fundamentals_analyst": [
        "get_fundamentals", "get_forward_estimates",
        "get_long_term_thesis", "get_sector_peers",
    ],
    "reports_analyst": ["get_report_insights", "get_forward_estimates"],
    "macro_analyst": [
        "get_macro_data", "get_geopolitical_news", "get_supply_chain_risk",
    ],
}

MAX_EVIDENCE_REQUESTS_PER_RUN = 2
MAX_EXTRA_DEBATE_ROUNDS = 1


@dataclass(frozen=True)
class GraphNode:
    agent_key: str
    role: str
    kind: str
    optional: bool = False
    has_tools: bool = False


@dataclass(frozen=True)
class GraphEdge:
    src_role: str
    dst_role: str


@dataclass
class WorkflowSpec:
    """Default 11-agent DAG (optional QR / compliance stay off)."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    asset_class: str = "stock"

    def node_by_role(self, role: str) -> Optional[GraphNode]:
        for n in self.nodes:
            if n.role == role:
                return n
        return None

    def node_ids(self) -> List[str]:
        return [n.role for n in self.nodes]


def stock_spec() -> WorkflowSpec:
    def N(key: str, kind: str, *, optional: bool = False, has_tools: bool = False) -> GraphNode:
        return GraphNode(
            agent_key=key,
            role=AGENT_KEY_TO_ROLE[key],
            kind=kind,
            optional=optional,
            has_tools=has_tools,
        )

    analysts = [
        N("market_analyst", "analyst", has_tools=True),
        N("news_analyst", "analyst", has_tools=True),
        N("fundamentals_analyst", "analyst", has_tools=True),
        N("reports_analyst", "analyst", has_tools=True),
        N("macro_analyst", "analyst", has_tools=True),
    ]
    nodes = analysts + [
        N("bull_researcher", "bull"),
        N("bear_researcher", "bear"),
        N("research_manager", "manager"),
        N("quality_reviewer", "reviewer", optional=True),
        N("trader", "trader", has_tools=True),
        N("risk_aggressive", "risk_a"),
        N("risk_neutral", "risk_n"),
        N("risk_conservative", "risk_c"),
        N("compliance_officer", "reviewer", optional=True),
        N("portfolio_manager", "manager"),
    ]
    edges: List[GraphEdge] = []
    for a in analysts:
        edges.append(GraphEdge(a.role, "Bullish Researcher"))
        edges.append(GraphEdge(a.role, "Bearish Researcher"))
    edges += [
        GraphEdge("Bullish Researcher", "Research Manager"),
        GraphEdge("Bearish Researcher", "Research Manager"),
        GraphEdge("Research Manager", "Trader"),
        GraphEdge("Research Manager", "Quality Reviewer"),
        GraphEdge("Quality Reviewer", "Portfolio Manager"),
        GraphEdge("Trader", "Aggressive Risk Analyst"),
        GraphEdge("Trader", "Neutral Risk Analyst"),
        GraphEdge("Trader", "Conservative Risk Analyst"),
        GraphEdge("Aggressive Risk Analyst", "Compliance Officer"),
        GraphEdge("Neutral Risk Analyst", "Compliance Officer"),
        GraphEdge("Conservative Risk Analyst", "Compliance Officer"),
        GraphEdge("Aggressive Risk Analyst", "Portfolio Manager"),
        GraphEdge("Neutral Risk Analyst", "Portfolio Manager"),
        GraphEdge("Conservative Risk Analyst", "Portfolio Manager"),
        GraphEdge("Compliance Officer", "Portfolio Manager"),
        GraphEdge("Portfolio Manager", "decision"),
    ]
    return WorkflowSpec(nodes=nodes, edges=edges, asset_class="stock")


def role_to_agent_key(role: str) -> Optional[str]:
    return ROLE_TO_AGENT_KEY.get(role)
