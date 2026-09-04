"""Stock workflow spec stays aligned with runner role strings."""

from trading_crew.agentic.graph.spec import ROLE_TO_AGENT_KEY, stock_spec


def test_stock_spec_roles_are_the_runner_ids():
    spec = stock_spec()
    roles = spec.node_ids()
    assert "Market Analyst" in roles
    assert "Reports Analyst" in roles
    assert "Bullish Researcher" in roles
    assert "Portfolio Manager" in roles
    for node in spec.nodes:
        assert ROLE_TO_AGENT_KEY[node.role] == node.agent_key


def test_researcher_tools_are_registered():
    from trading_crew.tools import ALL_TOOLS, DEFAULT_AGENT_TOOLS

    for name in (
        "lookup_blackboard", "request_evidence",
        "read_scratchpad", "write_scratchpad",
        "research_web", "get_news", "get_event_proximity",
    ):
        assert name in ALL_TOOLS
    debate_tools = [
        "read_scratchpad", "write_scratchpad",
        "research_web", "get_news", "get_event_proximity",
        "get_stock_data", "get_fundamentals",
        "get_analyst_recommendations",
        "request_evidence",
    ]
    assert DEFAULT_AGENT_TOOLS["bull_researcher"] == debate_tools
    assert DEFAULT_AGENT_TOOLS["bear_researcher"] == debate_tools
    assert DEFAULT_AGENT_TOOLS["research_manager"] == [
        "read_scratchpad", "lookup_blackboard", "request_evidence",
    ]
    risk_tools = [
        "read_scratchpad", "write_scratchpad",
        "research_web", "get_news", "get_event_proximity",
        "get_stock_data", "get_fundamentals",
        "get_options_summary",
    ]
    assert DEFAULT_AGENT_TOOLS["risk_aggressive"] == risk_tools
    assert DEFAULT_AGENT_TOOLS["risk_conservative"] == risk_tools
    assert DEFAULT_AGENT_TOOLS["risk_neutral"] == risk_tools
