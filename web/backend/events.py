"""Event taxonomy streamed from the runner to the browser via WebSocket.

Each event is a plain dict serialized as JSON. The frontend dispatches on
``type`` and the agent layer (callbacks in ``runner.py``) just calls
``make_event(...)``.

Event types
-----------
* ``run_started``       -> {ticker, debate_rounds, risk_rounds, agent_count, task_count, expected_order}
* ``cascade_status``    -> {regime, route, reason}          (M4 — regime router verdict)
* ``node_started``      -> {node, role, kind}               (agent about to run)
* ``node_completed``    -> {node, role, kind, output}       (task completed)
* ``agent_step``        -> {node, role, kind, content}      (LLM step in react loop)
* ``tool_call``         -> {node, tool, call_id, args}      (tool about to run)
* ``tool_result``       -> {node, tool, call_id, output, error?, elapsed_ms}
* ``final_decision``    -> {decision, raw}                  (parsed PortfolioDecision)
* ``action_proposal``   -> {proposal, markdown}             (M1 — typed ActionProposal)
* ``execution_result``  -> {order, fill, state_after, sizing, risk_gate, cost_scenarios, rejected, note}  (M2+M5)
* ``reflection_records``-> {records, final_action, final_size_pct, revised}  (M4 — Reflective Critic per-sample audit)
* ``episode_recorded``  -> {episode_id, regime, decision_ts, outcome_ts}  (M3 — episodic memory write)
* ``run_completed``     -> {usage}
* ``error``             -> {message, traceback?}
"""

import time
from typing import Any, Dict


def now_ms() -> int:
    return int(time.time() * 1000)


def make_event(event_type: str, **fields: Any) -> Dict[str, Any]:
    return {"type": event_type, "ts": now_ms(), **fields}


# Map agent role string -> "kind" for the workflow diagram.
#
# These role strings MUST match the YAML ``role:`` fields exactly because the
# CrewAI step/task callbacks identify agents by role. Stock and commodity
# crews coexist in the same map — there's no asset-class collision since
# role names are disjoint (with one exception: ``Macro Analyst`` is reused
# verbatim by both crews and intentionally maps to the same kind).
NODE_KIND: Dict[str, str] = {
    # --- stock crew (trading_crew/config/agents.yaml) ---
    "Market Analyst": "analyst",
    "Social Analyst": "analyst",
    "News Analyst": "analyst",
    "Fundamentals Analyst": "analyst",
    "Macro Analyst": "analyst",
    "Geopolitical Analyst": "analyst",
    "Sector / Peer Analyst": "analyst",
    "Quant / Options Analyst": "analyst",
    "Bullish Researcher": "bull",
    "Bearish Researcher": "bear",
    "Research Manager": "manager",
    "Quality Reviewer": "reviewer",
    "Trader": "trader",
    "Aggressive Risk Analyst": "risk_a",
    "Neutral Risk Analyst": "risk_n",
    "Conservative Risk Analyst": "risk_c",
    "Compliance Officer": "reviewer",
    "Portfolio Manager": "manager",
    # --- commodity crew (commodity_crew/config/agents.yaml) ---
    # Most personas (Quality Reviewer, risk team, Compliance Officer,
    # Portfolio Manager) share role strings with the stock crew above —
    # those are deliberately re-used since the personas are asset-agnostic.
    # Only the analysts + researchers + trader carry futures-specific names.
    "Commodity Market Analyst": "analyst",
    "Term Structure Analyst": "analyst",
    "Inventories & Stocks Analyst": "analyst",
    "Supply & Demand Analyst": "analyst",
    "Geopolitical & Supply-Risk Analyst": "analyst",
    "Positioning & Seasonality Quant": "analyst",
    "Bullish Futures Researcher": "bull",
    "Bearish Futures Researcher": "bear",
    "Futures Research Manager": "manager",
    "Senior Futures Trader": "trader",
}
