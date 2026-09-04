"""Graph runtime primitives: spec, evidence blackboard, scratch pad, playbook."""

from .blackboard import Blackboard, begin_run, current, drain_queued_requests, end_run
from .playbook import playbook_key, preferred_action, record_outcome
from .scratchpad import ScratchPad
from .spec import (
    AGENT_KEY_TO_ROLE,
    ANALYST_AGENT_KEYS,
    ROLE_TO_AGENT_KEY,
    WorkflowSpec,
    role_to_agent_key,
    stock_spec,
)

__all__ = [
    "AGENT_KEY_TO_ROLE",
    "ANALYST_AGENT_KEYS",
    "Blackboard",
    "ROLE_TO_AGENT_KEY",
    "ScratchPad",
    "WorkflowSpec",
    "begin_run",
    "current",
    "drain_queued_requests",
    "end_run",
    "playbook_key",
    "preferred_action",
    "record_outcome",
    "role_to_agent_key",
    "stock_spec",
]
