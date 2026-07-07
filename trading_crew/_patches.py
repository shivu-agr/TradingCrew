"""Runtime patches for CrewAI quirks we've hit on OSS chat-completions servers.

Importing this module installs the patches once (idempotent). It is imported
from :mod:`trading_crew._common`, so every entry point (CLI, web backend,
notebook) gets the patch automatically.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable, List, Tuple

logger = logging.getLogger(__name__)

_APPLIED_FLAG = "_tc_runtime_patches_applied"

# Sentinel prefix the runner / UI can use to recognise a placeholder
# produced by the patch.  Anything that *starts with* this marker is a
# degraded analyst output (the LLM emitted tool-call objects as its
# final answer instead of synthesised text) — the rest of the report
# body explains which tool calls the LLM attempted, so the operator
# can decide whether to retry, switch LLM, or shorten the tool chain.
DEGRADED_OUTPUT_MARKER = "[[DEGRADED_TOOL_CALL_OUTPUT]]"

# Separate marker for "LLM stalled and we gave up after the configured
# timeout".  Same downstream semantics as ``DEGRADED_OUTPUT_MARKER`` —
# the runner's UI / persistence layer treats anything starting with
# either marker as a degraded report — but we keep them distinct so
# operators can tell at a glance which failure mode they hit.
DEGRADED_TIMEOUT_MARKER = "[[DEGRADED_LLM_TIMEOUT]]"


def _looks_like_tool_call_obj(item: Any) -> bool:
    """Heuristic — does ``item`` look like an OpenAI/LiteLLM tool-call dataclass?

    We don't import the OpenAI SDK here (the package version varies),
    so we duck-type: the object exposes a ``function`` attribute with a
    ``name``, or it is a dict with a ``"function"`` key.  Robust to
    both ``ChatCompletionMessageFunctionToolCall`` (newer OpenAI SDK)
    and the older dict-shaped variants emitted by LiteLLM proxies.
    """
    fn = getattr(item, "function", None)
    if fn is not None and getattr(fn, "name", None):
        return True
    if isinstance(item, dict):
        return isinstance(item.get("function"), (dict,)) or "tool_calls" in item
    return False


def _extract_tool_call_summary(items: Iterable[Any]) -> List[Tuple[str, str]]:
    """Return ``[(tool_name, pretty_args), ...]`` from a tool-call list."""
    summary: List[Tuple[str, str]] = []
    for item in items:
        name: str = ""
        raw_args: Any = None
        fn = getattr(item, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "") or ""
            raw_args = getattr(fn, "arguments", None)
        elif isinstance(item, dict):
            fn_dict = item.get("function") or {}
            name = fn_dict.get("name") or ""
            raw_args = fn_dict.get("arguments")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                # Leave as-is — the raw string is still informative.
                pass
        pretty = (
            json.dumps(raw_args, sort_keys=True)
            if not isinstance(raw_args, str)
            else raw_args
        )
        summary.append((name or "<unknown>", pretty or ""))
    return summary


def _format_timeout_placeholder(
    *,
    task_name: str,
    role: str,
    exc: BaseException,
) -> str:
    """Build a placeholder for an LLM-timeout / connection-error failure.

    The runner / UI recognises the :data:`DEGRADED_TIMEOUT_MARKER`
    prefix and flags the per-role report as degraded so downstream
    debate / research-manager prompts can see "this analyst is missing"
    instead of staring at an empty string.
    """
    exc_type = type(exc).__name__
    exc_msg = str(exc) or "(no message)"
    return (
        f"{DEGRADED_TIMEOUT_MARKER}\n"
        f"**Analyst report missing — LLM timed out.**\n\n"
        f"The model serving `{role}` on task `{task_name}` did not "
        f"return a complete answer within the configured timeout.\n\n"
        f"Failure: `{exc_type}: {exc_msg[:200]}`\n\n"
        f"Common causes:\n"
        f"  - GPU is saturated by another workload (check `nvidia-smi`).\n"
        f"  - vLLM is queued behind earlier concurrent requests.\n"
        f"  - The prompt is unusually long for this model's window.\n\n"
        f"Suggested mitigations:\n"
        f"  - Re-run the workflow on a less loaded endpoint.\n"
        f"  - Pick a different LLM from the sidebar dropdown.\n"
        f"  - Raise `LLM_TIMEOUT_SECONDS` in `.env` if the model is\n"
        f"    genuinely slow but eventually correct."
    )


def _format_degraded_placeholder(
    items: List[Any],
    *,
    task_name: str,
    role: str,
) -> str:
    """Build a human-readable + downstream-LLM-readable placeholder.

    The body intentionally starts with :data:`DEGRADED_OUTPUT_MARKER`
    so the runner can detect the case at task-completion time and
    flag the per-role report as degraded in the WebSocket stream.
    """
    calls = _extract_tool_call_summary(items)
    if calls:
        bullets = "\n".join(
            f"  - {name}({args})" for name, args in calls
        )
    else:
        bullets = "  - (could not parse tool-call list)"
    return (
        f"{DEGRADED_OUTPUT_MARKER}\n"
        f"**Analyst report missing — degraded LLM output.**\n\n"
        f"The model emitted a tool-call list as its final answer "
        f"on `{task_name}` (role `{role}`) instead of a synthesised "
        f"markdown report.  This is a known limitation of some "
        f"open-source LLMs on tool-heavy tasks: the model fails to "
        f"compose a closing assistant message after issuing a tool "
        f"call, so the agent loop terminates with the tool-call "
        f"objects in the final-answer slot.\n\n"
        f"Attempted tool calls:\n{bullets}\n\n"
        f"Suggested mitigations:\n"
        f"  - Try a closed-source LLM via the LLM picker (the OSS\n"
        f"    preset is the usual offender).\n"
        f"  - Re-run the workflow — the failure is non-deterministic\n"
        f"    so a retry often produces a clean report.\n"
        f"  - Disable or shorten the tool chain for this analyst\n"
        f"    (Advanced settings -> Tools)."
    )


def _coerce_task_output_raw_to_str() -> None:
    """Stringify non-string agent results before ``TaskOutput`` is built.

    Why
    ---
    Some OSS chat-completions servers (notably ``gpt-oss`` on vLLM) sometimes
    emit a final assistant message whose content is a *list of
    ``ChatCompletionMessageFunctionToolCall`` objects* instead of a string -
    i.e. the model "answers" with raw tool-call objects rather than text.
    CrewAI's ``Task._execute_core`` then constructs ``TaskOutput(raw=<list>)``,
    which trips Pydantic's ``string_type`` validator and aborts the *whole*
    crew with::

        pydantic_core._pydantic_core.ValidationError: 1 validation error for
        TaskOutput raw  Input should be a valid string [type=string_type, ...]

    What this patch does
    --------------------
    Wraps ``Agent.execute_task`` at the class level so any non-string return
    value is coerced to a *structured placeholder* (see
    :func:`_format_degraded_placeholder`) before it reaches ``TaskOutput``.
    The placeholder starts with :data:`DEGRADED_OUTPUT_MARKER` so the runner
    can flag the resulting per-role report as degraded in the WebSocket
    stream / saved run record.  A WARNING is logged for the operator.

    Previous behaviour ``str(<list>)`` produced unreadable Python ``repr``
    output that ended up in the Reports tab as garbage; the structured
    placeholder both tells the operator exactly what happened and gives
    downstream agents (debate, research manager) a clean "this report is
    missing" signal instead of asking them to parse a tool-call ``repr``.

    The patch is idempotent.
    """
    import crewai.agent as _cagent
    from pydantic import BaseModel as _BaseModel

    if getattr(_cagent.Agent, _APPLIED_FLAG, False):
        return

    _orig_execute_task = _cagent.Agent.execute_task

    # Exception classes that mean "the LLM took too long, could not be
    # reached, OR the upstream model server failed to respond" — we want
    # to convert these into a degraded placeholder so the rest of the
    # workflow keeps moving instead of hanging or aborting.
    # ``litellm`` raises a chain of openai-shaped exceptions; we duck-type
    # on the class name to stay version-agnostic.
    _TIMEOUT_NAME_PATTERNS = (
        # ---- Pure client-side timeouts ------------------------------
        "Timeout",          # litellm.Timeout, openai.APITimeoutError, etc.
        "TimeoutError",     # asyncio / httpx
        "ReadTimeout",      # httpx.ReadTimeout
        "ConnectTimeout",   # httpx.ConnectTimeout
        # ---- Connection-layer failures ------------------------------
        "APIConnectionError",  # openai.APIConnectionError, litellm.APIConnectionError
        "ConnectionError",     # generic
        # ---- Upstream HTTP 5xx and rate-limit failures --------------
        # The hosted ``gpt-oss`` endpoint surfaces a slow LLM generation
        # as ``openai.InternalServerError: upstream request timeout``
        # (an HTTP 504 wrapped in the SDK's 500-class).  Functionally
        # identical to a ``ReadTimeout`` from our side — the model
        # server failed to respond in time — so we treat it as a
        # transient failure that should degrade the per-role report
        # rather than abort the whole crew.  Observed live on a
        # BAJAJ-AUTO.NS run at the Quality Reviewer step.
        "InternalServerError",       # openai.InternalServerError (HTTP 5xx)
        "ServiceUnavailableError",   # openai / litellm 503
        "BadGateway",                # 502 (sometimes a separate class)
        "GatewayTimeout",            # 504
        "RateLimitError",            # 429 — transient; retry doesn't help us
    )

    # Message fragments that prove an exception is a transient upstream
    # failure even if its class name doesn't match — e.g. some proxies
    # wrap a 504 as a generic ``APIError`` with the gateway message in
    # the body.  Kept lowercase + matched case-insensitively.
    _TIMEOUT_MESSAGE_FRAGMENTS = (
        "upstream request timeout",
        "upstream connect error",
        "upstream timeout",
        "service unavailable",
        "gateway timeout",
        "request timed out",
        "read timed out",
    )

    def _looks_like_timeout(exc: BaseException) -> bool:
        for cls in type(exc).__mro__:
            if any(p in cls.__name__ for p in _TIMEOUT_NAME_PATTERNS):
                return True
        msg = (str(exc) or "").lower()
        return any(f in msg for f in _TIMEOUT_MESSAGE_FRAGMENTS)

    def _safe_execute_task(self, task, context=None, tools=None):  # type: ignore[no-redef]
        try:
            result = _orig_execute_task(self, task=task, context=context, tools=tools)
        except BaseException as exc:  # noqa: BLE001 — we re-raise non-timeout errors
            task_name = getattr(task, "name", None) or getattr(task, "description", "<task>")
            role = getattr(self, "role", "<agent>")
            if _looks_like_timeout(exc):
                logger.warning(
                    "LLM timeout on task %r (agent=%r): %s — substituting "
                    "timeout placeholder so the run can continue.",
                    str(task_name)[:120], str(role)[:80], type(exc).__name__,
                )
                return _format_timeout_placeholder(
                    task_name=str(task_name)[:120],
                    role=str(role)[:80],
                    exc=exc,
                )
            # Anything else (model returned 4xx, schema validation, etc.)
            # is a real bug we want surfaced — preserve the original raise.
            raise
        # str  -> agent's plain-text final answer (the common case).
        # BaseModel -> task uses output_pydantic / output_json; CrewAI's
        #              Task._execute_core handles BaseModel natively, so we
        #              MUST NOT stringify it here or we drop the structured
        #              decision payload.
        # other -> almost always a list of ChatCompletionMessageFunctionToolCall
        #          objects from a misbehaving OSS LLM.  Replace with a
        #          structured placeholder so the run continues but the
        #          failure mode is visible end-to-end.
        if not isinstance(result, (str, _BaseModel)):
            task_name = getattr(task, "name", None) or getattr(task, "description", "<task>")
            role = getattr(self, "role", "<agent>")
            items = list(result) if isinstance(result, (list, tuple)) else [result]
            looks_toolcally = all(_looks_like_tool_call_obj(i) for i in items) if items else False
            if looks_toolcally:
                logger.warning(
                    "Analyst output degraded on task %r (agent=%r): OSS LLM "
                    "emitted %d tool-call object(s) as final answer; "
                    "substituting structured placeholder.",
                    task_name, role, len(items),
                )
                result = _format_degraded_placeholder(
                    items, task_name=str(task_name)[:120], role=str(role)[:80],
                )
            else:
                # Unknown non-string shape — keep the prior best-effort
                # behaviour so we don't lose any information the user
                # might be able to act on, but flag it as degraded so
                # the UI doesn't render the repr as a "report".
                logger.warning(
                    "TaskOutput.raw coerced from %s -> str on task %r "
                    "(agent=%r); unrecognised final-answer shape.",
                    type(result).__name__, task_name, role,
                )
                result = f"{DEGRADED_OUTPUT_MARKER}\n{result!r}"
        return result

    _cagent.Agent.execute_task = _safe_execute_task  # type: ignore[assignment]
    setattr(_cagent.Agent, _APPLIED_FLAG, True)
    logger.info("Installed CrewAI runtime patch: Agent.execute_task str-coercion")


def install() -> None:
    """Apply all runtime patches. Idempotent."""
    _coerce_task_output_raw_to_str()
