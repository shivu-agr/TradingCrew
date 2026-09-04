"""Runtime patches for CrewAI quirks we've hit on OSS chat-completions servers.

Importing this module installs the patches once (idempotent). It is imported
from :mod:`trading_crew._common`, so every entry point (CLI, web backend,
notebook) gets the patch automatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from typing import Any, Iterable, List, Tuple

logger = logging.getLogger(__name__)

_APPLIED_FLAG = "_tc_runtime_patches_applied"


def _task_retry_attempts() -> int:
    """How many *extra* attempts to make on a transient upstream 504/gateway
    failure before degrading the task (env ``LLM_TASK_RETRIES``, default 2)."""
    try:
        return max(0, int(os.environ.get("LLM_TASK_RETRIES", "2")))
    except ValueError:
        return 2


def _task_retry_backoff() -> float:
    """Base seconds for exponential backoff between task retries
    (env ``LLM_TASK_RETRY_BACKOFF``, default 5s)."""
    try:
        return max(0.0, float(os.environ.get("LLM_TASK_RETRY_BACKOFF", "5")))
    except ValueError:
        return 5.0


def _task_startup_wait_sec() -> float:
    """Bounded seconds to keep retrying while the LLM server is still loading /
    starting (503 "Loading model") or unreachable (connection error) before
    degrading the task (env ``LLM_STARTUP_WAIT_SEC``, default 600s).

    A llama.cpp server restart leaves the port unresponsive for minutes while
    the model loads + KV cache initialises; the default 2-retry budget is far
    too short, which is what produced the "Failed to convert text into a
    Pydantic model due to error: Failed to connect to ..." cell failures.
    """
    try:
        return max(0.0, float(os.environ.get("LLM_STARTUP_WAIT_SEC", "600")))
    except ValueError:
        return 600.0


_LOADING_MODEL_FRAGMENTS = (
    "loading model",
    "model is loading",
)


def _is_loading_model(exc: BaseException) -> bool:
    """True iff the server is still loading (llama.cpp 503 ``Loading model`` /
    ``unavailable_error``).  Safe to keep retrying: the request never reached
    the generation engine, so there is no orphaned in-flight request.
    """
    msg = (str(exc) or "").lower()
    if "503" not in str(exc) and "unavailable_error" not in str(exc):
        return False
    return any(f in msg for f in _LOADING_MODEL_FRAGMENTS)


def _wait_for_server_ready(
    self,
    task,
    context=None,
    tools=None,
    *,
    orig_execute_task,
    task_name: str,
    role: str,
    backoff: float,
) -> Any:
    """Bounded retry while the LLM server is still loading.

    Returns the task result if a retry eventually succeeds, else ``None``
    (caller then degrades).  Only retries on the unambiguous ``Loading model``
    condition (safe — no zombie request); any other error returns ``None`` so
    the caller applies its generic handling.
    """
    deadline = time.time() + _task_startup_wait_sec()
    n = 0
    while time.time() < deadline:
        try:
            return orig_execute_task(self, task=task, context=context, tools=tools)
        except BaseException as exc:  # noqa: BLE001
            if not _is_loading_model(exc):
                return None  # not a startup condition — caller handles it.
        n += 1
        delay = min(backoff * (2 ** min(n, 4)), 30.0) + random.uniform(0, backoff)
        logger.warning(
            "LLM server still loading on task %r (agent=%r): %s "
            "— waiting for readiness (%.1fs).",
            str(task_name)[:120], str(role)[:80], type(exc).__name__, delay,
        )
        time.sleep(delay)
    logger.warning(
        "gave up waiting for LLM server readiness on task %r (agent=%r) "
        "after %.0fs.",
        str(task_name)[:120], str(role)[:80], _task_startup_wait_sec(),
    )
    return None

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
        # Pure connection-layer failures ("Failed to connect to OpenAI API:
        # Connection error." etc).  These are transient blips where the client
        # never established/reached the server — NO in-flight request exists
        # server-side, so retrying cannot create a vLLM zombie (unlike a
        # client-side read-timeout).  Recognised here so a ``ConverterError``
        # wrapping such a blip is treated as a retryable timeout rather than
        # falling through to a hard re-raise (see the 08-02 AXISBANK.NS run).
        "failed to connect",
        "connection error",
        "connection refused",
        "connection reset",
        "connection aborted",
        "connection failed",
        "network unreachable",
        "could not connect",
        "connect error",
        "error while connecting",
    )

    def _looks_like_timeout(exc: BaseException) -> bool:
        for cls in type(exc).__mro__:
            if any(p in cls.__name__ for p in _TIMEOUT_NAME_PATTERNS):
                return True
        msg = (str(exc) or "").lower()
        return any(f in msg for f in _TIMEOUT_MESSAGE_FRAGMENTS)

    # Only *safe-to-retry* transient failures are worth retrying: (a) upstream
    # gateway failures (HTTP 502/503/504 — the hosted endpoint wraps its 504 as
    # ``InternalServerError: upstream request timeout``) where the previous
    # request already failed at the gateway, and (b) pure connection errors
    # (see ``_is_retryable_connection``) where the client never reached the
    # server.  Neither leaves a vLLM "zombie request".  Client-side
    # *read-timeouts* (the request WAS dispatched and the server may still be
    # generating) remain deliberately non-retryable — retrying those spawns a
    # zombie; see ``_common.get_llm`` max_retries=0.
    _RETRYABLE_UPSTREAM_NAME_PATTERNS = (
        "GatewayTimeout",           # 504
        "BadGateway",               # 502
        "ServiceUnavailableError",  # 503
    )
    _RETRYABLE_MESSAGE_FRAGMENTS = (
        "upstream request timeout",
        "gateway timeout",
        "bad gateway",
        "upstream connect error",
        "service unavailable",
        " 502",
        " 503",
        " 504",
    )

    def _is_retryable_upstream(exc: BaseException) -> bool:
        for cls in type(exc).__mro__:
            if any(p in cls.__name__ for p in _RETRYABLE_UPSTREAM_NAME_PATTERNS):
                return True
        msg = (str(exc) or "").lower()
        return any(f in msg for f in _RETRYABLE_MESSAGE_FRAGMENTS)

    # Pure *connection* failures (surgical retry set).  Distinct from
    # read-timeouts / 5xx: the client never reached the server (or the
    # transport died before a complete request), so there is NO orphaned
    # in-flight request server-side and retrying cannot spawn a vLLM zombie.
    # This is the fix for the transient "Failed to connect to OpenAI API:
    # Connection error." that hard-failed AXISBANK.NS on 2026-08-02.
    _CONNECTION_NAME_PATTERNS = (
        "ConnectionError",
        "APIConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "ConnectionAbortedError",
    )
    _CONNECTION_MESSAGE_FRAGMENTS = (
        "failed to connect",
        "connection error",
        "connection refused",
        "connection reset",
        "connection aborted",
        "connection failed",
        "network unreachable",
        "could not connect",
        "connect error",
    )

    def _is_retryable_connection(exc: BaseException) -> bool:
        for cls in type(exc).__mro__:
            if any(p in cls.__name__ for p in _CONNECTION_NAME_PATTERNS):
                return True
        msg = (str(exc) or "").lower()
        return any(f in msg for f in _CONNECTION_MESSAGE_FRAGMENTS)

    def _safe_execute_task(self, task, context=None, tools=None):  # type: ignore[no-redef]
        task_name = getattr(task, "name", None) or getattr(task, "description", "<task>")
        role = getattr(self, "role", "<agent>")
        max_attempts = _task_retry_attempts()
        backoff = _task_retry_backoff()
        result = None
        for attempt in range(max_attempts + 1):
            try:
                result = _orig_execute_task(self, task=task, context=context, tools=tools)
                break
            except BaseException as exc:  # noqa: BLE001 — we re-raise non-timeout errors
                if _looks_like_timeout(exc):
                    # Retry transient upstream 504/gateway errors AND pure
                    # connection errors with exponential backoff + jitter
                    # before giving up.  Both are safe to retry: the previous
                    # request never reached the server's generation engine (or
                    # failed at the gateway), so a fresh attempt does NOT leave
                    # a vLLM "zombie request" (unlike retrying a client-side
                    # read-timeout, which the code deliberately avoids).
                    if _is_loading_model(exc):
                        # Server is (re)starting / model still loading — the
                        # short retry budget below is too small for a
                        # multi-minute load window.  Wait (bounded) for
                        # readiness before giving up; this prevents the
                        # "Failed to convert text into a Pydantic model due
                        # to error: Failed to connect ..." cell failures
                        # during a server restart.
                        waited = _wait_for_server_ready(
                            self, task, context, tools,
                            orig_execute_task=_orig_execute_task,
                            task_name=str(task_name)[:120],
                            role=str(role)[:80],
                            backoff=backoff,
                        )
                        if waited is not None:
                            result = waited
                            break
                        logger.warning(
                            "LLM server never became ready on task %r "
                            "(agent=%r) — degrading after bounded startup wait.",
                            str(task_name)[:120], str(role)[:80],
                        )
                        return _format_timeout_placeholder(
                            task_name=str(task_name)[:120],
                            role=str(role)[:80],
                            exc=exc,
                        )
                    if _is_retryable_upstream(exc) and attempt < max_attempts:
                        delay = backoff * (2 ** attempt) + random.uniform(0, backoff)
                        logger.warning(
                            "Upstream gateway error on task %r (agent=%r): %s — "
                            "retry %d/%d in %.1fs.",
                            str(task_name)[:120], str(role)[:80],
                            type(exc).__name__, attempt + 1, max_attempts, delay,
                        )
                        time.sleep(delay)
                        continue
                    if _is_retryable_connection(exc) and attempt < max_attempts:
                        delay = backoff * (2 ** attempt) + random.uniform(0, backoff)
                        logger.warning(
                            "Transient connection error on task %r (agent=%r): %s "
                            "— retry %d/%d in %.1fs.",
                            str(task_name)[:120], str(role)[:80],
                            type(exc).__name__, attempt + 1, max_attempts, delay,
                        )
                        time.sleep(delay)
                        continue
                    logger.warning(
                        "LLM timeout on task %r (agent=%r): %s — substituting "
                        "timeout placeholder so the run can continue%s.",
                        str(task_name)[:120], str(role)[:80], type(exc).__name__,
                        f" (after {attempt} retr{'y' if attempt == 1 else 'ies'})"
                        if attempt else "",
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


def _patch_llm_empty_response_retry() -> None:
    """Retry a transient *empty* LLM response instead of hard-failing a cell.

    Why
    ---
    CrewAI 1.15.2's default (experimental) agent executor — and the legacy
    ``CrewAgentExecutor`` — call the LLM through
    ``crewai.utilities.agent_utils.get_llm_response`` / ``aget_llm_response``,
    which raises ``ValueError: Invalid response from LLM call - None or empty.``
    whenever the model returns empty content.  On an OSS chat-completions
    server that is frequently a transient hiccup — retrying the same prompt
    succeeds.  Without a retry here, the exception propagates out of the
    executor's flow listeners (``call_llm_and_parse`` / ``call_llm_native_tools``,
    logged as ``crewai.flow.runtime``) and fails the whole stock-prediction cell
    (surfaced as ``cell_failed <TICKER>``).

    What this patch does
    --------------------
    Both executors import ``get_llm_response`` into their own module namespace
    (``from crewai.utilities.agent_utils import get_llm_response``) and call it
    as a module-level function, so a plain reassign of the *source* module's
    attribute would not be seen.  We therefore reassign the module-level binding
    in each executor module to a bounded retry wrapper (idempotent).
    """
    import crewai.agents.crew_agent_executor as _legacy_exec
    import crewai.experimental.agent_executor as _exp_exec
    import crewai.utilities.agent_utils as _agent_utils

    if getattr(_agent_utils, "_tc_llm_empty_retry_applied", False):
        return

    _orig_get = _agent_utils.get_llm_response
    _orig_aget = _agent_utils.aget_llm_response

    def _safe_get(*args, **kwargs):
        max_attempts = _task_retry_attempts()
        backoff = _task_retry_backoff()
        for attempt in range(max_attempts + 1):
            try:
                return _orig_get(*args, **kwargs)
            except ValueError as exc:  # noqa: BLE001 — re-raise non-empty errors
                if "None or empty" in str(exc) and attempt < max_attempts:
                    delay = backoff * (2 ** attempt) + random.uniform(0, backoff)
                    logger.warning(
                        "LLM returned empty response; retry %d/%d in %.1fs.",
                        attempt + 1, max_attempts, delay,
                    )
                    time.sleep(delay)
                    continue
                raise

    async def _safe_aget(*args, **kwargs):
        max_attempts = _task_retry_attempts()
        backoff = _task_retry_backoff()
        for attempt in range(max_attempts + 1):
            try:
                return await _orig_aget(*args, **kwargs)
            except ValueError as exc:  # noqa: BLE001 — re-raise non-empty errors
                if "None or empty" in str(exc) and attempt < max_attempts:
                    delay = backoff * (2 ** attempt) + random.uniform(0, backoff)
                    logger.warning(
                        "LLM returned empty response; retry %d/%d in %.1fs.",
                        attempt + 1, max_attempts, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    # Source module (harmless, in case something calls it via the module path).
    _agent_utils.get_llm_response = _safe_get
    _agent_utils.aget_llm_response = _safe_aget
    # Executor modules hold their own imported binding — patch those too.
    _exp_exec.get_llm_response = _safe_get
    _legacy_exec.get_llm_response = _safe_get
    _legacy_exec.aget_llm_response = _safe_aget
    setattr(_agent_utils, "_tc_llm_empty_retry_applied", True)
    logger.info("Installed CrewAI runtime patch: LLM empty-response retry")


def install() -> None:
    """Apply all runtime patches. Idempotent."""
    _coerce_task_output_raw_to_str()
    _patch_llm_empty_response_retry()
