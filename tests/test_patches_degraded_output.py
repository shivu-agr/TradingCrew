"""Tests for the degraded-LLM-output patch in ``trading_crew/_patches.py``.

Background
----------
Some OSS chat-completions servers (notably ``gpt-oss`` on vLLM) emit a
final assistant message whose ``content`` is ``None`` and whose
``tool_calls`` field is a list of tool-call objects — i.e. the model
"answers" with raw tool calls rather than text.  CrewAI then tries to
build ``TaskOutput(raw=<list>)`` which trips Pydantic's ``string_type``
validator and aborts the whole crew.

The patch wraps ``Agent.execute_task`` so any non-string return is
replaced with a structured placeholder.  These tests verify:

* String / BaseModel returns are passed through untouched.
* A list of tool-call-shaped objects produces a placeholder that
  starts with ``DEGRADED_OUTPUT_MARKER`` and names every tool the LLM
  attempted (so the operator can decide whether to retry, switch LLM,
  or shorten the chain).
* JSON-string ``arguments`` payloads (the OpenAI SDK shape) are
  rendered as parsed dicts in the placeholder.
* Dict-shaped tool-call entries (older LiteLLM proxies) are handled.
* Unrecognised non-string shapes still get a degraded marker so the
  runner never sees raw ``repr()`` output in the Reports tab.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from trading_crew._patches import (
    DEGRADED_OUTPUT_MARKER,
    DEGRADED_TIMEOUT_MARKER,
    _format_degraded_placeholder,
    _format_timeout_placeholder,
    _looks_like_tool_call_obj,
)


def _openai_style_call(name: str, arguments: Any, call_id: str = "chatcmpl-tool-x"):
    """Mimic the shape of ``ChatCompletionMessageFunctionToolCall``."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=(
                arguments if isinstance(arguments, str) else json.dumps(arguments)
            ),
        ),
    )


def _dict_style_call(name: str, arguments: Any):
    """Mimic the LiteLLM-proxy dict shape."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": (
                arguments if isinstance(arguments, str) else json.dumps(arguments)
            ),
        },
    }


def test_looks_like_tool_call_recognises_both_shapes():
    assert _looks_like_tool_call_obj(
        _openai_style_call("get_news", {"ticker": "LT.NS"})
    )
    assert _looks_like_tool_call_obj(
        _dict_style_call("get_news", {"ticker": "LT.NS"})
    )


def test_looks_like_tool_call_rejects_unrelated_objects():
    assert not _looks_like_tool_call_obj("just a string")
    assert not _looks_like_tool_call_obj({"foo": "bar"})
    assert not _looks_like_tool_call_obj(SimpleNamespace(unrelated="x"))


def test_placeholder_starts_with_marker_and_lists_calls():
    placeholder = _format_degraded_placeholder(
        [
            _openai_style_call("get_news", {"ticker": "LT.NS"}),
            _openai_style_call(
                "get_global_news",
                {"ticker": "LT.NS", "query": "RBI repo rate"},
            ),
        ],
        task_name="news_task",
        role="News Analyst",
    )
    assert placeholder.startswith(DEGRADED_OUTPUT_MARKER), placeholder[:80]
    assert "news_task" in placeholder
    assert "News Analyst" in placeholder
    assert "get_news" in placeholder
    assert "get_global_news" in placeholder
    # Arguments should be rendered as parsed JSON, not raw quoted strings.
    assert '"ticker": "LT.NS"' in placeholder
    # Includes operator-facing mitigation hints.
    assert "closed-source" in placeholder.lower()
    # Tool-name bullet lines start with two-space indent so markdown
    # rendering keeps them inside the bulleted list when the LLM in
    # debate consumes the report.
    assert "  - get_news(" in placeholder


def test_placeholder_handles_dict_shaped_calls():
    placeholder = _format_degraded_placeholder(
        [_dict_style_call("get_news", {"ticker": "X"})],
        task_name="social_task",
        role="Social Analyst",
    )
    assert "get_news" in placeholder
    assert '"ticker": "X"' in placeholder


def test_placeholder_unparseable_arguments_kept_as_string():
    """The OpenAI SDK occasionally returns malformed JSON in ``arguments``.

    We must not crash — the raw string should appear verbatim so the
    operator can still see what the LLM tried to send.
    """
    bad = _openai_style_call("get_news", "{not json")
    placeholder = _format_degraded_placeholder(
        [bad], task_name="news_task", role="News Analyst",
    )
    assert "get_news" in placeholder
    assert "{not json" in placeholder


def test_placeholder_unknown_calls_still_produces_marker():
    placeholder = _format_degraded_placeholder(
        [], task_name="news_task", role="News Analyst",
    )
    assert placeholder.startswith(DEGRADED_OUTPUT_MARKER)
    assert "could not parse" in placeholder


# ---------------------------------------------------------------------------
# Integration with the patched Agent.execute_task
# ---------------------------------------------------------------------------


class _SamplePydanticOutput(BaseModel):
    action: str
    size: float


@pytest.fixture
def reinstalled_patch(monkeypatch):
    """Install the patch and ensure each test gets a fresh wrap so we
    can drive the inner callable per-test via ``monkeypatch``.

    We bypass ``Agent.__new__`` entirely — instantiating a CrewAI
    ``Agent`` requires a non-empty Pydantic init payload and pulling in
    a real LLM/tooling stack just to exercise a 30-line wrapper would
    be wasteful and brittle.  Instead we patch
    ``crewai.agent.Agent.execute_task`` at the class level with a stub
    inner callable, re-run ``install()``, and invoke the patched method
    as an unbound function with a ``SimpleNamespace`` standing in for
    ``self``.  This exercises the *exact* wrapping logic shipped at
    runtime without ever touching Pydantic.
    """
    import crewai.agent as cagent
    from trading_crew import _patches

    # Default the degrade tests to *no* retries so they exercise the
    # placeholder path directly and don't sleep on backoff.  The dedicated
    # retry tests below opt back in via ``monkeypatch.setenv``.
    monkeypatch.setenv("LLM_TASK_RETRIES", "0")
    monkeypatch.setenv("LLM_TASK_RETRY_BACKOFF", "0")

    original = cagent.Agent.execute_task
    setattr(cagent.Agent, _patches._APPLIED_FLAG, False)
    yield cagent, _patches
    cagent.Agent.execute_task = original
    setattr(cagent.Agent, _patches._APPLIED_FLAG, False)


def _install_with_inner(cagent, patches, inner_fn):
    """Replace the class method with ``inner_fn`` then wrap it."""
    cagent.Agent.execute_task = inner_fn  # type: ignore[assignment]
    setattr(cagent.Agent, patches._APPLIED_FLAG, False)
    patches.install()
    return cagent.Agent.execute_task


def test_patched_execute_task_passes_through_strings(reinstalled_patch):
    cagent, patches = reinstalled_patch
    inner = lambda self, task, context=None, tools=None: "a normal markdown report"  # noqa: E731
    wrapped = _install_with_inner(cagent, patches, inner)

    fake_self = SimpleNamespace(role="Market Analyst")
    out = wrapped(fake_self, task=SimpleNamespace(name="market_task"))
    assert out == "a normal markdown report"


def test_patched_execute_task_passes_through_basemodel(reinstalled_patch):
    cagent, patches = reinstalled_patch
    inner = lambda self, task, context=None, tools=None: _SamplePydanticOutput(  # noqa: E731
        action="OVERWEIGHT", size=0.05,
    )
    wrapped = _install_with_inner(cagent, patches, inner)

    fake_self = SimpleNamespace(role="Portfolio Manager")
    out = wrapped(fake_self, task=SimpleNamespace(name="pm_task"))
    assert isinstance(out, _SamplePydanticOutput)
    assert out.action == "OVERWEIGHT"


def test_patched_execute_task_substitutes_placeholder_for_tool_call_list(
    reinstalled_patch, caplog,
):
    cagent, patches = reinstalled_patch
    bad_output = [_openai_style_call("get_news", {"ticker": "LT.NS"})]
    inner = lambda self, task, context=None, tools=None: bad_output  # noqa: E731
    wrapped = _install_with_inner(cagent, patches, inner)

    fake_self = SimpleNamespace(role="News Analyst")
    with caplog.at_level("WARNING", logger="trading_crew._patches"):
        out = wrapped(fake_self, task=SimpleNamespace(name="news_task"))

    assert isinstance(out, str)
    assert out.startswith(DEGRADED_OUTPUT_MARKER)
    assert "get_news" in out
    assert any(
        "degraded" in rec.getMessage().lower() for rec in caplog.records
    )


def test_patched_execute_task_marks_unknown_shapes_as_degraded(reinstalled_patch):
    """An unrecognised non-string return (e.g. a stray dict) must still
    end up flagged so the runner never tries to render a Python ``repr``
    as a real report."""
    cagent, patches = reinstalled_patch
    inner = lambda self, task, context=None, tools=None: {"weird": "shape"}  # noqa: E731
    wrapped = _install_with_inner(cagent, patches, inner)

    fake_self = SimpleNamespace(role="News Analyst")
    out = wrapped(fake_self, task=SimpleNamespace(name="news_task"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_OUTPUT_MARKER)
    assert "weird" in out


# ---------------------------------------------------------------------------
# LLM-timeout patch (Phase 2G — Gemma-4 stall fix)
# ---------------------------------------------------------------------------


def test_timeout_placeholder_starts_with_timeout_marker():
    """Timeouts must use a distinct marker so the operator can tell at a
    glance whether a report is missing because the LLM emitted bogus
    tool calls vs because it stalled past the configured timeout."""
    placeholder = _format_timeout_placeholder(
        task_name="bull_round",
        role="Bullish Researcher",
        exc=TimeoutError("read timed out after 120s"),
    )
    assert placeholder.startswith(DEGRADED_TIMEOUT_MARKER)
    assert placeholder != ""
    assert "Bullish Researcher" in placeholder
    assert "bull_round" in placeholder
    assert "TimeoutError" in placeholder
    assert "read timed out" in placeholder
    # Operator should see actionable mitigation hints.
    assert "nvidia-smi" in placeholder
    assert "LLM_TIMEOUT_SECONDS" in placeholder


def test_timeout_placeholder_truncates_huge_exception_messages():
    huge = "x" * 10_000
    placeholder = _format_timeout_placeholder(
        task_name="bull_round", role="Bullish Researcher",
        exc=RuntimeError(huge),
    )
    # We cap the embedded message at 200 chars so a giant stack trace
    # from litellm doesn't blow out the downstream prompt budget.
    assert placeholder.count("x") <= 250


def test_timeout_and_tool_call_markers_are_distinct():
    """Distinct markers so the runner / UI / audit script can split them."""
    assert DEGRADED_TIMEOUT_MARKER != DEGRADED_OUTPUT_MARKER
    assert "TIMEOUT" in DEGRADED_TIMEOUT_MARKER
    assert "TOOL_CALL" in DEGRADED_OUTPUT_MARKER


class _FakeOpenAITimeout(Exception):
    """Stand-in for litellm.Timeout / openai.APITimeoutError."""


def test_patched_execute_task_catches_timeout_and_returns_placeholder(
    reinstalled_patch, caplog,
):
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeOpenAITimeout("Request timed out after 120.0 seconds")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Bullish Researcher")
    with caplog.at_level("WARNING", logger="trading_crew._patches"):
        out = wrapped(fake_self, task=SimpleNamespace(name="bull_round"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)
    assert "Bullish Researcher" in out
    assert "_FakeOpenAITimeout" in out
    assert any("LLM timeout" in rec.getMessage() for rec in caplog.records)


class _FakeAPIConnectionError(Exception):
    """Stand-in for openai.APIConnectionError / litellm.APIConnectionError."""


def test_patched_execute_task_catches_connection_error_as_timeout(reinstalled_patch):
    """A dropped TCP connection should be treated like a timeout — same
    "the model never gave us an answer" semantics — so the rest of the
    workflow keeps moving instead of aborting mid-run."""
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeAPIConnectionError("connection reset by peer")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="News Analyst")
    out = wrapped(fake_self, task=SimpleNamespace(name="news_task"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)
    assert "News Analyst" in out


class _FakeBusinessLogicError(Exception):
    """Unrelated error class — must NOT be swallowed by the timeout catch."""


def test_patched_execute_task_propagates_non_timeout_exceptions(reinstalled_patch):
    """Real bugs (e.g. schema validation, 4xx from the LLM, KeyError in
    a tool) must keep propagating so they show up in the run-error
    event — only timeout / connection failures should be converted to
    degraded placeholders."""
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeBusinessLogicError("something is genuinely broken")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Market Analyst")
    with pytest.raises(_FakeBusinessLogicError):
        wrapped(fake_self, task=SimpleNamespace(name="market_task"))


# ---------------------------------------------------------------------------
# Upstream HTTP 5xx — observed live as ``openai.InternalServerError:
# upstream request timeout`` on the hosted gpt-oss endpoint during the
# Quality Reviewer step.  Functionally identical to a client-side
# ``ReadTimeout`` (the model server failed to respond in time), so we
# must convert these to a degraded placeholder rather than abort the
# whole crew.
# ---------------------------------------------------------------------------


class _FakeInternalServerError(Exception):
    """Stand-in for openai.InternalServerError (HTTP 5xx wrapper)."""


class _FakeServiceUnavailableError(Exception):
    """Stand-in for openai.ServiceUnavailableError / 503."""


class _FakeRateLimitError(Exception):
    """Stand-in for openai.RateLimitError / 429."""


class _FakeBadGateway(Exception):
    """Stand-in for an SDK-shaped 502 class."""


def test_patched_execute_task_catches_upstream_internal_server_error(
    reinstalled_patch, caplog,
):
    """The hosted ``gpt-oss`` endpoint returns ``InternalServerError:
    upstream request timeout`` when the underlying model takes too long.
    This must become a degraded placeholder, not an abort."""
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeInternalServerError("upstream request timeout")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Quality Reviewer")
    with caplog.at_level("WARNING", logger="trading_crew._patches"):
        out = wrapped(fake_self, task=SimpleNamespace(name="quality_review"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)
    assert "Quality Reviewer" in out
    assert "_FakeInternalServerError" in out
    assert any("LLM timeout" in rec.getMessage() for rec in caplog.records)


def test_patched_execute_task_catches_service_unavailable(reinstalled_patch):
    """503 from the model server is also transient — degrade and continue."""
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeServiceUnavailableError("service temporarily unavailable")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="News Analyst")
    out = wrapped(fake_self, task=SimpleNamespace(name="news_task"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)


def test_patched_execute_task_catches_rate_limit_error(reinstalled_patch):
    """429 is transient — converting to a degraded placeholder lets the
    remaining agents run rather than abort the whole crew."""
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeRateLimitError("rate limit exceeded")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Market Analyst")
    out = wrapped(fake_self, task=SimpleNamespace(name="market_task"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)


def test_patched_execute_task_catches_bad_gateway(reinstalled_patch):
    """502 from any intermediate proxy is also transient."""
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeBadGateway("bad gateway")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Research Manager")
    out = wrapped(fake_self, task=SimpleNamespace(name="research_synthesis"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)


class _FakeGenericAPIError(Exception):
    """Stand-in for a generic ``APIError`` that wraps a 504 in its message
    rather than via its class name (some proxies do this)."""


def test_patched_execute_task_catches_timeout_via_message_text(reinstalled_patch):
    """When the class name doesn't match any of our patterns but the
    exception message clearly says "upstream request timeout" / "gateway
    timeout" / etc., we should still treat the failure as transient."""
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeGenericAPIError("upstream request timeout")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Quality Reviewer")
    out = wrapped(fake_self, task=SimpleNamespace(name="quality_review"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)
    assert "upstream request timeout" in out


def test_patched_execute_task_message_text_match_is_case_insensitive(reinstalled_patch):
    """Some servers shout in uppercase — we shouldn't miss them."""
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeGenericAPIError("UPSTREAM REQUEST TIMEOUT")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Quality Reviewer")
    out = wrapped(fake_self, task=SimpleNamespace(name="quality_review"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)


def test_patched_execute_task_still_propagates_unrelated_apierror(reinstalled_patch):
    """A generic APIError with NO timeout-ish message (e.g. 4xx with a
    'context length exceeded' body) must NOT be swallowed — that is a
    real bug the operator needs to see and fix."""
    cagent, patches = reinstalled_patch

    def inner(self, task, context=None, tools=None):
        raise _FakeGenericAPIError("context length exceeded")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Market Analyst")
    with pytest.raises(_FakeGenericAPIError):
        wrapped(fake_self, task=SimpleNamespace(name="market_task"))


# ---------------------------------------------------------------------------
# Retry-on-504: a transient upstream gateway error should be retried (with
# backoff) before we fall back to the degraded placeholder.
# ---------------------------------------------------------------------------


class _FakeGatewayTimeout(Exception):
    """Stand-in for an SDK-shaped 504 GatewayTimeout class."""


class _FakeReadTimeout(Exception):
    """Stand-in for a client-side httpx.ReadTimeout."""


def test_retry_then_succeed_on_upstream_504(reinstalled_patch, monkeypatch, caplog):
    """A 504 that clears on the 2nd attempt yields the real result — no
    placeholder — proving the retry actually re-runs the task."""
    cagent, patches = reinstalled_patch
    monkeypatch.setenv("LLM_TASK_RETRIES", "3")
    monkeypatch.setenv("LLM_TASK_RETRY_BACKOFF", "0")  # no sleeping in tests

    calls = {"n": 0}

    def inner(self, task, context=None, tools=None):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _FakeGatewayTimeout("upstream request timeout")
        return "recovered report"

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Quality Reviewer")
    with caplog.at_level("WARNING", logger="trading_crew._patches"):
        out = wrapped(fake_self, task=SimpleNamespace(name="quality_review"))
    assert out == "recovered report"
    assert calls["n"] == 2
    assert any("retry 1/3" in rec.getMessage() for rec in caplog.records)


def test_retry_exhausted_then_degrades(reinstalled_patch, monkeypatch):
    """A 504 that never clears is retried the configured number of times
    (total attempts = retries + 1) and then degrades."""
    cagent, patches = reinstalled_patch
    monkeypatch.setenv("LLM_TASK_RETRIES", "2")
    monkeypatch.setenv("LLM_TASK_RETRY_BACKOFF", "0")

    calls = {"n": 0}

    def inner(self, task, context=None, tools=None):
        calls["n"] += 1
        raise _FakeGatewayTimeout("upstream request timeout")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="News Analyst")
    out = wrapped(fake_self, task=SimpleNamespace(name="news_task"))
    assert isinstance(out, str)
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)
    assert calls["n"] == 3  # 1 initial + 2 retries


def test_rate_limit_is_not_retried(reinstalled_patch, monkeypatch):
    """429 is transient but retrying doesn't help our setup — it degrades
    immediately even with retries enabled (matches _common.get_llm intent)."""
    cagent, patches = reinstalled_patch
    monkeypatch.setenv("LLM_TASK_RETRIES", "3")
    monkeypatch.setenv("LLM_TASK_RETRY_BACKOFF", "0")

    calls = {"n": 0}

    def inner(self, task, context=None, tools=None):
        calls["n"] += 1
        raise _FakeRateLimitError("rate limit exceeded")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Market Analyst")
    out = wrapped(fake_self, task=SimpleNamespace(name="market_task"))
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)
    assert calls["n"] == 1  # not retried


def test_client_side_read_timeout_is_not_retried(reinstalled_patch, monkeypatch):
    """A client-side ReadTimeout means the model is still generating; a
    fresh attempt would leave a vLLM 'zombie request', so we degrade
    immediately instead of retrying (only upstream 5xx/gateway are retried)."""
    cagent, patches = reinstalled_patch
    monkeypatch.setenv("LLM_TASK_RETRIES", "3")
    monkeypatch.setenv("LLM_TASK_RETRY_BACKOFF", "0")

    calls = {"n": 0}

    def inner(self, task, context=None, tools=None):
        calls["n"] += 1
        raise _FakeReadTimeout("read timed out")

    wrapped = _install_with_inner(cagent, patches, inner)
    fake_self = SimpleNamespace(role="Macro Analyst")
    out = wrapped(fake_self, task=SimpleNamespace(name="macro_task"))
    assert out.startswith(DEGRADED_TIMEOUT_MARKER)
    assert calls["n"] == 1  # not retried
