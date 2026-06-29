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
    _format_degraded_placeholder,
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
