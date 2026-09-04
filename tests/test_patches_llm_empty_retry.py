"""Tests for the empty-LLM-response retry patch in ``trading_crew/_patches.py``.

CrewAI's default experimental agent executor (and the legacy executor) call the
LLM through ``crewai.utilities.agent_utils.get_llm_response`` /
``aget_llm_response``, which raises ``ValueError: Invalid response from LLM call
- None or empty.`` on an empty model response.  That exception used to propagate
out of the executor's flow listeners and fail the whole stock-prediction cell
(``cell_failed <TICKER>``).  The patch wraps the function so a transient empty
response is retried (bounded) instead of hard-failing.
"""

from __future__ import annotations

import asyncio
import os

import pytest


@pytest.fixture
def low_backoff(monkeypatch):
    """Make retries instant so the test doesn't sleep for real."""
    monkeypatch.setenv("LLM_TASK_RETRIES", "2")
    monkeypatch.setenv("LLM_TASK_RETRY_BACKOFF", "0")


class _FakeLLM:
    def __init__(self, empty_first: bool = True):
        self.calls = 0
        self.empty_first = empty_first

    def call(self, messages, **kwargs):
        self.calls += 1
        return None if (self.empty_first and self.calls == 1) else "ok"

    async def acall(self, messages, **kwargs):
        self.calls += 1
        return None if (self.empty_first and self.calls == 1) else "ok"


class _FakePrinter:
    def print(self, **kwargs):
        pass


@pytest.fixture
def installed(monkeypatch):
    from trading_crew import _patches

    # Reinstall so the module-level bindings are the patched wrappers.
    monkeypatch.setattr(_patches, "_APPLIED_FLAG", "_tc_apply_again")
    _patches.install()
    return _patches


def test_sync_empty_response_is_retried(installed, low_backoff):
    import crewai.utilities.agent_utils as au

    llm = _FakeLLM()
    result = au.get_llm_response(llm, [], [], _FakePrinter())
    assert result == "ok"
    assert llm.calls == 2, "first empty response should be retried"


def test_sync_non_empty_response_succeeds_first_try(installed):
    import crewai.utilities.agent_utils as au

    llm = _FakeLLM(empty_first=False)
    result = au.get_llm_response(llm, [], [], _FakePrinter())
    assert result == "ok"
    assert llm.calls == 1


def test_async_empty_response_is_retried(installed, low_backoff):
    import crewai.utilities.agent_utils as au

    async def run():
        llm = _FakeLLM()
        result = await au.aget_llm_response(llm, [], [], _FakePrinter())
        return result, llm.calls

    result, calls = asyncio.run(run())
    assert result == "ok"
    assert calls == 2


def test_executor_module_bindings_are_patched(installed):
    """The executors import get_llm_response into their own namespace, so their
    module-level bindings must be the retry wrapper too."""
    import crewai.agents.crew_agent_executor as legacy
    import crewai.experimental.agent_executor as exp

    assert legacy.get_llm_response.__name__ == "_safe_get"
    assert exp.get_llm_response.__name__ == "_safe_get"
    assert legacy.aget_llm_response.__name__ == "_safe_aget"
