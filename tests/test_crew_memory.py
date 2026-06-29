"""Tests for the memory wiring inside ``trading_crew.crew.TradingCrew``.

The crew gates LLM-enriched memory behind the ``memory`` constructor
flag.  When on, it constructs a CrewAI ``Memory`` object pointed at
our local ``get_llm()`` and our truncating embedder, then overrides
the default 1-worker ``_save_pool`` with a parallel pool so saves
don't serialize.  These tests pin both behaviors.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest


def _ensure_env(monkeypatch):
    """The embedder reads ``VLLM_EMBEDDING_*`` at construction time
    (see ``trading_crew/embedder.py``).  Tests must set them or
    Memory's embedder factory will refuse to build."""
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://example.com")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "abc")


def test_memory_save_pool_overridden_to_parallel(monkeypatch):
    """When memory is enabled, the crew must replace CrewAI's default
    1-worker ``_save_pool`` with a parallel one so concurrent
    ``remember()`` calls fan out instead of serializing.  Without this
    override the analyze calls would queue up to ~17 min on a single
    ticker run."""
    _ensure_env(monkeypatch)
    monkeypatch.setenv("TRADINGCREW_MEMORY_SAVE_WORKERS", "8")

    from trading_crew.crew import TradingCrew

    crew = TradingCrew(ticker="NTNX", debate_rounds=1, risk_rounds=1, memory=True).crew()

    assert crew.memory is not None and crew.memory is not True, \
        "Crew should hold the actual Memory instance we constructed, not True"
    assert isinstance(crew.memory._save_pool, ThreadPoolExecutor)
    assert crew.memory._save_pool._max_workers == 8, \
        "Save pool must run with 8 parallel workers"


def test_memory_save_pool_env_var_override(monkeypatch):
    """``TRADINGCREW_MEMORY_SAVE_WORKERS`` controls the pool size so
    operators can dial it for endpoints with different rate limits
    without redeploying."""
    _ensure_env(monkeypatch)
    monkeypatch.setenv("TRADINGCREW_MEMORY_SAVE_WORKERS", "3")

    from trading_crew.crew import TradingCrew

    crew = TradingCrew(ticker="NTNX", debate_rounds=1, risk_rounds=1, memory=True).crew()
    assert crew.memory._save_pool._max_workers == 3


def test_memory_disabled_means_no_memory_object(monkeypatch):
    """When the UI toggle is OFF, the crew must NOT construct a Memory
    object — keeping the run free of any analyze calls and any
    embedding writes (zero LLM overhead from memory)."""
    _ensure_env(monkeypatch)

    from trading_crew.crew import TradingCrew

    crew = TradingCrew(ticker="NTNX", debate_rounds=1, risk_rounds=1, memory=False).crew()
    assert not crew.memory, \
        "Crew with memory=False must not carry a Memory instance " \
        f"(got {type(crew.memory).__name__})"


def test_memory_uses_local_llm_not_default(monkeypatch):
    """The Memory must be wired to OUR local LLM, not CrewAI's default
    ``"gpt-4o-mini"`` string (which would lazily build an OpenAI LLM
    and 401 against the placeholder ``OPENAI_API_KEY=dummy``)."""
    _ensure_env(monkeypatch)

    from trading_crew._common import get_llm
    from trading_crew.crew import TradingCrew

    crew = TradingCrew(ticker="NTNX", debate_rounds=1, risk_rounds=1, memory=True).crew()
    # ``Memory.llm`` is whatever we passed.  Two things must hold:
    #   1. it's NOT the default string "gpt-4o-mini"
    #   2. it shares its model_name with what ``get_llm()`` returns
    assert not isinstance(crew.memory.llm, str), \
        "Memory should hold a concrete LLM instance, not the default string"
    assert crew.memory.llm.model == get_llm().model, \
        f"Memory LLM model {crew.memory.llm.model!r} does not match get_llm() {get_llm().model!r}"
