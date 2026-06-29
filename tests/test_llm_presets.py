"""Tests for the UI-driven LLM preset catalog + thread-local override."""

from __future__ import annotations

import os
import threading
from unittest import mock

import pytest

from trading_crew import llm_presets


def test_builtin_presets_have_required_fields() -> None:
    for pid, preset in llm_presets.BUILTIN_PRESETS.items():
        assert preset.id == pid, f"{pid!r} id field mismatch"
        assert preset.label, f"{pid!r} missing label"
        assert preset.kind in ("open-source", "closed-source"), f"{pid!r} bad kind"
        assert preset.provider, f"{pid!r} missing provider"
        # Open-source vLLM presets are env-driven and may have model=None
        # until the user populates VLLM_LLM_MODEL / LOCAL_LLM_MODEL.
        # Closed-source presets ship their model statically.
        if preset.kind == "closed-source":
            assert preset.model, f"{pid!r} closed-source preset missing static model"


def test_list_presets_marks_local_configured_when_model_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local-vllm preset only lights up when ``LOCAL_LLM_MODEL`` is set
    in .env — without that it would route at a server with no model loaded."""
    monkeypatch.setenv("LOCAL_LLM_MODEL", "my-local-model")
    presets = {p["id"]: p for p in llm_presets.list_presets()}
    assert presets["local-vllm"]["api_key_configured"] is True

    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    presets = {p["id"]: p for p in llm_presets.list_presets()}
    assert presets["local-vllm"]["api_key_configured"] is False


def test_list_presets_marks_closed_source_unconfigured_when_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_PROD_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    presets = {p["id"]: p for p in llm_presets.list_presets()}
    assert presets["openai-gpt-4o-mini"]["api_key_configured"] is False
    assert presets["anthropic-claude-sonnet"]["api_key_configured"] is False


def test_list_presets_marks_closed_source_configured_with_real_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_PROD_API_KEY", "sk-real-not-dummy")
    presets = {p["id"]: p for p in llm_presets.list_presets()}
    assert presets["openai-gpt-4o-mini"]["api_key_configured"] is True


def test_list_presets_treats_dummy_value_as_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dummy`` is a placeholder used by the local vLLM client — closed-source
    presets that resolve to it must NOT light up as configured."""
    monkeypatch.setenv("OPENAI_PROD_API_KEY", "dummy")
    presets = {p["id"]: p for p in llm_presets.list_presets()}
    assert presets["openai-gpt-4o-mini"]["api_key_configured"] is False


def test_set_active_applies_overrides_to_get_llm_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """The thread-local override returned by ``get_active_overrides`` must
    contain the preset's model / base_url / provider, sourced from the
    real env at activation time."""
    monkeypatch.setenv("OPENAI_PROD_API_KEY", "sk-real-key")
    llm_presets.set_active("openai-gpt-4o-mini")
    try:
        ov = llm_presets.get_active_overrides()
        assert ov["model"] == "gpt-4o-mini"
        assert ov["provider"] == "openai"
        assert ov["api_key"] == "sk-real-key"
        assert llm_presets.get_active_preset_id() == "openai-gpt-4o-mini"
    finally:
        llm_presets.clear_active()
    assert llm_presets.get_active_overrides() == {}
    assert llm_presets.get_active_preset_id() is None


def test_set_active_is_thread_local() -> None:
    """Concurrent WS sessions on different threads must each see their own
    preset.  This is the invariant that lets two crews run on two different
    LLMs in the same process."""
    results: dict[str, str | None] = {}
    barrier = threading.Barrier(2)

    def worker(name: str, preset_id: str) -> None:
        llm_presets.set_active(preset_id)
        # Wait so both threads have set their preset before we read.
        barrier.wait()
        results[name] = llm_presets.get_active_preset_id()
        llm_presets.clear_active()

    t1 = threading.Thread(target=worker, args=("t1", "local-vllm"))
    t2 = threading.Thread(target=worker, args=("t2", "hosted-vllm-oss"))
    t1.start(); t2.start()
    t1.join(); t2.join()
    assert results == {"t1": "local-vllm", "t2": "hosted-vllm-oss"}


def test_set_active_unknown_id_clears() -> None:
    llm_presets.set_active("hosted-vllm-oss")
    llm_presets.set_active("does-not-exist")
    assert llm_presets.get_active_preset_id() is None
    llm_presets.clear_active()


def test_get_llm_consults_active_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_llm()`` must apply the preset overlay BEFORE constructing the LLM,
    so the resulting ``model`` reflects the UI choice rather than the .env
    default."""
    # Stub crewai.LLM so the test doesn't need a real network endpoint.
    captured: dict[str, object] = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    # Patch crewai.LLM at import time inside _common.get_llm.
    monkeypatch.setattr("crewai.LLM", _FakeLLM, raising=False)

    # Pretend the .env points at the local vLLM (so we can detect the swap).
    monkeypatch.setenv("VLLM_LLM_BASE_URL", "http://default.local/v1")
    monkeypatch.setenv("VLLM_LLM_MODEL", "default-oss")
    monkeypatch.setenv("VLLM_LLM_API_KEY", "default-key")
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "fallback")
    monkeypatch.setenv("OPENAI_PROD_API_KEY", "sk-real")

    from trading_crew._common import get_llm

    # No preset -> .env default flows through.
    llm_presets.clear_active()
    captured.clear()
    get_llm()
    assert captured["model"].endswith("/default-oss")

    # Active preset -> overlay wins.
    llm_presets.set_active("openai-gpt-4o-mini")
    try:
        captured.clear()
        get_llm()
        assert captured["model"] == "openai/gpt-4o-mini"
        assert captured["api_key"] == "sk-real"
        # OpenAI proper has no base_url override -> param dropped.
        assert "base_url" not in captured
    finally:
        llm_presets.clear_active()
