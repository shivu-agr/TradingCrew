"""Tests for ``trading_crew.embedding_presets``.

Mirrors ``tests/test_llm_presets.py`` in spirit: the UI exposes a typed
catalog + thread-local override, and the embedder must consult both at
construction time.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture(autouse=True)
def _isolate_active(monkeypatch):
    """Reset the thread-local active preset between tests so leaking
    state from one test can't poison another's resolution chain."""
    from trading_crew import embedding_presets

    embedding_presets.clear_active()
    yield
    embedding_presets.clear_active()


# ---------------------------------------------------------------------------
# Catalog / API surface
# ---------------------------------------------------------------------------

def test_builtin_presets_have_required_metadata():
    """Every preset must expose the fields the UI dropdown renders."""
    from trading_crew.embedding_presets import BUILTIN_PRESETS

    assert BUILTIN_PRESETS, "expected at least one built-in embedding preset"
    for preset in BUILTIN_PRESETS.values():
        assert preset.id and preset.label, preset
        assert preset.kind in {"open-source", "closed-source"}, preset
        # Either a static URL OR an env-driven one — but the provider
        # name must be set so the embedder knows how to route.
        assert preset.provider, preset
        assert preset.api_env, preset


def test_list_presets_marks_unconfigured_keys(monkeypatch):
    """Closed-source presets without an API key in the env must report
    ``api_key_configured=False`` so the UI can grey them out."""
    from trading_crew.embedding_presets import list_presets

    monkeypatch.delenv("OPENAI_PROD_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_EMBEDDING_MODEL", raising=False)

    presets = {p["id"]: p for p in list_presets()}

    assert presets["openai-text-embedding-3-small"]["api_key_configured"] is False
    assert presets["openai-text-embedding-3-large"]["api_key_configured"] is False
    # vllm preset needs ALL of api_key + base_url + model — none set, so False.
    assert presets["vllm-embedding"]["api_key_configured"] is False


def test_list_presets_treats_dummy_api_key_as_unconfigured(monkeypatch):
    """The local vLLM client requires a non-empty bearer; users sometimes
    paste ``dummy`` to satisfy that.  Closed-source presets must NOT
    accept ``dummy`` as a real key (it would only ever produce a 401)."""
    from trading_crew.embedding_presets import list_presets

    monkeypatch.setenv("OPENAI_PROD_API_KEY", "dummy")
    presets = {p["id"]: p for p in list_presets()}
    assert presets["openai-text-embedding-3-small"]["api_key_configured"] is False


def test_list_presets_marks_vllm_configured_only_when_all_set(monkeypatch):
    from trading_crew.embedding_presets import list_presets

    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "real-key")
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://example.com")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")

    presets = {p["id"]: p for p in list_presets()}
    assert presets["vllm-embedding"]["api_key_configured"] is True
    assert presets["vllm-embedding"]["resolved_model"] == "embeddinggemma-300m"
    assert presets["vllm-embedding"]["resolved_base_url"] == "https://example.com"


# ---------------------------------------------------------------------------
# Thread-local override
# ---------------------------------------------------------------------------

def test_set_active_and_clear(monkeypatch):
    from trading_crew import embedding_presets as ep

    monkeypatch.setenv("OPENAI_PROD_API_KEY", "sk-real")

    applied = ep.set_active("openai-text-embedding-3-small")
    assert applied is not None
    assert applied.id == "openai-text-embedding-3-small"
    assert ep.get_active_preset_id() == "openai-text-embedding-3-small"

    overrides = ep.get_active_overrides()
    assert overrides["model_name"] == "text-embedding-3-small"
    assert overrides["api_key"] == "sk-real"
    # The OpenAI proper preset intentionally has base_url=None so chromadb's
    # client uses https://api.openai.com/v1 — exposed as api_base=None here.
    assert overrides["api_base"] is None

    ep.clear_active()
    assert ep.get_active_preset_id() is None
    assert ep.get_active_overrides() == {}


def test_set_active_unknown_id_clears(monkeypatch):
    from trading_crew import embedding_presets as ep

    monkeypatch.setenv("OPENAI_PROD_API_KEY", "sk-real")
    ep.set_active("openai-text-embedding-3-small")
    ep.set_active("does-not-exist")
    # Unknown id wipes the override (preset = None).
    assert ep.get_active_preset_id() is None


# ---------------------------------------------------------------------------
# Integration: TruncatingOpenAIEmbedder consults the active preset
# ---------------------------------------------------------------------------

class _StubOpenAIEmbedding:
    """Stand-in for ``chromadb.utils.embedding_functions.OpenAIEmbeddingFunction``.

    Captures the kwargs the wrapper passes in so the test can assert that
    the active preset's values actually reached the underlying client.
    """

    last_kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)

    def __call__(self, docs):
        return [[0.0] * 8 for _ in docs]


def test_embedder_uses_active_preset_over_env(monkeypatch):
    """When a preset is active, its model / api_key / api_base must win
    over the ``VLLM_EMBEDDING_*`` env chain — including the all-important
    ``api_base=None`` of closed-source OpenAI presets (which would
    otherwise inherit the local vLLM URL and 4xx)."""
    from trading_crew import embedding_presets as ep

    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "vllm-key")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://vllm.example.com")
    monkeypatch.setenv("OPENAI_PROD_API_KEY", "sk-real")
    # Clear the explicit clamp so the preset's model-aware default
    # (24000 for OpenAI-3-small) shows through.
    monkeypatch.delenv("TRADINGCREW_EMBEDDER_MAX_CHARS", raising=False)

    monkeypatch.setattr(
        "chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction",
        _StubOpenAIEmbedding,
    )

    ep.set_active("openai-text-embedding-3-small")

    from trading_crew.embedder import TruncatingOpenAIEmbedder

    embedder = TruncatingOpenAIEmbedder()

    captured = _StubOpenAIEmbedding.last_kwargs
    assert captured["api_key"] == "sk-real"
    assert captured["model_name"] == "text-embedding-3-small"
    assert captured["api_base"] is None
    assert embedder._max_chars == 24000


def test_embedder_explicit_max_chars_env_overrides_preset(monkeypatch):
    """``TRADINGCREW_EMBEDDER_MAX_CHARS`` is the operator escape hatch
    for clamping the truncation budget below a preset's sensible
    default (e.g. when running an OpenAI embedder against a small
    proxy that enforces a tighter cap).  It must win over the preset's
    ``default_max_chars``."""
    from trading_crew import embedding_presets as ep

    monkeypatch.setenv("OPENAI_PROD_API_KEY", "sk-real")
    monkeypatch.setenv("TRADINGCREW_EMBEDDER_MAX_CHARS", "5000")
    monkeypatch.setattr(
        "chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction",
        _StubOpenAIEmbedding,
    )

    ep.set_active("openai-text-embedding-3-small")

    from trading_crew.embedder import TruncatingOpenAIEmbedder

    embedder = TruncatingOpenAIEmbedder()
    assert embedder._max_chars == 5000


def test_embedder_falls_back_to_env_when_no_active_preset(monkeypatch):
    """Backward-compat: with no preset active, the embedder still reads
    purely from ``VLLM_EMBEDDING_*`` env vars."""
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "vllm-key")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://vllm.example.com")
    monkeypatch.delenv("OPENAI_PROD_API_KEY", raising=False)

    monkeypatch.setattr(
        "chromadb.utils.embedding_functions.openai_embedding_function.OpenAIEmbeddingFunction",
        _StubOpenAIEmbedding,
    )

    from trading_crew.embedder import TruncatingOpenAIEmbedder

    TruncatingOpenAIEmbedder()

    captured = _StubOpenAIEmbedding.last_kwargs
    assert captured["api_key"] == "vllm-key"
    assert captured["model_name"] == "embeddinggemma-300m"
    # The env URL gets normalised to end with /v1.
    assert captured["api_base"] == "https://vllm.example.com/v1"
