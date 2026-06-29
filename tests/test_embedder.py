"""Tests for the TruncatingOpenAIEmbedder wrapper.

We mock chromadb's ``OpenAIEmbeddingFunction`` to capture exactly what
gets forwarded, so the test can assert truncation happens *before* the
inner embedder ever sees the input.
"""

from __future__ import annotations

from typing import List
from unittest import mock

import pytest


class _FakeInner:
    """Stand-in for chromadb's ``OpenAIEmbeddingFunction``.

    Records every call so the test can assert on what was forwarded.
    Returns deterministic length-3 vectors so the wrapper can be
    smoke-checked for shape preservation.
    """

    instances: list["_FakeInner"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[list[str]] = []
        _FakeInner.instances.append(self)

    def __call__(self, input: List[str]):  # noqa: A002 (mirrors chromadb)
        self.calls.append(list(input))
        return [[float(len(s)) if isinstance(s, str) else 0.0,
                 0.0, 0.0] for s in input]


@pytest.fixture(autouse=True)
def _patch_chromadb(monkeypatch):
    """Patch chromadb's OpenAIEmbeddingFunction with our fake so the
    test never opens a network connection.

    We patch the import-site name (the symbol our embedder imports
    inside its __init__) by patching the chromadb module attribute.
    """
    import chromadb.utils.embedding_functions.openai_embedding_function as mod
    _FakeInner.instances.clear()
    monkeypatch.setattr(mod, "OpenAIEmbeddingFunction", _FakeInner)
    yield


def test_truncates_long_inputs_before_forwarding():
    from trading_crew.embedder import TruncatingOpenAIEmbedder

    emb = TruncatingOpenAIEmbedder(
        api_key="dummy",
        model_name="embeddinggemma-300m",
        api_base="https://example/v1",
        max_chars=100,
    )
    short = "a" * 50
    long_ = "b" * 5000
    out = emb([short, long_])
    assert len(out) == 2
    inner = _FakeInner.instances[-1]
    # First was short -> passed through unchanged.
    assert inner.calls[0][0] == short
    # Second was long -> truncated to exactly max_chars=100.
    assert inner.calls[0][1] == "b" * 100


def test_short_inputs_are_not_modified():
    from trading_crew.embedder import TruncatingOpenAIEmbedder

    emb = TruncatingOpenAIEmbedder(
        api_key="dummy",
        model_name="embeddinggemma-300m",
        api_base="https://example/v1",
        max_chars=10_000,
    )
    docs = ["alpha", "beta", "gamma"]
    emb(docs)
    inner = _FakeInner.instances[-1]
    assert inner.calls[0] == docs


def test_max_chars_env_var_default(monkeypatch):
    monkeypatch.setenv("TRADINGCREW_EMBEDDER_MAX_CHARS", "42")
    from trading_crew.embedder import TruncatingOpenAIEmbedder

    emb = TruncatingOpenAIEmbedder(
        api_key="dummy",
        model_name="embeddinggemma-300m",
        api_base="https://example/v1",
    )
    long_ = "z" * 1000
    emb([long_])
    inner = _FakeInner.instances[-1]
    assert inner.calls[0][0] == "z" * 42


def test_rejects_zero_max_chars(monkeypatch):
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "abc")
    from trading_crew.embedder import TruncatingOpenAIEmbedder

    with pytest.raises(ValueError):
        TruncatingOpenAIEmbedder(
            api_key="dummy",
            model_name="x",
            api_base="https://x/v1",
            max_chars=0,
        )


def test_get_embedder_config_picks_custom_provider(monkeypatch):
    """The CrewAI embedder config returned by get_embedder_config() must
    route through our truncating callable so memory writes are clipped
    before the embedder ever sees them.

    Note: extra config keys (api_key / model / api_base / max_chars) are
    intentionally NOT passed through this dict — CrewAI's ``CustomProviderConfig``
    TypedDict only declares ``embedding_callable``, so pydantic strips
    every other key during ``EmbedderConfig`` validation.  Our embedder
    reads its config from the ``VLLM_EMBEDDING_*`` env vars at
    construction time instead.
    """
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://example.com")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "abc")
    monkeypatch.setenv("TRADINGCREW_EMBEDDER_MAX_CHARS", "1234")

    from trading_crew._common import get_embedder_config
    from trading_crew.embedder import TruncatingOpenAIEmbedder

    cfg = get_embedder_config()
    assert cfg["provider"] == "custom"
    assert cfg["config"]["embedding_callable"] is TruncatingOpenAIEmbedder
    # Only the class — extra config would get stripped by pydantic anyway.
    assert set(cfg["config"].keys()) == {"embedding_callable"}


def test_constructs_from_env_vars_when_no_kwargs(monkeypatch):
    """Reproduce the production code path: CrewAI's pydantic strips the
    custom-provider config dict down to just ``embedding_callable``,
    then calls our class with **no kwargs**.  The class must still
    construct a working embedder by reading ``VLLM_EMBEDDING_*`` from
    the environment.  Before this fix, a no-kwargs construction crashed
    with `The CHROMA_OPENAI_API_KEY environment variable is not set`.
    """
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://example.com")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "env-key-xyz")
    monkeypatch.setenv("TRADINGCREW_EMBEDDER_MAX_CHARS", "777")
    from trading_crew.embedder import TruncatingOpenAIEmbedder

    emb = TruncatingOpenAIEmbedder()
    assert emb._max_chars == 777
    # Inner OpenAIEmbeddingFunction got its api_key / model / base from env.
    inner = _FakeInner.instances[-1]
    assert inner.kwargs["api_key"] == "env-key-xyz"
    assert inner.kwargs["model_name"] == "embeddinggemma-300m"
    assert inner.kwargs["api_base"].endswith("/v1")


def test_full_build_embedder_pipeline_no_kwargs(monkeypatch):
    """End-to-end through CrewAI's factory: validating the embedder dict
    through ``EmbedderConfig`` then calling ``build_embedder`` (the same
    path Crew's ``create_crew_memory`` model_validator takes) must
    yield a working ``TruncatingOpenAIEmbedder``."""
    monkeypatch.setenv("VLLM_EMBEDDING_BASE_URL", "https://example.com")
    monkeypatch.setenv("VLLM_EMBEDDING_MODEL", "embeddinggemma-300m")
    monkeypatch.setenv("VLLM_EMBEDDING_API_KEY", "abc")

    from pydantic import TypeAdapter
    from crewai.rag.embeddings.types import EmbedderConfig
    from crewai.rag.embeddings.factory import build_embedder
    from trading_crew._common import get_embedder_config
    from trading_crew.embedder import TruncatingOpenAIEmbedder

    spec = get_embedder_config()
    # Step 1: pydantic validation (this is what was stripping fields).
    validated = TypeAdapter(EmbedderConfig).validate_python(spec)
    assert set(validated["config"].keys()) == {"embedding_callable"}
    # Step 2: build_embedder (the model_validator step in Crew).
    embedder = build_embedder(dict(validated))
    assert isinstance(embedder, TruncatingOpenAIEmbedder)


def test_non_string_inputs_pass_through_untouched():
    """chromadb sometimes passes through non-string items (numpy arrays etc.).
    Make sure we don't blow up trying to call ``len`` on them — only ``str``
    instances get the truncation treatment."""
    from trading_crew.embedder import TruncatingOpenAIEmbedder

    emb = TruncatingOpenAIEmbedder(
        api_key="dummy",
        model_name="x",
        api_base="https://x/v1",
        max_chars=10,
    )
    # Mixed: a too-long string, a number, a too-long string again.
    out = emb(["a" * 50, 12345, "b" * 50])
    assert len(out) == 3
    inner = _FakeInner.instances[-1]
    assert inner.calls[0] == ["a" * 10, 12345, "b" * 10]


def test_passes_crewai_pydantic_validation():
    """Regression test for the 23-error pydantic validation explosion we
    hit when ``TruncatingOpenAIEmbedder`` only inherited from CrewAI's
    ``CustomEmbeddingFunction``.

    The Crew model declares ``embedder: EmbedderConfig`` where
    ``EmbedderConfig`` is a Union of every ProviderSpec including
    ``CustomProviderSpec`` whose ``config.embedding_callable`` is typed
    as ``type[chromadb.api.types.EmbeddingFunction]``.  Pydantic's
    ``is_subclass_of`` validator requires *explicit* inheritance from
    that protocol — structural conformance only satisfies isinstance(),
    not issubclass().  So our class has to multi-inherit from both
    chromadb's EmbeddingFunction AND CrewAI's CustomEmbeddingFunction.
    """
    from chromadb.api.types import EmbeddingFunction as ChromaEF
    from crewai.rag.embeddings.providers.custom.embedding_callable import (
        CustomEmbeddingFunction,
    )
    from trading_crew.embedder import TruncatingOpenAIEmbedder

    # Both must succeed — if either is False, Crew(embedder=...)
    # validation explodes at runtime.
    assert issubclass(TruncatingOpenAIEmbedder, ChromaEF), (
        "Must extend chromadb's EmbeddingFunction so CustomProviderSpec "
        "pydantic validation accepts it"
    )
    assert issubclass(TruncatingOpenAIEmbedder, CustomEmbeddingFunction), (
        "Must extend CrewAI's CustomEmbeddingFunction so the inner "
        "CustomProvider constructor accepts it"
    )

    # Validate the embedder dict the same way Crew does — through the
    # ``EmbedderConfig`` discriminated union.  This is the exact call
    # path that blew up in the real UI with 23 errors; we don't need a
    # full Crew construction to assert the regression is fixed.
    from pydantic import TypeAdapter
    from crewai.rag.embeddings.types import EmbedderConfig

    spec = {
        "provider": "custom",
        "config": {
            "embedding_callable": TruncatingOpenAIEmbedder,
            "api_key": "dummy",
            "model_name": "embeddinggemma-300m",
            "api_base": "https://example/v1",
            "max_chars": 4096,
        },
    }
    validated = TypeAdapter(EmbedderConfig).validate_python(spec)
    assert validated["provider"] == "custom"
    assert validated["config"]["embedding_callable"] is TruncatingOpenAIEmbedder
