"""Custom embedding function with input-length truncation.

Why this exists
===============
Small embedding models like ``embeddinggemma-300m`` cap inputs at 2048
tokens.  CrewAI's native short-term / long-term / entity memory dumps
**full** task outputs into the embedder — and a single Portfolio
Manager decision JSON (rationale + falsifiers + analyst summaries +
risk debate) can easily blow past 2048 tokens.  When that happens the
embedder returns a 400 right after the PM completes:

    Error code: 400 - {'error': {'message': "This model's maximum
    context length is 2048 tokens. However, your request has 2177
    input tokens. Please reduce the length of the input messages.",
    'type': 'BadRequestError', 'code': 400}}

This wrapper sits in front of chromadb's ``OpenAIEmbeddingFunction``
and clips every document to a character budget before forwarding.  The
budget defaults to 6000 chars (~1500 tokens, comfortably below the
2048-token cap) and is configurable via the ``max_chars`` kwarg or
``TRADINGCREW_EMBEDDER_MAX_CHARS`` env var.

Why truncation (not chunking)?  Memory writes are dense single-shot
embeddings used for *retrieval similarity*, not for round-tripping the
full text.  Truncating the head of a document keeps the semantic
signal (the lead paragraphs of an analyst report or the action +
confidence + rationale of a PM decision) — exactly what makes memory
retrieval useful.  Chunking + averaging would be more thorough but
also more expensive and harder to compose with CrewAI's storage
layer, which expects 1 embedding per stored item.

Why env-driven config (and not kwargs from ``get_embedder_config()``)?
CrewAI's ``CustomProviderConfig`` TypedDict only declares one field —
``embedding_callable`` — so when pydantic validates a Crew's
``embedder: EmbedderConfig`` field, every extra config key (api_key,
model_name, api_base, max_chars, …) is **stripped** during validation.
By the time CrewAI's factory finally calls ``TruncatingOpenAIEmbedder(**stripped_config)``,
it gets called with no kwargs.  The fix is to read the OpenAI-compatible
client settings from the same ``VLLM_EMBEDDING_*`` env vars that
``get_embedder_config()`` already requires, so the inner embedder
is configured even when constructed with zero kwargs.  We still
*accept* kwargs (for tests and direct usage) — env vars are the
fallback when kwargs are missing.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List

from chromadb.api.types import EmbeddingFunction as ChromaEmbeddingFunction
from crewai.rag.embeddings.providers.custom.embedding_callable import (
    CustomEmbeddingFunction,
)

logger = logging.getLogger(__name__)


# Default character budget. ~6000 chars ≈ 1500 tokens, leaves headroom
# under embeddinggemma's 2048-token cap for tokenizer variance.
DEFAULT_MAX_CHARS = 6000


def _normalize_api_base(base: str | None) -> str | None:
    """Match ``get_embedder_config()``: strip trailing slashes and ensure /v1."""
    if not base:
        return base
    base = base.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


# We inherit from BOTH CustomEmbeddingFunction (so CrewAI's
# ``CustomProvider.embedding_callable: type[CustomEmbeddingFunction]``
# field accepts us) AND chromadb's ``EmbeddingFunction`` Protocol (so
# the OUTER ``CustomProviderSpec.config.embedding_callable: type[chromadb.EmbeddingFunction]``
# pydantic validator also passes).  Both are ``runtime_checkable``
# Protocols, but ``issubclass()`` against a Protocol still requires
# *explicit* inheritance — structural conformance only works for
# ``isinstance()``.  Multi-inheriting both is the only way to satisfy
# both validators at once; see crewai-rag types.py vs custom_provider.py
# for the inconsistency between the two ``EmbeddingFunction`` symbols.
class TruncatingOpenAIEmbedder(CustomEmbeddingFunction, ChromaEmbeddingFunction):
    """OpenAI-compatible embedder that clips every input to ``max_chars``.

    Built on top of chromadb's ``OpenAIEmbeddingFunction`` so we keep
    OpenAI / vLLM compatibility and don't reinvent the auth + retry
    logic.  The truncation step runs *before* the inner embedder ever
    sees the text, so the inner client never gets a chance to raise
    the 2048-token error.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        api_base: str | None = None,
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Lazy import — chromadb's OpenAIEmbeddingFunction pulls in the
        # OpenAI SDK, which is fine but expensive at import-time.
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        # Resolution priority (highest to lowest):
        # 1. Direct kwargs (tests / explicit construction).
        # 2. **Active UI preset** via ``embedding_presets`` (thread-local,
        #    mirrors the LLM picker).  Closed-source presets ship their
        #    own model + leave ``api_base=None`` so the client defaults
        #    to ``https://api.openai.com/v1``; the open-source
        #    ``vllm-embedding`` preset is env-driven and pulls from
        #    ``VLLM_EMBEDDING_*`` itself.
        # 3. ``VLLM_EMBEDDING_*`` env vars (legacy / no-preset path).
        #
        # The key invariant: once a preset is active, every slot it
        # populates wins — including the ``api_base=None`` of OpenAI
        # proper.  We must NOT fall back into ``VLLM_EMBEDDING_BASE_URL``
        # for an active closed-source preset (that would route OpenAI
        # traffic at the vLLM endpoint and 4xx).
        from . import embedding_presets as _embedding_presets

        active_preset = _embedding_presets.get_active_preset()
        preset_overrides = (
            _embedding_presets.get_active_overrides() if active_preset else {}
        )

        if api_key is None:
            api_key = (
                preset_overrides["api_key"]
                if active_preset is not None
                else os.environ.get("VLLM_EMBEDDING_API_KEY")
            )
        if model_name is None:
            model_name = (
                preset_overrides["model_name"]
                if active_preset is not None
                else os.environ.get("VLLM_EMBEDDING_MODEL")
            )
        if api_base is None:
            api_base = (
                preset_overrides["api_base"]
                if active_preset is not None
                else os.environ.get("VLLM_EMBEDDING_BASE_URL")
            )
        api_base = _normalize_api_base(api_base)
        if max_chars is None:
            max_chars = (
                preset_overrides["max_chars"]
                if active_preset is not None
                else int(
                    os.environ.get(
                        "TRADINGCREW_EMBEDDER_MAX_CHARS", DEFAULT_MAX_CHARS
                    )
                )
            )

        if not api_key:
            raise ValueError(
                "TruncatingOpenAIEmbedder needs an API key — set "
                "VLLM_EMBEDDING_API_KEY in .env or pass api_key= explicitly."
            )

        # Filter out kwargs the inner OpenAIEmbeddingFunction doesn't
        # accept (CrewAI's CustomProvider has ``extra="allow"`` and
        # forwards every config key — we explicitly drop ours).
        passthrough = {
            k: v for k, v in kwargs.items()
            if k not in {"max_chars", "embedding_callable"}
        }
        self._inner = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name,
            api_base=api_base,
            **passthrough,
        )
        self._max_chars = int(max_chars)
        if self._max_chars <= 0:
            raise ValueError(
                f"max_chars must be positive, got {self._max_chars!r}"
            )
        self._model_name = model_name or "unknown"

    # ------------------------------------------------------------------
    # chromadb EmbeddingFunction interface
    # ------------------------------------------------------------------

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Embed ``input`` after clipping each document to ``max_chars``.

        ``input`` may legitimately contain non-string entries on some
        chromadb versions (it can pass numpy arrays, ints, etc.).  We
        only truncate ``str`` items; everything else flows through
        unchanged so the inner embedder can decide how to handle it.
        """
        truncated: List[str] = []
        any_truncated = False
        for doc in input:
            if isinstance(doc, str) and len(doc) > self._max_chars:
                truncated.append(doc[: self._max_chars])
                any_truncated = True
            else:
                truncated.append(doc)
        if any_truncated:
            # Debug, not warning — for a chatty memory subsystem this
            # would otherwise drown the logs.  Flip to warning if you
            # need to audit how often memory writes are getting clipped.
            logger.debug(
                "TruncatingOpenAIEmbedder clipped %d/%d input(s) to %d chars (model=%s)",
                sum(1 for d in input if isinstance(d, str) and len(d) > self._max_chars),
                len(input),
                self._max_chars,
                self._model_name,
            )
        return self._inner(truncated)
