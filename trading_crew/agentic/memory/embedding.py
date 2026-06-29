"""Embedding-function factory for the episodic memory.

The default ``EpisodicMemory.retrieve`` path uses a deterministic TF-IDF
bag-of-words vector — fast, model-free, perfect for unit tests and CI.
Production deployments want richer semantic similarity though, which is
where this module comes in.

vLLM-backed embedder
====================
We point at the same OpenAI-compatible embedding endpoint that CrewAI
uses for its native memory (``get_embedder_config()`` in
``trading_crew/_common.py``).  The endpoint is configured via
``VLLM_EMBEDDING_BASE_URL`` / ``VLLM_EMBEDDING_MODEL`` /
``VLLM_EMBEDDING_API_KEY``.

The wrapper returns a ``Dict[str, float]`` so the existing
``EpisodicMemory._cosine`` consumer doesn't change.  We synthesise the
keys as ``f"dim{i:04d}"`` — opaque, stable, sparse-friendly.

Selecting an embedder
=====================
The default is picked via ``TRADINGCREW_MEMORY_EMBEDDER``:

  * ``tfidf`` (default — works offline; what CI uses).
  * ``vllm``  — calls the OpenAI-compatible ``/embeddings`` endpoint.

Use ``get_default_embed_fn()`` from callers that want the env-selected
default; or pass ``embed_fn=`` explicitly into ``EpisodicMemory``.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _vllm_embed_fn() -> Callable[[str], Dict[str, float]]:
    """Build an embedding callable backed by the vLLM endpoint.

    Imports the OpenAI client lazily so callers who never request the
    vLLM embedder don't pay the import cost.  Raises if the env is
    misconfigured — there is no implicit fallback to TF-IDF (the caller
    asked for vLLM; silently switching would be the wrong behaviour).
    """
    base = os.environ.get("VLLM_EMBEDDING_BASE_URL")
    model = os.environ.get("VLLM_EMBEDDING_MODEL")
    api_key = os.environ.get("VLLM_EMBEDDING_API_KEY")
    if not (base and model and api_key):
        raise RuntimeError(
            "vLLM memory embedder requested but VLLM_EMBEDDING_BASE_URL / "
            "VLLM_EMBEDDING_MODEL / VLLM_EMBEDDING_API_KEY are not set."
        )
    base = base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "vLLM memory embedder needs the `openai` package. "
            "Install it via `pip install openai`."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=base)

    def _embed(text: str) -> Dict[str, float]:
        # Empty input → no embedding; mimic _default_embed semantics.
        if not text or not text.strip():
            return {}
        resp = client.embeddings.create(model=model, input=text)
        vec = resp.data[0].embedding
        return {f"dim{i:04d}": float(v) for i, v in enumerate(vec)}

    return _embed


def get_default_embed_fn() -> Optional[Callable[[str], Dict[str, float]]]:
    """Resolve the embedder selected by ``TRADINGCREW_MEMORY_EMBEDDER``.

    Returns ``None`` when the env selects ``tfidf`` (or is unset) — the
    caller then leaves ``embed_fn`` unset and ``EpisodicMemory`` uses
    its built-in deterministic TF-IDF.
    """
    choice = (os.environ.get("TRADINGCREW_MEMORY_EMBEDDER") or "tfidf").lower()
    if choice == "tfidf":
        return None
    if choice == "vllm":
        try:
            return _vllm_embed_fn()
        except RuntimeError as exc:
            # Non-fatal — surfaces a single warning, the caller falls
            # back to TF-IDF.  This keeps offline test runs working even
            # when someone exports TRADINGCREW_MEMORY_EMBEDDER=vllm.
            logger.warning("vLLM embedder unavailable, using TF-IDF: %s", exc)
            return None
    logger.warning(
        "Unknown TRADINGCREW_MEMORY_EMBEDDER=%r — falling back to TF-IDF", choice
    )
    return None


def make_memory(path, **overrides):
    """Helper: return an ``EpisodicMemory`` wired with the env-selected embedder.

    Production callers (the runner, the agent tools, the FastAPI endpoint)
    use this so they all share the same embedder selection without
    re-implementing the env check in five places.  Tests construct
    ``EpisodicMemory`` directly with ``embed_fn=None`` to keep CI offline
    and deterministic.
    """
    # Imported lazily so test-only modules that import this file don't
    # pay the cost of the openai client import.
    from trading_crew.agentic.memory.episodic import EpisodicMemory

    kwargs = dict(overrides)
    kwargs.setdefault("embed_fn", get_default_embed_fn())
    return EpisodicMemory(path, **kwargs)
