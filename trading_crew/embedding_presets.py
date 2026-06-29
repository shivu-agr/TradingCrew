"""Embedding-model preset catalog + runtime override mechanism.

This is the *embedding* twin of ``llm_presets.py``.  The crew's
LLM-enriched memory (CrewAI's short-term / long-term / entity stores)
calls an OpenAI-compatible embeddings endpoint via
``TruncatingOpenAIEmbedder``; this module exposes a typed catalog so
the UI can let the user pick which embedder to use, plus a
thread-local override that the embedder consults before falling back
to the ``VLLM_EMBEDDING_*`` env chain.

Why separate from the LLM picker?
=================================
The chat LLM and the embedder are independent endpoints with
different operational constraints — context window vs. embedding
dimension, vector vs. token cost, etc.  A user might prefer the
hosted open-source ``embeddinggemma-300m`` for embeddings while
running their generation against ``openai-gpt-4o``, or vice versa.
Pinning them to separate presets keeps both axes orthogonal.

Per-preset shape
================
Each ``EmbeddingPreset`` carries the *static* parts of an
OpenAI-compatible embedding config (model, base URL template,
provider prefix, embedding dim, max input tokens, default
character budget for our truncation guard) plus the **env-var name**
that holds the API key (so the secret stays in ``.env`` and never
crosses the WS boundary).

A few presets are intentionally env-driven (``base_url`` / ``model``
left ``None`` to be filled by ``VLLM_EMBEDDING_*`` at runtime) so we
don't bake any deployment-specific URL into the codebase.
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class EmbeddingPreset:
    """One selectable embedding configuration the UI offers in its dropdown."""

    id: str
    label: str
    kind: str  # "open-source" | "closed-source"
    provider: str  # currently "openai-compatible" or "openai" (LiteLLM-style)
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_env: str = "VLLM_EMBEDDING_API_KEY"
    # When True, ``model`` / ``base_url`` left ``None`` here will be
    # filled at runtime from the ``VLLM_EMBEDDING_MODEL`` /
    # ``VLLM_EMBEDDING_BASE_URL`` env vars.  Closed-source presets
    # (OpenAI proper) must NOT fall back to those — they ship the
    # model name statically and let the OpenAI client default the URL
    # to ``https://api.openai.com/v1``.
    env_driven: bool = False
    description: str = ""
    # Embedding dim is informational — the LanceDB schema is locked in
    # the first time the store is created with a given embedder, so
    # switching dims requires a fresh root_scope (the crew folds the
    # preset id into ``root_scope`` for exactly this reason).
    dim: Optional[int] = None
    # Max input tokens the upstream model accepts.  Used to pick a
    # safe ``max_chars`` truncation cap for our wrapper.
    max_input_tokens: Optional[int] = None
    # Default character budget for ``TruncatingOpenAIEmbedder``.  We
    # ship ~25 % headroom below ``max_input_tokens * 4`` so tokenizer
    # variance doesn't push us past the upstream cap.
    default_max_chars: int = 6000
    tags: List[str] = field(default_factory=list)

    def resolve_model(self) -> Optional[str]:
        if self.model is not None:
            return self.model
        return os.environ.get("VLLM_EMBEDDING_MODEL") if self.env_driven else None

    def resolve_base_url(self) -> Optional[str]:
        if self.base_url is not None:
            return self.base_url
        return os.environ.get("VLLM_EMBEDDING_BASE_URL") if self.env_driven else None

    def resolve_api_key(self) -> Optional[str]:
        return os.environ.get(self.api_env) if self.api_env else None

    def to_overrides(self) -> Dict[str, Optional[str | int]]:
        """Resolve into the kwargs ``TruncatingOpenAIEmbedder`` actually uses.

        ``None`` is a legitimate value here — the embedder keeps any
        field whose override is ``None`` and falls back to its env-var
        chain for that slot only.
        """
        return {
            "model_name": self.resolve_model(),
            "api_base": self.resolve_base_url(),
            "api_key": self.resolve_api_key(),
            "max_chars": int(
                os.environ.get(
                    "TRADINGCREW_EMBEDDER_MAX_CHARS", self.default_max_chars
                )
            ),
        }


# ---------------------------------------------------------------------------
# Built-in presets.  ``id`` must be URL-safe (it travels in the WS payload
# and gets folded into the CrewAI memory ``root_scope``).
# ---------------------------------------------------------------------------

BUILTIN_PRESETS: Dict[str, EmbeddingPreset] = {
    # Open-source / hosted vLLM — entirely env-driven so we don't bake
    # any deployment-specific URL into the codebase.  The label nudges
    # the user toward the .env that powers it.
    "vllm-embedding": EmbeddingPreset(
        id="vllm-embedding",
        label="Hosted vLLM embedding (VLLM_EMBEDDING_*)",
        kind="open-source",
        provider="openai-compatible",
        model=None,
        base_url=None,
        env_driven=True,
        api_env="VLLM_EMBEDDING_API_KEY",
        description=(
            "Any OpenAI-compatible embedding server you point "
            "VLLM_EMBEDDING_BASE_URL / VLLM_EMBEDDING_MODEL / "
            "VLLM_EMBEDDING_API_KEY at in .env.  Default deployments "
            "use embeddinggemma-300m (768-d, 2048-token input cap); "
            "the truncation guard keeps writes under that limit."
        ),
        dim=None,  # depends on what the user points it at
        max_input_tokens=2048,
        default_max_chars=6000,
        tags=["vLLM", "open-source"],
    ),
    "openai-text-embedding-3-small": EmbeddingPreset(
        id="openai-text-embedding-3-small",
        label="OpenAI · text-embedding-3-small (closed)",
        kind="closed-source",
        provider="openai",
        model="text-embedding-3-small",
        base_url=None,  # the OpenAI client defaults to https://api.openai.com/v1
        api_env="OPENAI_PROD_API_KEY",
        description=(
            "OpenAI's text-embedding-3-small (1536-d, 8191-token input "
            "cap).  Closed-source, billed per token.  Set "
            "OPENAI_PROD_API_KEY in .env to enable."
        ),
        dim=1536,
        max_input_tokens=8191,
        # 8191 tokens ≈ 30K chars; we cap below that with headroom.
        default_max_chars=24000,
        tags=["closed-source", "OpenAI"],
    ),
    "openai-text-embedding-3-large": EmbeddingPreset(
        id="openai-text-embedding-3-large",
        label="OpenAI · text-embedding-3-large (closed)",
        kind="closed-source",
        provider="openai",
        model="text-embedding-3-large",
        base_url=None,
        api_env="OPENAI_PROD_API_KEY",
        description=(
            "OpenAI's text-embedding-3-large (3072-d, 8191-token input "
            "cap).  Higher dimensional and more expensive than -small; "
            "recommended only when retrieval quality on long memories "
            "matters more than embedding cost."
        ),
        dim=3072,
        max_input_tokens=8191,
        default_max_chars=24000,
        tags=["closed-source", "OpenAI"],
    ),
}


def list_presets() -> List[Dict[str, object]]:
    """Serialisable preset list for ``/api/options``.

    Includes a per-preset ``api_key_configured`` flag (mirrors the LLM
    picker) so the UI greys out closed-source options whose key isn't
    set.  The hosted vLLM preset also requires the model + base URL to
    be populated; the flag captures that too.
    """
    out: List[Dict[str, object]] = []
    for preset in BUILTIN_PRESETS.values():
        api_key_val = preset.resolve_api_key()
        configured = bool(api_key_val) and api_key_val.lower() != "dummy"
        # The hosted vLLM preset is fully env-driven — also require the
        # model + base URL to be set, otherwise selecting it would fail
        # at embedder construction with a confusing 4xx.
        if preset.id == "vllm-embedding":
            configured = configured and bool(preset.resolve_model()) and bool(
                preset.resolve_base_url()
            )
        d = asdict(preset)
        d["api_key_configured"] = configured
        d["resolved_model"] = preset.resolve_model()
        d["resolved_base_url"] = preset.resolve_base_url()
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Thread-local active override.  The embedder consults this before its
# env-var chain so a WS session can swap embedders without touching
# ``os.environ``.
# ---------------------------------------------------------------------------

_active = threading.local()


def set_active(preset_id: Optional[str]) -> Optional[EmbeddingPreset]:
    """Mark ``preset_id`` as active for the current thread.

    Pass ``None`` (or an unknown id) to clear the override — the
    env-var chain then takes over again.  Returns the resolved
    ``EmbeddingPreset`` for logging convenience.
    """
    if not preset_id:
        _active.preset = None
        return None
    preset = BUILTIN_PRESETS.get(preset_id)
    _active.preset = preset
    return preset


def clear_active() -> None:
    _active.preset = None


def get_active_preset() -> Optional[EmbeddingPreset]:
    return getattr(_active, "preset", None)


def get_active_overrides() -> Dict[str, Optional[str | int]]:
    """Return the kwargs ``TruncatingOpenAIEmbedder`` should overlay on
    top of its env-var defaults.  Empty dict when no preset is active
    on this thread (legacy env-only path)."""
    preset = get_active_preset()
    if preset is None:
        return {}
    return preset.to_overrides()


def get_active_preset_id() -> Optional[str]:
    preset = get_active_preset()
    return preset.id if preset is not None else None
