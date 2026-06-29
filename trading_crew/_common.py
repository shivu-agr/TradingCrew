"""Local LLM + Memory factory.

Mirrors the workspace's ``_common.py`` so the project depends on nothing
outside its own folder. Both factories build OpenAI-compatible objects
that route through a local vLLM (or any OpenAI-compatible) server.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from . import _patches as _patches

# Load .env from the project root (two levels up from this file)
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

# Apply CrewAI runtime patches once (TaskOutput.raw str-coercion etc.).
_patches.install()


def _load_per_agent_overrides() -> dict:
    """Phase 2E — per-agent LLM overrides via the ``LLM_PER_AGENT`` env var.

    The variable is a JSON map ``{agent_key: {field: value, ...}}`` where
    ``agent_key`` matches the role identifier used in ``agents.yaml``
    (e.g. ``"market_analyst"``, ``"bull_researcher"``, ``"portfolio_manager"``).

    Each value dict can override any of:

    * ``model``      — model name (without provider prefix).
    * ``base_url``   — OpenAI-compatible endpoint URL.
    * ``api_key``    — bearer token.
    * ``provider``   — LiteLLM provider prefix (default ``hosted_vllm``).
    * ``temperature`` — sampling temperature.

    Invalid JSON falls back silently to the global config (we never want
    a typo to brick the whole crew).
    """
    raw = os.environ.get("LLM_PER_AGENT")
    if not raw:
        return {}
    try:
        import json
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def get_llm(temperature: float = 0.3, agent_key: str | None = None, **overrides):
    """Build a ``crewai.LLM`` pointed at an OpenAI-compatible server.

    Resolution order (lowest to highest priority)
    ---------------------------------------------
    1. ``LOCAL_LLM_*`` env vars (in-workspace fallback).
    2. ``VLLM_LLM_*`` env vars (preferred default — any OpenAI-compatible
       hosted vLLM-style endpoint).
    3. **Per-agent JSON override** via ``LLM_PER_AGENT[agent_key]``
       (Phase 2E — lets cheap agents use a smaller model and long-context
       agents keep the 131K hosted endpoint).
    4. **Active UI preset** via ``llm_presets.get_active_overrides()``
       (Phase 2F — what the user picked from the LLM dropdown in the
       sidebar; thread-local so concurrent WS sessions don't collide).
    5. Direct kwargs passed to this function.
    """
    from crewai import LLM

    # Local import to avoid a circular dep — ``llm_presets`` is a sibling.
    from . import llm_presets

    base_url = os.environ.get("VLLM_LLM_BASE_URL") or os.environ.get("LOCAL_LLM_BASE_URL")
    model = os.environ.get("VLLM_LLM_MODEL") or os.environ.get("LOCAL_LLM_MODEL")
    api_key = (
        os.environ.get("VLLM_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or "dummy"
    )
    provider = os.environ.get("LLM_PROVIDER_PREFIX", "hosted_vllm")

    if agent_key:
        per_agent = _load_per_agent_overrides().get(agent_key) or {}
        if isinstance(per_agent, dict):
            if "model" in per_agent:
                model = per_agent["model"]
            if "base_url" in per_agent:
                base_url = per_agent["base_url"]
            if "api_key" in per_agent:
                api_key = per_agent["api_key"]
            if "provider" in per_agent:
                provider = per_agent["provider"]
            if "temperature" in per_agent:
                temperature = float(per_agent["temperature"])

    # Phase 2F — UI preset overlay.  The runner sets this just before
    # ``crew.kickoff()`` based on what the user picked in the sidebar.
    preset_overrides = llm_presets.get_active_overrides()
    if preset_overrides:
        if "model" in preset_overrides:
            model = preset_overrides["model"]
        if "base_url" in preset_overrides:
            base_url = preset_overrides["base_url"]
        if "api_key" in preset_overrides and preset_overrides["api_key"]:
            api_key = preset_overrides["api_key"]
        if "provider" in preset_overrides:
            provider = preset_overrides["provider"]

    if not model:
        raise RuntimeError(
            "No LLM model resolved — set VLLM_LLM_MODEL / LOCAL_LLM_MODEL "
            "in .env, or pick a preset from the UI."
        )

    # LiteLLM expects ``<provider>/<model>`` except for ``openai`` /
    # ``anthropic`` where the prefix is implicit when an API key + the
    # right base_url is set.  We keep the explicit prefix for clarity.
    if provider in ("openai", "anthropic") and base_url is None:
        full_model = f"{provider}/{model}"
    else:
        full_model = f"{provider}/{model}"

    params = dict(
        model=full_model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
    # Direct kwargs win last.
    params.update(overrides)
    # base_url=None confuses some LiteLLM providers — drop it for the
    # closed-source path where the SDK handles routing itself.
    if params.get("base_url") is None:
        params.pop("base_url", None)
    return LLM(**params)


def get_embedder_config() -> dict:
    """Build a CrewAI ``EmbedderConfig`` pointed at an OpenAI-compatible server.

    Used by both crews together with ``Crew(memory=True, embedder=...)`` —
    CrewAI's native memory wiring then builds the unified short-term /
    long-term / entity memories on top of this embedder.

    Required configuration:

    * ``VLLM_EMBEDDING_BASE_URL`` -- the OpenAI-compatible embedding endpoint.
    * ``VLLM_EMBEDDING_MODEL``    -- e.g. ``embeddinggemma-300m``.
    * ``VLLM_EMBEDDING_API_KEY``  -- bearer token for the endpoint.

    These can come from either the .env file *or* the active embedding
    preset (set by the UI via ``embedding_presets.set_active()``).
    Closed-source presets that target OpenAI proper (no base URL,
    different API key env var) bypass the URL check entirely.

    The base URL is normalised so callers can supply it with or without a
    trailing ``/v1`` (CrewAI / OpenAI client expects the ``/v1`` root).

    **Truncation guard.**  Small embedders like ``embeddinggemma-300m``
    cap inputs at 2048 tokens.  CrewAI's native memory dumps full task
    outputs (PM rationale, debate synthesis, analyst summaries) into the
    embedder, which can exceed that limit and surface as a confusing
    ``400 BadRequestError: maximum context length is 2048 tokens`` after
    the Portfolio Manager finishes.  We wrap the OpenAI embedding
    function with ``TruncatingOpenAIEmbedder`` so every input is clipped
    to ``TRADINGCREW_EMBEDDER_MAX_CHARS`` (or the active preset's
    ``default_max_chars``) before reaching the upstream model.
    """
    # Resolve via the active preset first (so a UI-picked closed-source
    # embedder doesn't trip the VLLM_EMBEDDING_* env check below), then
    # fall through to the env chain.
    from . import embedding_presets

    preset = embedding_presets.get_active_preset()
    if preset is not None:
        base = preset.resolve_base_url()
        model = preset.resolve_model()
        api_key = preset.resolve_api_key()
        # OpenAI proper (no base URL) is allowed — the chromadb client
        # falls back to https://api.openai.com/v1 when api_base is None.
        require_base = preset.provider != "openai"
    else:
        base = os.environ.get("VLLM_EMBEDDING_BASE_URL")
        model = os.environ.get("VLLM_EMBEDDING_MODEL")
        api_key = os.environ.get("VLLM_EMBEDDING_API_KEY")
        require_base = True

    missing_map = {"VLLM_EMBEDDING_MODEL": model, "VLLM_EMBEDDING_API_KEY": api_key}
    if require_base:
        missing_map["VLLM_EMBEDDING_BASE_URL"] = base
    missing = [k for k, v in missing_map.items() if not v]
    if missing:
        raise RuntimeError(
            "Memory is enabled but the embedding endpoint is not configured. "
            f"Set the following env vars in .env (or pick a configured "
            f"embedding preset in the UI): {', '.join(missing)}"
        )

    if base:
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"

    # Local import keeps the heavy chromadb / crewai.rag imports out of
    # the hot path for callers that never touch memory.
    from .embedder import TruncatingOpenAIEmbedder

    # Note: CrewAI's ``CustomProviderConfig`` TypedDict only declares
    # ``embedding_callable`` — pydantic strips every other key during
    # ``EmbedderConfig`` validation.  So we cannot pass api_key / model /
    # api_base / max_chars through this dict; instead our embedder class
    # reads them from the ``VLLM_EMBEDDING_*`` env vars at construction
    # time (see ``TruncatingOpenAIEmbedder.__init__``).  The .env checks
    # above guarantee those env vars are populated before we get here.
    return {
        "provider": "custom",
        "config": {
            "embedding_callable": TruncatingOpenAIEmbedder,
        },
    }


def banner(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")
