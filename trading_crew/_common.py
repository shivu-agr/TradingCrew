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

    # ------------------------------------------------------------------
    # Robustness knobs (Phase 2G — Gemma-4 stall fix).
    #
    # ``timeout`` ----------------------------------------------------
    # Without an explicit ``timeout`` CrewAI's underlying OpenAI client
    # falls back to its 600s read timeout *plus* up to two implicit
    # retries — so a single stalled call on a saturated vLLM can quietly
    # burn 20+ minutes before either succeeding or failing the run.  We
    # instead set a long-but-bounded ceiling so a true hang is
    # recoverable, while leaving plenty of headroom for legit slow
    # generations on a busy box (Gemma-4 / OSS-120B routinely take 8-12
    # minutes per debate round when the GPU is contended).
    #
    # ``max_retries`` ------------------------------------------------
    # The OpenAI SDK defaults to ``max_retries=2`` and retries on any
    # 5xx / connection / read-timeout failure by opening a *new* HTTP
    # connection — but vLLM never learns the previous request was
    # abandoned, so it keeps generating tokens for the orphaned call
    # as a "zombie request".  Verified live with ``vllm:num_requests
    # _running=3`` while only 2 client sockets were ESTAB on our
    # runner — i.e. one zombie per stuck Bullish Researcher call.
    # Zombies consume GPU decode slots and slow down the live request
    # the client *is* waiting on, which makes the stall worse.
    #
    # We default to ``max_retries=0`` so a failed call surfaces as the
    # ``[[DEGRADED_LLM_TIMEOUT]]`` placeholder (and the run continues)
    # instead of piling up zombies that the OpenAI client thinks were
    # retried successfully.  Operators on flaky transports can bump it
    # back via ``LLM_MAX_RETRIES`` in .env.
    #
    # ``max_tokens`` — OPT-IN safety net, not a default performance lever.
    # ---------------------------------------------------------------------
    # Capping ``max_tokens`` is a foot-gun in the general case:
    #   1. For non-thinking models (Gemma-4, OSS-120B, Qwen2.5-Instruct,
    #      Llama-Chat) any cap risks truncating a legitimately long
    #      analyst report below the prompted length budget.
    #   2. For thinking models (Qwen3-Thinking) a cap that fires
    #      *inside* the ``<think>`` block returns a response starting
    #      with a raw, unclosed ``<think>...`` prefix — the reasoning
    #      parser can't split it (needs both start + end tokens) so
    #      the raw monologue leaks into ``content`` and CrewAI accepts
    #      the truncated garbage as the final analyst answer.
    #
    # Verified live during a COCHINSHIP.NS + Qwen3-32B run: a single
    # debate turn emitted 24 k tokens in 10 minutes with no exit
    # condition.  Capping bounds worst-case latency but at real
    # quality cost.  The better lever for throughput is to run
    # multiple analyses concurrently (see the session-manager path
    # in ``web/backend/session_manager.py``) so vLLM's decode batch
    # size actually grows past 1 — a 4-way batch on 1×H100 with
    # Qwen3-32B triples aggregate tokens/sec on the workflow.
    #
    # We therefore leave ``max_tokens`` UNCAPPED by default (vLLM's
    # server-side ``max_model_len - prompt_tokens`` remains the
    # ceiling).  An operator who wants runaway-insurance on top can
    # opt in by setting ``LLM_MAX_TOKENS`` in the env — the value is
    # applied to *both* thinking and non-thinking models on the
    # explicit understanding that it may truncate.  ``0`` (default) or
    # a negative value disables the cap entirely.
    try:
        params["timeout"] = float(os.environ.get("LLM_TIMEOUT_SECONDS", "720"))
    except ValueError:
        params["timeout"] = 720.0
    try:
        params["max_retries"] = int(os.environ.get("LLM_MAX_RETRIES", "0"))
    except ValueError:
        params["max_retries"] = 0

    try:
        max_tokens_cap = int(os.environ.get("LLM_MAX_TOKENS", "0"))
    except ValueError:
        max_tokens_cap = 0
    if max_tokens_cap > 0:
        params["max_tokens"] = max_tokens_cap

    # Direct kwargs win last.
    params.update(overrides)
    # base_url=None confuses some LiteLLM providers — drop it for the
    # closed-source path where the SDK handles routing itself.
    if params.get("base_url") is None:
        params.pop("base_url", None)
    return LLM(**params)


# ---------------------------------------------------------------------------
# Memory-side LLM — thinking disabled for background utility calls.
# ---------------------------------------------------------------------------
#
# CrewAI's memory subsystem (unified_memory.Memory) calls four LLM
# helpers on every ``remember()`` and ``search_memory()``:
#
#   * ``crewai.memory.analyze.extract_memories_from_content``
#   * ``crewai.memory.analyze.analyze_query``
#   * ``crewai.memory.analyze.analyze_for_save``
#   * ``crewai.memory.analyze.analyze_for_consolidation``
#
# The prompts are small-schema classifiers ("pick a scope path, list
# 3-5 categories, importance 0-1") — the analyst-quality trade-off
# you'd worry about for a real reasoning task simply does not apply.
# But with Qwen3-Thinking as the default LLM, each of these calls
# spent 20-30 k tokens generating a ``<think>`` block before emitting
# its two-line JSON, and 3-4 of them run in parallel (see
# ``encoding_flow.py`` ThreadPoolExecutor).  In a live ZENTEC.NS run
# the vLLM logs showed KV cache climbing 91 % → 99 % at 0 prompt
# tok/s across 4 concurrent memory-analyze requests until the client
# hit its 10-min read timeout and gave up ("Memory save analysis
# failed, using defaults").
#
# The fix is client-side and *only* touches memory: we hand CrewAI a
# dedicated LLM instance whose ``additional_params`` bakes in
# ``extra_body={"chat_template_kwargs": {"enable_thinking": false}}``.
# vLLM 0.11.2's chat completions endpoint reads that key and forces
# Qwen3's template to pre-seed an empty ``<think></think>`` block, so
# the model skips CoT entirely for exactly these background calls.
# The main-analyst / debate / PM LLM is unchanged — those still get
# vanilla Qwen3-Thinking reasoning quality.
#
# For non-thinking models (OSS-120B, Gemma-4, Qwen2.5-Instruct, plain
# Llama), we return the LLM without any override — ``chat_template
# _kwargs`` would be ignored by vLLM anyway, but litellm complains
# about unknown params on some providers.
_THINKING_MODEL_MARKERS = ("qwen3", "qwen/qwen3", "qwen3-thinking")


def _is_thinking_model(model_name: str) -> bool:
    """True iff ``model_name`` is a HF id / provider-prefixed name for a
    reasoning model that honours the ``enable_thinking`` chat-template
    kwarg (Qwen3 family today)."""
    if not model_name:
        return False
    lower = model_name.lower()
    return any(marker in lower for marker in _THINKING_MODEL_MARKERS)


def get_memory_llm(**overrides):
    """Return an LLM instance for CrewAI's memory analyze subsystem.

    Identical to :func:`get_llm` except that when the resolved model
    is a known thinking model (Qwen3 family), we bake in
    ``additional_params={"extra_body": {"chat_template_kwargs":
    {"enable_thinking": False}}}`` so every memory-analyze call skips
    the chain-of-thought phase.  Non-thinking models get the vanilla
    LLM unchanged.

    Thread-safe by construction: the returned LLM instance is
    immutable-in-config and ``additional_params`` is set once at
    construction, so concurrent ``analyze_for_*`` calls from CrewAI's
    encoding-flow thread pool cannot race on it.
    """
    llm = get_llm(temperature=0.1, **overrides)  # low temp — utility calls, not creativity
    model_name = str(getattr(llm, "model", "") or "")
    if not _is_thinking_model(model_name):
        return llm

    existing = dict(getattr(llm, "additional_params", None) or {})
    existing_extra = dict(existing.get("extra_body") or {})
    existing_kwargs = dict(existing_extra.get("chat_template_kwargs") or {})
    # Explicit request override wins; otherwise force False.
    existing_kwargs.setdefault("enable_thinking", False)
    existing_extra["chat_template_kwargs"] = existing_kwargs
    existing["extra_body"] = existing_extra
    llm.additional_params = existing
    return llm


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
