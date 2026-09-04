"""LLM preset catalog + runtime override mechanism.

The 18-agent crew talks to whichever OpenAI-compatible endpoint the
``.env`` resolves via ``get_llm()``.  This module gives the **UI** a
typed catalog of presets the user can pick from (open-source local
vLLM, hosted vLLM, closed-source OpenAI / Anthropic, …) and a
context-var override that ``get_llm()`` consults *before* falling back
to the env chain.

Why a ContextVar (not process env, not bare threading.local)?
=============================================================
A single FastAPI process can be running multiple WS-driven analyses
concurrently (one thread per session via ``AnalysisRunner``).  Each
session may pick a different LLM.  Process-global env mutation would
race.  ``threading.local`` stays on the runner thread, so graph
``ThreadPoolExecutor`` workers (parallel analysts / risk wings) that
call ``get_llm()`` mid-tool — ``research_web`` summarize — silently
fall through to ``VLLM_LLM_*`` while the desks use the selected
preset.  Report map-reduce uses ``REPORT_PROCESS_*`` from the environment via
``get_report_process_llm()``.  A ContextVar copied into those workers
(``submit_in_active_context``) keeps one selected desk endpoint for
the rest of the run.

Per-preset shape
================
Each preset is a typed ``LlmPreset`` carrying the *static* parts of an
LLM config (model, base_url, provider prefix, kind) and the **env var
name** that holds the API key (so the actual secret stays in ``.env``
and never crosses the WS boundary).

Adding a new preset
===================
* Open-source / hosted vLLM: copy ``"hosted-vllm"`` and edit
  the URL / model / api_env. Set ``kind="open-source"``.
* Open-source / llama.cpp: copy ``"llamacpp-deepseek-v4"`` and edit
  the URL / model / api_env.  ``llama-server``'s OpenAI-compatible
  endpoint ignores the model name, so ``model`` is just a friendly label.
* Closed-source: set ``provider="openai"`` (works for OpenAI proper) or
  ``provider="anthropic"`` (LiteLLM routes via ``anthropic/<model>``),
  point ``api_env`` at the secret env var the user sets in .env, set
  ``kind="closed-source"``.

The UI calls ``/api/options`` to learn what's available; the runner
applies the user's selection via ``set_active(preset_id)`` before
``crew.kickoff()`` and clears it again in a ``finally`` block.
"""

from __future__ import annotations

import contextvars
import os
import threading
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class LlmPreset:
    """One selectable LLM configuration the UI offers in its dropdown."""

    id: str
    label: str
    kind: str  # "open-source" | "closed-source"
    provider: str  # LiteLLM provider prefix, e.g. "hosted_vllm" / "openai" / "anthropic"
    model: str
    base_url: Optional[str] = None
    api_env: str = "OPENAI_API_KEY"
    description: str = ""
    context_window: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    def to_overrides(self) -> Dict[str, Optional[str]]:
        """Resolve into the kwargs ``get_llm()`` actually uses.

        We read the API key from the env-var NAMED by ``api_env`` so
        the secret stays out of WS payloads / logs / browser memory.
        ``None`` is a valid value here — ``get_llm()`` keeps any field
        whose override is ``None``.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": os.environ.get(self.api_env) if self.api_env else None,
        }


# ---------------------------------------------------------------------------
# Built-in presets — extend freely.  ``id`` must be URL-safe (it travels
# in the WS payload and the recent-runs metadata).
# ---------------------------------------------------------------------------

BUILTIN_PRESETS: Dict[str, LlmPreset] = {
    "hosted-vllm": LlmPreset(
        id="hosted-vllm",
        label="Hosted vLLM (VLLM_LLM_*)",
        kind="open-source",
        provider="hosted_vllm",
        model=os.environ.get("VLLM_LLM_MODEL"),
        base_url=os.environ.get("VLLM_LLM_BASE_URL"),
        api_env="VLLM_LLM_API_KEY",
        description=(
            "Any OpenAI-compatible vLLM-style endpoint you point "
            "VLLM_LLM_BASE_URL / VLLM_LLM_MODEL / VLLM_LLM_API_KEY at "
            "in .env. Desks, planner, and critic use this endpoint when "
            "this preset is active. Report map-reduce uses "
            "REPORT_PROCESS_* from .env."
        ),
        context_window=131_072,
        tags=["default", "vLLM", "long-context"],
    ),
    "hosted-vllm-oss": LlmPreset(
        id="hosted-vllm-oss",
        label="Hosted vLLM OSS (VLLM_LLM_*)",
        kind="open-source",
        provider="hosted_vllm",
        # Entirely env-driven — no deployment-specific URL baked into
        # the codebase.  Point ``VLLM_LLM_BASE_URL`` / ``VLLM_LLM_MODEL``
        # / ``VLLM_LLM_API_KEY`` at any OpenAI-compatible vLLM-style
        # endpoint in .env to enable this preset.
        model=os.environ.get("VLLM_LLM_MODEL"),
        base_url=os.environ.get("VLLM_LLM_BASE_URL"),
        api_env="VLLM_LLM_API_KEY",
        description=(
            "Any OpenAI-compatible vLLM-style endpoint you point "
            "VLLM_LLM_BASE_URL / VLLM_LLM_MODEL / VLLM_LLM_API_KEY at "
            "in .env.  Recommended default for debate rounds that "
            "ship all 8 analyst reports — pick a long-context model "
            "(131K-ish) so the prompts don't overflow."
        ),
        context_window=131_072,
        tags=["default", "vLLM", "long-context"],
    ),
    "local-vllm": LlmPreset(
        id="local-vllm",
        label="Local vLLM (LOCAL_LLM_*)",
        kind="open-source",
        provider="hosted_vllm",
        # Env-driven — model + URL come from ``LOCAL_LLM_*`` in .env.
        model=os.environ.get("LOCAL_LLM_MODEL"),
        base_url=os.environ.get("LOCAL_LLM_BASE_URL"),
        # vLLM accepts any non-empty bearer token; the local server doesn't auth.
        api_env="OPENAI_API_KEY",
        description=(
            "A vLLM server you launched yourself with "
            "`vllm serve <model> --port 8081` (or wherever "
            "LOCAL_LLM_BASE_URL points).  Tool calling enabled. "
            "Watch the context window — keep debate rounds within "
            "the model's limit."
        ),
        context_window=None,
        tags=["local", "vLLM", "tool-calling"],
    ),
    "llamacpp-deepseek-v4": LlmPreset(
        id="llamacpp-deepseek-v4",
        label="Llama.cpp · DeepSeek-V4 (LLAMACPP_LLM_*)",
        kind="open-source",
        provider="hosted_vllm",
        # Env-driven — point ``LLAMACPP_LLM_BASE_URL`` at a llama.cpp
        # ``llama-server`` OpenAI-compatible endpoint (default port 8001).
        # llama.cpp serves the single loaded model and ignores the model
        # name sent in the request, so ``LLAMACPP_LLM_MODEL`` is just the
        # friendly label shown in the UI.
        model=os.environ.get("LLAMACPP_LLM_MODEL"),
        base_url=os.environ.get("LLAMACPP_LLM_BASE_URL", "http://localhost:8001/v1"),
        # llama.cpp accepts any non-empty bearer token; the server doesn't auth.
        api_env="LLAMACPP_LLM_API_KEY",
        description=(
            "A llama.cpp server you launched yourself with "
            "`llama-server -m <model>.gguf --port 8001` (or wherever "
            "LLAMACPP_LLM_BASE_URL points).  OpenAI-compatible. "
            "Long context window (131K n_ctx by default).  Watch "
            "debate-round token budgets — llama.cpp is CPU/GPU-hosted "
            "and slower than a hosted vLLM on big prompts."
        ),
        context_window=131_072,
        tags=["local", "llama.cpp", "deepseek", "long-context"],
    ),
    "openai-gpt-4o-mini": LlmPreset(
        id="openai-gpt-4o-mini",
        label="OpenAI · gpt-4o-mini (closed)",
        kind="closed-source",
        provider="openai",
        model="gpt-4o-mini",
        base_url=None,
        api_env="OPENAI_PROD_API_KEY",
        description=(
            "OpenAI's gpt-4o-mini. Closed-source, billed per token. "
            "Set OPENAI_PROD_API_KEY in .env (kept separate from "
            "OPENAI_API_KEY=dummy used by the local vLLM client)."
        ),
        context_window=128_000,
        tags=["closed-source", "OpenAI"],
    ),
    "openai-gpt-4o": LlmPreset(
        id="openai-gpt-4o",
        label="OpenAI · gpt-4o (closed)",
        kind="closed-source",
        provider="openai",
        model="gpt-4o",
        base_url=None,
        api_env="OPENAI_PROD_API_KEY",
        description="OpenAI's flagship gpt-4o. Higher cost than mini; recommended only for the PM / Reflective Critic if you want a closed-source second opinion.",
        context_window=128_000,
        tags=["closed-source", "OpenAI"],
    ),
    "anthropic-claude-sonnet": LlmPreset(
        id="anthropic-claude-sonnet",
        label="Anthropic · claude-3-5-sonnet (closed)",
        kind="closed-source",
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        base_url=None,
        api_env="ANTHROPIC_API_KEY",
        description="Anthropic Claude 3.5 Sonnet via LiteLLM. Set ANTHROPIC_API_KEY in .env.",
        context_window=200_000,
        tags=["closed-source", "Anthropic"],
    ),
}


# ---------------------------------------------------------------------------
# Runtime-registered presets.  The web UI's "serve a model with / without
# the RLMF LoRA adapter" control launches a vLLM server and then registers
# the served endpoint(s) here so they show up in the same run-config LLM
# dropdown as the built-ins — the operator picks "…(with LoRA)" or
# "…(base)" at inference time without touching ``.env``.  We keep them in
# a *separate* dict (merged into the catalog by ``list_presets`` /
# ``set_active``) so they can be swapped wholesale when the served model
# changes, and never accidentally persist into the built-in catalog.
# ---------------------------------------------------------------------------

_RUNTIME_PRESETS: Dict[str, LlmPreset] = {}
_RUNTIME_LOCK = threading.Lock()


def register_runtime_preset(preset: LlmPreset) -> None:
    """Add / replace a runtime preset (keyed by ``preset.id``)."""
    with _RUNTIME_LOCK:
        _RUNTIME_PRESETS[preset.id] = preset


def clear_runtime_presets(prefix: Optional[str] = None) -> None:
    """Drop runtime presets — all of them, or only those whose id starts
    with ``prefix`` (used when a served model is stopped / replaced)."""
    with _RUNTIME_LOCK:
        if prefix is None:
            _RUNTIME_PRESETS.clear()
            return
        for pid in [p for p in _RUNTIME_PRESETS if p.startswith(prefix)]:
            _RUNTIME_PRESETS.pop(pid, None)


def _resolve_preset(preset_id: str) -> Optional[LlmPreset]:
    """Look a preset up in the runtime registry first, then the built-ins."""
    with _RUNTIME_LOCK:
        rt = _RUNTIME_PRESETS.get(preset_id)
    return rt or BUILTIN_PRESETS.get(preset_id)


def resolve_preset(preset_id: str) -> Optional[LlmPreset]:
    """Read-only lookup of a preset by id (runtime registry, then built-ins).

    Unlike :func:`set_active` this does NOT touch the thread-local override —
    for callers that only need a preset's static fields (e.g. the scale
    runner's LLM readiness gate, which must probe the endpoint the batch
    will actually use without activating it on the request thread).
    """
    return _resolve_preset(preset_id)


def list_presets() -> List[Dict[str, object]]:
    """Serialisable preset list for ``/api/options``.

    Includes a per-preset ``api_key_configured`` flag so the UI can grey
    out closed-source options the user hasn't pointed at a real key yet
    (the env var either doesn't exist or holds the placeholder ``dummy``).
    """
    out: List[Dict[str, object]] = []
    with _RUNTIME_LOCK:
        runtime = list(_RUNTIME_PRESETS.values())
    # Runtime (server-launched) presets first so they float to the top of
    # the dropdown right after being served.
    for preset in [*runtime, *BUILTIN_PRESETS.values()]:
        api_key_val = os.environ.get(preset.api_env) if preset.api_env else None
        configured = bool(api_key_val) and api_key_val.lower() != "dummy"
        # Local vLLM doesn't need a real key — the server ignores it.
        # We only require ``LOCAL_LLM_MODEL`` to be set so the dropdown
        # entry isn't selectable when the user hasn't configured anything.
        if preset.id == "local-vllm":
            configured = bool(os.environ.get("LOCAL_LLM_MODEL"))
        # llama.cpp doesn't auth either; it lights up once the operator
        # points at a server (base_url) or names the loaded model.
        elif preset.id == "llamacpp-deepseek-v4":
            configured = bool(
                os.environ.get("LLAMACPP_LLM_MODEL")
                or os.environ.get("LLAMACPP_LLM_BASE_URL")
            )
        # Runtime presets point at a server we just launched locally; the
        # vLLM endpoint ignores the bearer token, so they're always usable.
        if preset.id in _RUNTIME_PRESETS:
            configured = True
        d = asdict(preset)
        d["api_key_configured"] = configured
        d["runtime"] = preset.id in _RUNTIME_PRESETS
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Per-run active override.  ``get_llm()`` consults this *before* the env
# chain so a WS session can swap LLMs without touching ``os.environ``.
# ContextVar so graph thread-pool workers can inherit it (see
# ``submit_in_active_context``).  Concurrent runner threads still each
# see their own value.
# ---------------------------------------------------------------------------

_active_preset: contextvars.ContextVar[Optional[LlmPreset]] = contextvars.ContextVar(
    "tradingcrew_llm_preset", default=None
)


def set_active(preset_id: Optional[str]) -> Optional[LlmPreset]:
    """Mark ``preset_id`` as active for the current context.

    Pass ``None`` (or an unknown id) to clear the override — the env-var
    chain then takes over again.  Returns the resolved ``LlmPreset`` for
    logging convenience, or ``None`` when the override was cleared.
    """
    if not preset_id:
        _active_preset.set(None)
        return None
    preset = _resolve_preset(preset_id)
    _active_preset.set(preset)
    return preset


def clear_active() -> None:
    _active_preset.set(None)


def get_active_overrides() -> Dict[str, Optional[str]]:
    """Return the kwargs ``get_llm()`` should overlay on top of its env defaults.

    Empty dict when no preset is active on this context.  ``base_url=None``
    is intentionally preserved for closed-source presets (OpenAI /
    Anthropic via LiteLLM route by provider+model, no base_url override)
    so the caller can explicitly clear it instead of accidentally
    inheriting the local vLLM endpoint from ``.env``.
    """
    preset = _active_preset.get()
    if preset is None:
        return {}
    return preset.to_overrides()


def get_active_preset_id() -> Optional[str]:
    preset = _active_preset.get()
    return preset.id if preset is not None else None


def submit_in_active_context(executor: Any, fn: Callable, /, *args: Any, **kwargs: Any):
    """``executor.submit`` that copies the current LLM preset into the worker.

    Bare ``ThreadPoolExecutor`` drops ContextVars. Parallel graph desks
    that call ``get_llm()`` (tool summarize) would otherwise hit
    ``VLLM_LLM_*`` while the selected preset is local.
    """
    ctx = contextvars.copy_context()
    return executor.submit(ctx.run, partial(fn, *args, **kwargs))
