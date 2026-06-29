"""LLM preset catalog + runtime override mechanism.

The 18-agent crew talks to whichever OpenAI-compatible endpoint the
``.env`` resolves via ``get_llm()``.  This module gives the **UI** a
typed catalog of presets the user can pick from (open-source local
vLLM, hosted vLLM, closed-source OpenAI / Anthropic, …) and a
thread-local override that ``get_llm()`` consults *before* falling back
to the env chain.

Why thread-local?
=================
A single FastAPI process can be running multiple WS-driven analyses
concurrently (one thread per session via ``AnalysisRunner``).  Each
session may pick a different LLM.  Process-global env mutation would
race; ``contextvars`` is task-bound not thread-bound; ``threading.local``
binds cleanly to the worker thread the runner blocks on, which is also
the thread that builds the CrewAI agents.

Per-preset shape
================
Each preset is a typed ``LlmPreset`` carrying the *static* parts of an
LLM config (model, base_url, provider prefix, kind) and the **env var
name** that holds the API key (so the actual secret stays in ``.env``
and never crosses the WS boundary).

Adding a new preset
===================
* Open-source / hosted vLLM: copy ``"hosted-vllm-oss"`` and edit
  the URL / model / api_env. Set ``kind="open-source"``.
* Closed-source: set ``provider="openai"`` (works for OpenAI proper) or
  ``provider="anthropic"`` (LiteLLM routes via ``anthropic/<model>``),
  point ``api_env`` at the secret env var the user sets in .env, set
  ``kind="closed-source"``.

The UI calls ``/api/options`` to learn what's available; the runner
applies the user's selection via ``set_active(preset_id)`` before
``crew.kickoff()`` and clears it again in a ``finally`` block.
"""

from __future__ import annotations

import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


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
    "hosted-vllm-oss": LlmPreset(
        id="hosted-vllm-oss",
        label="Hosted vLLM (VLLM_LLM_*)",
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
        # Env-driven too — the user supplies the model + URL via
        # ``LOCAL_LLM_*`` in .env.  Falls back to ``localhost:8081``
        # which is the conventional vLLM ``--port`` default.
        model=os.environ.get("LOCAL_LLM_MODEL"),
        base_url=os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:8081/v1"),
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


def list_presets() -> List[Dict[str, object]]:
    """Serialisable preset list for ``/api/options``.

    Includes a per-preset ``api_key_configured`` flag so the UI can grey
    out closed-source options the user hasn't pointed at a real key yet
    (the env var either doesn't exist or holds the placeholder ``dummy``).
    """
    out: List[Dict[str, object]] = []
    for preset in BUILTIN_PRESETS.values():
        api_key_val = os.environ.get(preset.api_env) if preset.api_env else None
        configured = bool(api_key_val) and api_key_val.lower() != "dummy"
        # Local vLLM doesn't need a real key — the server ignores it.
        # We only require ``LOCAL_LLM_MODEL`` to be set so the dropdown
        # entry isn't selectable when the user hasn't configured anything.
        if preset.id == "local-vllm":
            configured = bool(os.environ.get("LOCAL_LLM_MODEL"))
        d = asdict(preset)
        d["api_key_configured"] = configured
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Thread-local active override.  ``get_llm()`` consults this *before* the
# env chain so a WS session can swap LLMs without touching ``os.environ``.
# ---------------------------------------------------------------------------

_active = threading.local()


def set_active(preset_id: Optional[str]) -> Optional[LlmPreset]:
    """Mark ``preset_id`` as active for the current thread.

    Pass ``None`` (or an unknown id) to clear the override — the env-var
    chain then takes over again.  Returns the resolved ``LlmPreset`` for
    logging convenience, or ``None`` when the override was cleared.
    """
    if not preset_id:
        _active.preset = None
        return None
    preset = BUILTIN_PRESETS.get(preset_id)
    _active.preset = preset
    return preset


def clear_active() -> None:
    _active.preset = None


def get_active_overrides() -> Dict[str, Optional[str]]:
    """Return the kwargs ``get_llm()`` should overlay on top of its env defaults.

    Empty dict when no preset is active on this thread.  ``base_url=None``
    is intentionally preserved for closed-source presets (OpenAI /
    Anthropic via LiteLLM route by provider+model, no base_url override)
    so the caller can explicitly clear it instead of accidentally
    inheriting the local vLLM endpoint from ``.env``.
    """
    preset = getattr(_active, "preset", None)
    if preset is None:
        return {}
    return preset.to_overrides()


def get_active_preset_id() -> Optional[str]:
    preset = getattr(_active, "preset", None)
    return preset.id if preset is not None else None
