"""``get_memory_llm`` must disable thinking mode for Qwen3-family models
without touching the main-agent LLM or affecting non-thinking bases.

Live symptom this test guards against
-------------------------------------
CrewAI's ``crewai.memory.analyze`` module runs four LLM helpers on
every ``remember()`` and ``search_memory()``.  On Qwen3-Thinking
those helpers were burning 20-30 k tokens per call on chain-of-
thought for a two-line JSON classifier response — and with 3-4
concurrent memory-analyze calls per agent step, the vLLM KV cache
saturated (91% → 99% at 0 tok/s of prompt throughput) until the
OpenAI SDK's 10-min read timeout fired, surfacing as::

    WARNING crewai.memory.analyze: Memory save analysis failed, using defaults

The client-side fix in :func:`trading_crew._common.get_memory_llm`
builds a dedicated LLM instance whose ``additional_params`` bakes in
``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``.
vLLM's chat completions endpoint reads that key and pre-seeds an
empty ``<think></think>`` block, so Qwen3 skips CoT entirely for
exactly these background calls — the main analyst / debate / PM
LLMs are untouched and keep their full reasoning quality.

We exercise three cases here:

1. Qwen3 model → memory LLM gets the enable_thinking=False override.
2. Non-thinking model (Qwen2.5, Llama, OSS) → memory LLM is
   identical to the main LLM (no unknown-kwarg leaks that a
   non-vLLM provider might reject).
3. Existing ``extra_body`` on the underlying LLM must be preserved
   (we merge, not overwrite).
"""

from __future__ import annotations

import os
from contextlib import contextmanager


@contextmanager
def _env(**overrides):
    """Temporarily override environment variables."""
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update({k: v for k, v in overrides.items() if v is not None})
    for k, v in overrides.items():
        if v is None and k in os.environ:
            del os.environ[k]
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_is_thinking_model_detects_qwen3():
    from trading_crew._common import _is_thinking_model

    # Any HF id or provider-prefixed variant of Qwen3 is a thinking model.
    assert _is_thinking_model("Qwen/Qwen3-32B")
    assert _is_thinking_model("Qwen/Qwen3-Thinking-30B-A3B")
    assert _is_thinking_model("hosted_vllm/Qwen/Qwen3-14B")
    assert _is_thinking_model("openai/Qwen3-8B")

    # Anything not Qwen3 is not (yet) considered a thinking model here.
    assert not _is_thinking_model("Qwen/Qwen2.5-7B-Instruct")
    assert not _is_thinking_model("meta-llama/Llama-3.1-70B")
    assert not _is_thinking_model("google/gemma-4-31b-it")
    assert not _is_thinking_model("openai/gpt-oss-120b")
    assert not _is_thinking_model("")


def test_memory_llm_bakes_no_think_for_qwen3():
    from trading_crew import llm_presets
    from trading_crew._common import get_memory_llm

    with _env(
        VLLM_LLM_BASE_URL="http://localhost:8081/v1",
        VLLM_LLM_MODEL="Qwen/Qwen3-32B",
        VLLM_LLM_API_KEY="dummy",
        LLM_PROVIDER_PREFIX="hosted_vllm",
    ):
        # Reset any UI preset overrides that a previous test could have left.
        llm_presets.clear_active()
        llm = get_memory_llm()

    assert "qwen3" in llm.model.lower()
    ap = llm.additional_params or {}
    kwargs = (ap.get("extra_body") or {}).get("chat_template_kwargs") or {}
    assert kwargs.get("enable_thinking") is False, (
        f"memory LLM must have enable_thinking=False; got additional_params={ap!r}"
    )


def test_memory_llm_is_noop_for_non_thinking_model():
    """A Qwen2.5-Instruct / Llama base must not have extra_body injected —
    those providers may 400 on an unknown chat_template_kwargs field."""
    from trading_crew import llm_presets
    from trading_crew._common import get_memory_llm

    with _env(
        VLLM_LLM_BASE_URL="http://localhost:8081/v1",
        VLLM_LLM_MODEL="Qwen/Qwen2.5-7B-Instruct",
        VLLM_LLM_API_KEY="dummy",
        LLM_PROVIDER_PREFIX="hosted_vllm",
    ):
        llm_presets.clear_active()
        llm = get_memory_llm()

    ap = llm.additional_params or {}
    assert "extra_body" not in ap, (
        f"non-thinking model must NOT get extra_body; got additional_params={ap!r}"
    )


def test_max_tokens_uncapped_by_default():
    """Default MUST be uncapped — capping mid-``<think>`` on Qwen3
    truncates responses and leaks raw ``<think>`` into content.  The
    proper throughput lever is concurrency (session manager), not a
    per-request length cap."""
    from trading_crew import llm_presets
    from trading_crew._common import get_llm

    for model in (
        "Qwen/Qwen3-32B",
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "openai/gpt-oss-120b",
        "google/gemma-4-31b-it",
    ):
        with _env(
            VLLM_LLM_BASE_URL="http://localhost:8081/v1",
            VLLM_LLM_MODEL=model,
            VLLM_LLM_API_KEY="dummy",
            LLM_PROVIDER_PREFIX="hosted_vllm",
            LLM_MAX_TOKENS=None,  # default
        ):
            llm_presets.clear_active()
            llm = get_llm()
        assert llm.max_tokens is None, (
            f"{model}: default must be uncapped; got max_tokens={llm.max_tokens!r}"
        )


def test_max_tokens_env_override_is_opt_in():
    """``LLM_MAX_TOKENS > 0`` is an explicit opt-in runaway-insurance
    ceiling — applied uniformly to thinking and non-thinking models
    (the operator has taken responsibility for the truncation risk)."""
    from trading_crew import llm_presets
    from trading_crew._common import get_llm

    for model in ("Qwen/Qwen3-32B", "Qwen/Qwen2.5-7B-Instruct", "openai/gpt-oss-120b"):
        with _env(
            VLLM_LLM_BASE_URL="http://localhost:8081/v1",
            VLLM_LLM_MODEL=model,
            VLLM_LLM_API_KEY="dummy",
            LLM_PROVIDER_PREFIX="hosted_vllm",
            LLM_MAX_TOKENS="20000",
        ):
            llm_presets.clear_active()
            llm = get_llm()
        assert llm.max_tokens == 20000, (
            f"{model}: LLM_MAX_TOKENS=20000 must be honoured; got {llm.max_tokens!r}"
        )


def test_max_tokens_zero_disables_cap():
    """``LLM_MAX_TOKENS=0`` and negative values must disable the cap."""
    from trading_crew import llm_presets
    from trading_crew._common import get_llm

    for value in ("0", "-1"):
        with _env(
            VLLM_LLM_BASE_URL="http://localhost:8081/v1",
            VLLM_LLM_MODEL="Qwen/Qwen3-32B",
            VLLM_LLM_API_KEY="dummy",
            LLM_PROVIDER_PREFIX="hosted_vllm",
            LLM_MAX_TOKENS=value,
        ):
            llm_presets.clear_active()
            llm = get_llm()
        assert llm.max_tokens is None, (
            f"LLM_MAX_TOKENS={value} must disable the cap; got {llm.max_tokens!r}"
        )


def test_memory_llm_preserves_pre_existing_additional_params():
    """If a caller passed additional_params in overrides, the no-think
    injection must merge (not clobber) so user config survives."""
    from trading_crew import llm_presets
    from trading_crew._common import get_memory_llm

    with _env(
        VLLM_LLM_BASE_URL="http://localhost:8081/v1",
        VLLM_LLM_MODEL="Qwen/Qwen3-32B",
        VLLM_LLM_API_KEY="dummy",
        LLM_PROVIDER_PREFIX="hosted_vllm",
    ):
        llm_presets.clear_active()
        llm = get_memory_llm(
            additional_params={
                "extra_body": {
                    "chat_template_kwargs": {"custom_flag": "keep_me"},
                    "guided_json": {"foo": "bar"},
                },
                "other_key": "still_here",
            }
        )

    ap = llm.additional_params or {}
    assert ap.get("other_key") == "still_here"

    extra = ap.get("extra_body") or {}
    assert extra.get("guided_json") == {"foo": "bar"}, "sibling extra_body keys must survive"

    kwargs = extra.get("chat_template_kwargs") or {}
    assert kwargs.get("enable_thinking") is False, "no-think must be baked in"
    assert kwargs.get("custom_flag") == "keep_me", (
        "pre-existing chat_template_kwargs must survive"
    )
