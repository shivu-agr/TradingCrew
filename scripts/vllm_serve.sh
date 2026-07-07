#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# vllm_serve.sh — serve a base LLM (default Qwen/Qwen3-32B) with or
# without the RLMF LoRA adapter produced by scripts/rlmf_train.sh.
#
# Typical flow:
#
#   # 1) train
#   bash scripts/rlmf_train.sh --lora --base-model Qwen/Qwen3-32B --epochs 3
#
#   # 2a) serve BASE model only (baseline reference)
#   bash scripts/vllm_serve.sh --without-lora
#
#   # 2b) serve BASE + LoRA adapter
#   bash scripts/vllm_serve.sh --with-lora
#
# The training script writes the adapter to
#   ${LORA_OUTPUT_DIR:-${TRADINGCREW_CACHE_DIR:-~/.trading_crew}/rlmf/lora_adapter}/adapter
# and this serve script reads the same env vars, so wiring "train then
# serve" is a single environment.  On any base model besides
# Qwen/Qwen3-32B, override with `--base-model <hf-id>`.
#
# vLLM 0.11.2 is the target — earlier versions (<= 0.6.6) do not
# support Qwen3-32B or the newer LoRA adapter format.  We resolve the
# vllm binary from ${VLLM_PY_ENV}/bin/vllm and fall back to $(which vllm)
# so users on other machines still get sensible behaviour.
#
# Serving conventions (both modes):
#   * OpenAI-compatible API on http://localhost:${PORT}/v1
#   * bf16 weights (H100 native; falls back to fp16 on Ampere-)
#   * gpu-memory-utilization capped at 0.90 to keep headroom for
#     KV cache spikes and other tenants on the box
#   * max-model-len clamped to 16 384 by default — Qwen3-32B natively
#     supports 131 072 tokens but a 128 K KV cache would blow past
#     the 95 GB budget on a single H100.  Raise via MAX_MODEL_LEN.
#
# With --with-lora, the adapter is registered under the model name
# "rlmf".  Client calls therefore pick their model:
#   * curl ... -d '{"model":"Qwen/Qwen3-32B", ...}'   -> base weights
#   * curl ... -d '{"model":"rlmf", ...}'             -> base + LoRA
# Both are served by the same process; --enable-lora just adds the
# rlmf adapter alongside the base weights.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- config ---------------------------------------------------------------
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-32B}"
_DEFAULT_TRAINING_DIR="${LORA_OUTPUT_DIR:-${TRADINGCREW_CACHE_DIR:-$HOME/.trading_crew}/rlmf/lora_adapter}"
ADAPTER_DIR="${ADAPTER_DIR:-$_DEFAULT_TRAINING_DIR/adapter}"
PORT="${PORT:-8081}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Empty by default → let vLLM pick (256/1024).  Lower it (e.g. 8) to
# squeeze a small served model into the VRAM left by another tenant —
# vLLM warms up the sampler for max_num_seqs dummy requests at startup,
# which is the usual OOM when co-locating a second server on one GPU.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
LORA_NAME="${LORA_NAME:-rlmf}"
DTYPE="${DTYPE:-bfloat16}"
VLLM_PY_ENV="${VLLM_PY_ENV:-/home/gpt_user/.pyenv/versions/gpt_vllm_new}"

# Tool-calling support.  CrewAI (and any agent framework that sets
# ``tool_choice=auto``) requires vLLM to be launched with the auto-tool
# flags, otherwise tool-using requests fail with:
#   "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser
# We default the parser to ``hermes`` because Qwen3 and Qwen2.5 both
# emit tool calls in Hermes format.  For a Llama-family base model,
# pass ``--tool-parser llama3_json``.  Pass ``--no-tool-choice`` to
# omit both flags for a base model that doesn't support tool-calling
# at all (rare — we keep them on by default so agent workflows work).
TOOL_PARSER="${TOOL_PARSER:-hermes}"
ENABLE_TOOL_CHOICE=1

# Reasoning-parser support.  Qwen3 (Thinking), DeepSeek-R1, GPT-OSS,
# etc. emit chain-of-thought inside ``<think>...</think>`` (or a
# model-specific variant).  Without a parser, those raw tokens leak
# into ``choices[0].message.content`` and clients (CrewAI) render the
# monologue as the final analyst report.  With a parser, vLLM strips
# the reasoning span into a separate ``reasoning_content`` field, so
# only the post-think answer lands in ``content``.
#
# Default to ``qwen3`` because our default base is Qwen/Qwen3-32B.
# Common alternatives: ``deepseek_r1`` (DeepSeek-R1 / R1-Distill),
# ``openai_gptoss`` (OSS-120B), ``granite`` (Granite-Reasoning),
# ``minimax_m2``.  Full list: vllm/reasoning/__init__.py.
# Pass ``--no-reasoning-parser`` to disable (e.g. for a plain chat
# model that doesn't emit think tags).
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
ENABLE_REASONING_PARSER=1

# Chat-template override — thinking mode is ON by default so the
# analyst / debate / PM pipeline benefits from Qwen3-Thinking's
# reasoning quality.  Only the *client* (see
# ``trading_crew/_patches.py``) selectively disables it for CrewAI
# background utility calls (memory analyzer, consolidation planner,
# metadata extractor) — those emit 20-30 k tokens of monologue for a
# two-line JSON classifier response and saturate the vLLM KV cache
# with 3-4 concurrent utility calls, stalling real analyst requests
# until the OpenAI SDK's 10-min read timeout fires (verified live in
# a ZENTEC.NS + COCHINSHIP.NS run — vllm logs showed KV cache
# 91% → 99% at 0 prompt tok/s with 4 preemptions).
#
# Pass ``--disable-thinking`` to bake ``enable_thinking=false`` into
# the *server* chat template (via
# ``scripts/chat_templates/qwen3_no_think.jinja``) for e.g. a fast-
# throughput benchmarking server.  vLLM 0.11.2 has no
# ``--chat-template-kwargs`` CLI so the template file is how we pin
# it globally.
DISABLE_THINKING=0
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"

WITH_LORA=1
DRY_RUN=0

usage() {
  cat <<HELP
Usage: $0 [--with-lora | --without-lora] [options]

Modes:
  --with-lora      (default) Serve base + LoRA adapter under name '${LORA_NAME}'
  --without-lora   Serve only the base model

Options:
  --base-model <hf-id>   Base model (default: ${BASE_MODEL})
  --adapter <dir>        LoRA adapter directory (default: ${ADAPTER_DIR})
  --port <n>             HTTP port (default: ${PORT})
    --gpu-mem-util <f>     vLLM --gpu-memory-utilization (default: ${GPU_MEM_UTIL})
  --max-model-len <n>    vLLM --max-model-len (default: ${MAX_MODEL_LEN})
  --max-num-seqs <n>     vLLM --max-num-seqs (default: vLLM's own). Lower it
                         (e.g. 8) to fit a small server alongside another
                         GPU tenant — avoids the sampler-warmup OOM.
  --dtype <t>            weights dtype (default: ${DTYPE})
  --lora-name <s>        served-model name for the adapter (default: ${LORA_NAME})
  --tool-parser <s>      vLLM --tool-call-parser (default: ${TOOL_PARSER})
                         Common values: hermes (Qwen*), llama3_json,
                         mistral, granite, pythonic.
  --no-tool-choice       Omit --enable-auto-tool-choice / --tool-call-parser
                         (use for models without tool-calling support).
  --reasoning-parser <s> vLLM --reasoning-parser (default: ${REASONING_PARSER})
                         Strips <think>...</think> spans out of the
                         assistant content into reasoning_content so the
                         monologue doesn't leak into the final answer.
                         Common: qwen3, deepseek_r1, openai_gptoss,
                         granite, minimax_m2, glm45.
  --no-reasoning-parser  Omit --reasoning-parser (for non-thinking models).
  --disable-thinking     Bake enable_thinking=false into the *server*
                         chat template (from
                         scripts/chat_templates/qwen3_no_think.jinja).
                         Use for a fast, thinking-free throughput
                         server.  NOT the default — the workflow
                         relies on client-side per-request switching
                         (see trading_crew/_patches.py) so analyst
                         quality stays intact.
  --enable-thinking      (default) Serve the vanilla Qwen3-Thinking
                         template.  Clients pass
                         ``chat_template_kwargs={"enable_thinking":
                         false}`` per-request to opt out.
  --chat-template <path> Explicit --chat-template override; takes
                         precedence over --disable-thinking.  Useful
                         for other model families that need a bespoke
                         template.
  --dry-run              print the vllm command that would run and exit
  -h, --help             this help

Env vars honoured (in order of precedence: CLI > env > default):
  BASE_MODEL, ADAPTER_DIR, LORA_OUTPUT_DIR, TRADINGCREW_CACHE_DIR,
  PORT, GPU_MEM_UTIL, MAX_MODEL_LEN, DTYPE, LORA_NAME, VLLM_PY_ENV,
  CHAT_TEMPLATE, TOOL_PARSER, REASONING_PARSER

Client call examples once the server is up:
  # Base model
  curl -s http://localhost:${PORT}/v1/chat/completions \\
       -H 'Content-Type: application/json' \\
       -d '{"model":"${BASE_MODEL}","messages":[{"role":"user","content":"hi"}]}'

  # LoRA-adapted (requires --with-lora)
  curl -s http://localhost:${PORT}/v1/chat/completions \\
       -H 'Content-Type: application/json' \\
       -d '{"model":"${LORA_NAME}","messages":[{"role":"user","content":"hi"}]}'
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-lora)     WITH_LORA=1; shift;;
    --without-lora)  WITH_LORA=0; shift;;
    --base-model)    BASE_MODEL="$2"; shift 2;;
    --adapter)       ADAPTER_DIR="$2"; shift 2;;
    --port)          PORT="$2"; shift 2;;
    --gpu-mem-util)  GPU_MEM_UTIL="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --max-num-seqs)  MAX_NUM_SEQS="$2"; shift 2;;
    --dtype)         DTYPE="$2"; shift 2;;
    --lora-name)     LORA_NAME="$2"; shift 2;;
    --tool-parser)   TOOL_PARSER="$2"; shift 2;;
    --no-tool-choice) ENABLE_TOOL_CHOICE=0; shift;;
    --reasoning-parser) REASONING_PARSER="$2"; shift 2;;
    --no-reasoning-parser) ENABLE_REASONING_PARSER=0; shift;;
    --disable-thinking) DISABLE_THINKING=1; shift;;
    --enable-thinking)  DISABLE_THINKING=0; shift;;
    --chat-template) CHAT_TEMPLATE="$2"; shift 2;;
    --dry-run)       DRY_RUN=1; shift;;
    -h|--help)       usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 1;;
  esac
done

# ---- vllm binary resolution ----------------------------------------------
VLLM_BIN="$VLLM_PY_ENV/bin/vllm"
if [ ! -x "$VLLM_BIN" ]; then
  VLLM_BIN="$(command -v vllm || true)"
fi
if [ -z "$VLLM_BIN" ] || [ ! -x "$VLLM_BIN" ]; then
  echo "ERR: vllm binary not found."  >&2
  echo "     Tried: ${VLLM_PY_ENV}/bin/vllm and \$PATH." >&2
  echo "     Install vllm >= 0.11.2 or set VLLM_PY_ENV to the right pyenv." >&2
  exit 2
fi
# ``vllm --version`` also spawns subprocess init (its own log lines
# swallow the version number), so read the vLLM package metadata
# directly from the target pyenv instead — that's fast and correct.
VLLM_VER=$(head -3 "$VLLM_PY_ENV"/lib/python*/site-packages/vllm-*.dist-info/METADATA 2>/dev/null \
             | awk 'tolower($1)=="version:" {print $2; exit}')
echo "==> vllm binary: $VLLM_BIN  (v${VLLM_VER:-unknown})"

# ---- adapter readiness check (only when --with-lora) ---------------------
if [ "$WITH_LORA" = 1 ]; then
  if [ ! -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
    echo "ERR: no adapter at $ADAPTER_DIR"  >&2
    echo "     Expected file: $ADAPTER_DIR/adapter_model.safetensors" >&2
    echo "     Run the training first:" >&2
    echo "       bash scripts/rlmf_train.sh --lora --base-model $BASE_MODEL" >&2
    echo "     Or pass --without-lora to serve the base model alone." >&2
    exit 3
  fi
  # Sanity: adapter's base_model_name_or_path should match BASE_MODEL.
  # A mismatch is a foot-gun: peft won't fail at load, but the LoRA
  # will silently produce garbage because it was trained against a
  # different Q/K/V projection space.
  if command -v python >/dev/null; then
    ADAPTER_BASE=$(python -c "import json; print(json.load(open('$ADAPTER_DIR/adapter_config.json')).get('base_model_name_or_path',''))" 2>/dev/null || echo "")
    if [ -n "$ADAPTER_BASE" ] && [ "$ADAPTER_BASE" != "$BASE_MODEL" ]; then
      echo "WARN: adapter's base_model_name_or_path='$ADAPTER_BASE' but --base-model='$BASE_MODEL'." >&2
      echo "      LoRA weights are model-specific; a mismatched pair will produce garbage." >&2
      echo "      Continuing anyway — pass --base-model $ADAPTER_BASE to be safe." >&2
    fi
  fi
fi

# ---- build vllm invocation -----------------------------------------------
COMMON_ARGS=(
  serve "$BASE_MODEL"
  --port "$PORT"
  --dtype "$DTYPE"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --served-model-name "$BASE_MODEL"
)

if [ -n "$MAX_NUM_SEQS" ]; then
  COMMON_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

# CrewAI / any agent framework that emits ``tool_choice=auto`` needs
# both flags below; without them, vLLM answers every tool-using
# request with a 400.  See TOOL_PARSER comment above the defaults.
if [ "$ENABLE_TOOL_CHOICE" = 1 ]; then
  COMMON_ARGS+=(
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_PARSER"
  )
fi

if [ "$ENABLE_REASONING_PARSER" = 1 ]; then
  COMMON_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

# Resolve final chat-template path.  Explicit --chat-template wins;
# otherwise, if --disable-thinking is set and we're serving a Qwen3
# base, use the pinned no-think template.  Other base models are
# left with their default template.
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_NO_THINK_TEMPLATE="$_SCRIPT_DIR/chat_templates/qwen3_no_think.jinja"
if [ -z "$CHAT_TEMPLATE" ] && [ "$DISABLE_THINKING" = 1 ]; then
  case "$BASE_MODEL" in
    Qwen/Qwen3-*|Qwen3-*)
      if [ -r "$_NO_THINK_TEMPLATE" ]; then
        CHAT_TEMPLATE="$_NO_THINK_TEMPLATE"
      else
        echo "WARN: --disable-thinking requested but $_NO_THINK_TEMPLATE not found." >&2
        echo "      Falling back to vanilla Qwen3-Thinking template." >&2
      fi
      ;;
  esac
fi
if [ -n "$CHAT_TEMPLATE" ]; then
  if [ ! -r "$CHAT_TEMPLATE" ]; then
    echo "ERR: --chat-template file not readable: $CHAT_TEMPLATE" >&2
    exit 4
  fi
  echo "==> chat template: $CHAT_TEMPLATE"
  COMMON_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

if [ "$WITH_LORA" = 1 ]; then
  echo "==> mode: BASE + LoRA (adapter=$ADAPTER_DIR, served as '$LORA_NAME')"
  VLLM_ARGS=("${COMMON_ARGS[@]}"
    --enable-lora
    --max-loras 1
    --max-lora-rank 16
    --lora-modules "$LORA_NAME=$ADAPTER_DIR"
  )
else
  echo "==> mode: BASE only (no LoRA adapter)"
  VLLM_ARGS=("${COMMON_ARGS[@]}")
fi

echo "==> host: http://0.0.0.0:${PORT}"
echo "==> served model names: $BASE_MODEL$([ "$WITH_LORA" = 1 ] && echo " , $LORA_NAME")"
echo

CMD=("$VLLM_BIN" "${VLLM_ARGS[@]}")
printf '    %q ' "${CMD[@]}"; echo

if [ "$DRY_RUN" = 1 ]; then
  echo "(dry-run: not executing)"
  exit 0
fi

exec "${CMD[@]}"
