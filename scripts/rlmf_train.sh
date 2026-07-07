#!/usr/bin/env bash
# =============================================================================
# rlmf_train.sh — one-stop entrypoint for the RLMF training pipeline
# =============================================================================
#
# The RLMF (Reinforcement Learning from Market Feedback) loop has TWO
# operational modes; this script drives both.
#
# 1. **Prompt-only surrogate (free, no GPU, runs every workflow).**
#    The desk's live PM task is prefixed with a small "case book" of
#    the top-K wins to emulate + bottom-K losses to avoid, drawn from
#    the resolved-episode store on disk.  This is what actually
#    influences live decisions — it's the FinGPT-inspired analog of
#    the RLHF exemplar pool.
#
#    Activation is env-gated:
#        TRADINGCREW_RLMF_ENABLED=1
#        TRADINGCREW_RLMF_K_WINS=3
#        TRADINGCREW_RLMF_K_LOSSES=2
#
#    You do NOT need this script for that path — the prefix is
#    rebuilt in-process on every workflow.  What this script gives
#    you is:
#      * the JSONL export so you can inspect / diff / version the
#        dataset the LLM will implicitly consume, and
#      * a summary of the reward distribution so you can decide
#        whether the case book has enough coverage to be useful.
#
# 2. **LoRA fine-tune (offline, one GPU, occasional).**
#    The exported JSONL is shaped like the FinGPT-Forecaster training
#    file on Hugging Face
#    (``ai4finance-foundation/fingpt-forecaster-dow30-202305-202405``),
#    so you can drop it into the standard FinGPT-Forecaster notebook
#    or run the ``run_finetune`` step this script wraps.  The output
#    is a LoRA adapter that can be merged back into the base LLM
#    served by vLLM.
#
# Usage
# -----
#   scripts/rlmf_train.sh                # export + summary (safe default)
#   scripts/rlmf_train.sh --few-shot     # print the prompt-only prefix
#   scripts/rlmf_train.sh --symbol AAPL  # restrict to one ticker
#   scripts/rlmf_train.sh --horizon LONG # restrict to LONG-tier episodes
#   scripts/rlmf_train.sh --lora         # kick off the FinGPT LoRA fine-tune
#                                        # (requires torch + peft + a GPU)
#   scripts/rlmf_train.sh --help         # show underlying CLI help
#
# Exit codes
# ----------
#   0   ran successfully
#   1   no resolved episodes on disk yet — expected on a fresh install;
#       run the workflow once with M3 outcome resolution enabled and try
#       again after the first horizon window closes
#   2   the LoRA path was requested but the toolchain is missing
# =============================================================================

set -euo pipefail

# Resolve the repo root regardless of where the script is invoked from
# so the caller can `bash scripts/rlmf_train.sh` OR
# `cd scripts && ./rlmf_train.sh` and get the same behaviour.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
cd "$repo_root"

# Pick up the venv Python if one is present.  Falls back to the
# ambient `python` so this script also works inside the container image
# / CI where the venv is baked into the base layer.
if [[ -x "$repo_root/../.venv/bin/python" ]]; then
    PY="$repo_root/../.venv/bin/python"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
    PY="$repo_root/.venv/bin/python"
else
    PY="${PYTHON:-python}"
fi

# --- Argument parsing -------------------------------------------------------
mode="dataset"           # dataset | few-shot | lora | help
symbol=""
horizon=""
out_path=""
k_wins=3
k_losses=2
base_model=""            # empty => rlmf_lora.py's own default
epochs=""
lora_extra=()            # passthrough for less-common lora flags

while [[ $# -gt 0 ]]; do
    case "$1" in
        --few-shot)
            mode="few-shot"
            shift
            ;;
        --lora)
            mode="lora"
            shift
            ;;
        --symbol)
            symbol="$2"
            shift 2
            ;;
        --horizon)
            horizon="$2"
            shift 2
            ;;
        --path)
            out_path="$2"
            shift 2
            ;;
        --k-wins)
            k_wins="$2"
            shift 2
            ;;
        --k-losses)
            k_losses="$2"
            shift 2
            ;;
        --base-model)
            base_model="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        -h|--help)
            mode="help"
            shift
            ;;
        --)
            # Everything after -- is passed straight through to rlmf_lora
            shift
            lora_extra+=("$@")
            break
            ;;
        *)
            echo "unknown flag: $1" >&2
            mode="help"
            shift
            ;;
    esac
done

# Build the shared filter args (symbol + horizon).
common_args=()
[[ -n "$symbol" ]] && common_args+=(--symbol "$symbol")
[[ -n "$horizon" ]] && common_args+=(--horizon "$horizon")

case "$mode" in
    help)
        "$PY" -m trading_crew.agentic.rlmf --help || true
        cat <<'EOF'

--- rlmf_train.sh extras ---
  --few-shot          print the prompt-only prefix the PM would see
  --lora              run the FinGPT-Forecaster LoRA fine-tune (needs GPU)

Env vars honoured:
  TRADINGCREW_CACHE_DIR     where dataset.jsonl lands (default ~/.trading_crew)
  TRADINGCREW_RLMF_ENABLED  gate the prompt-only prefix at PM task build
  TRADINGCREW_RLMF_K_WINS   # of wins in the live PM prefix
  TRADINGCREW_RLMF_K_LOSSES # of losses in the live PM prefix
EOF
        exit 0
        ;;

    dataset)
        # 1) Export the JSONL that the FinGPT-Forecaster training
        #    notebook can ingest directly.
        export_args=("$PY" -m trading_crew.agentic.rlmf --export)
        [[ -n "$out_path" ]] && export_args+=(--path "$out_path")
        export_args+=("${common_args[@]}")
        echo "==> Exporting FinGPT-compatible JSONL"
        "${export_args[@]}"

        # 2) Print the summary so a human can spot "no resolved episodes yet"
        #    or a lopsided distribution immediately.
        summary_args=("$PY" -m trading_crew.agentic.rlmf --summary "${common_args[@]}")
        echo
        echo "==> Reward-distribution summary"
        "${summary_args[@]}"
        exit 0
        ;;

    few-shot)
        few_args=(
            "$PY" -m trading_crew.agentic.rlmf --few-shot
            --k-wins "$k_wins" --k-losses "$k_losses"
        )
        few_args+=("${common_args[@]}")
        echo "==> Prompt-only RLMF prefix (what the PM task would see)"
        "${few_args[@]}"
        exit 0
        ;;

    lora)
        echo "==> LoRA fine-tune (FinGPT-Forecaster recipe)"
        # LoRA needs torch + peft + accelerate + a JSONL to train on.
        # We fail loudly (rather than pip-installing behind the user's
        # back) — the user rule forbids fallback / hidden installs.
        if ! "$PY" -c "import torch, peft, transformers, accelerate" 2>/dev/null; then
            cat <<'EOF' >&2

The LoRA path needs: torch, transformers, peft, accelerate, datasets.
Install them into your venv (this is a heavy set — expect a few GB and
a compatible CUDA build) with:

    pip install torch transformers peft accelerate datasets bitsandbytes

then re-run `scripts/rlmf_train.sh --lora`.  We do not install these
automatically because they materially change the container image
size / GPU driver requirements.

EOF
            exit 2
        fi

        # Ensure the dataset exists first.
        target="${out_path:-${TRADINGCREW_CACHE_DIR:-$HOME/.trading_crew}/rlmf/dataset.jsonl}"
        if [[ ! -s "$target" ]]; then
            echo "no dataset at $target — running --export first" >&2
            "$PY" -m trading_crew.agentic.rlmf --export "${common_args[@]}"
        fi
        n_rows=$(wc -l <"$target" || echo 0)
        if [[ "$n_rows" -lt 20 ]]; then
            echo "dataset has $n_rows rows — LoRA fine-tune needs >= 20 to be worthwhile" >&2
            echo "run more workflows and let outcome resolution catch up, then retry" >&2
            exit 1
        fi

        # Delegate to the FinGPT-Forecaster training entrypoint.
        # ``trading_crew.agentic.rlmf_lora`` is a thin wrapper around
        # the FinGPT recipe so the user can also import it
        # programmatically (e.g. from a notebook).
        lora_args=(
            "$PY" -m trading_crew.agentic.rlmf_lora
            --dataset "$target"
            --output-dir "${LORA_OUTPUT_DIR:-${TRADINGCREW_CACHE_DIR:-$HOME/.trading_crew}/rlmf/lora_adapter}"
        )
        [[ -n "$base_model" ]] && lora_args+=(--base-model "$base_model")
        [[ -n "$epochs" ]]     && lora_args+=(--epochs "$epochs")
        lora_args+=("${lora_extra[@]}")
        echo "==> Delegating to trading_crew.agentic.rlmf_lora"
        echo "    ${lora_args[*]}"
        exec "${lora_args[@]}"
        ;;
esac
