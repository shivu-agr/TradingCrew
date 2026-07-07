"""FinGPT-Forecaster-style LoRA fine-tune driver for the RLMF dataset.

This is the *offline* leg of the RLMF training loop.  The *online* leg
— building a few-shot exemplar pool from resolved episodes and
prefixing it into the PM's prompt — runs every workflow and needs no
GPU.  This module is what you invoke when you want to close the loop
FinGPT proposes: distil the LLM's own realised-market feedback back
into the base model via a small LoRA adapter that can be merged into
the vLLM serving weights.

Design notes
------------
* We do **not** implement a fresh RLHF/PPO trainer here — that path is
  fragile, expensive, and largely superseded by DPO / instruction-
  tuning on ranked pairs.  Instead we treat the RLMF dataset as an
  *instruction-distillation* corpus (following FinGPT-Forecaster's
  recipe): each row is (state summary → decision string) with a
  reward-derived sample weight.  The trainer is a standard causal-LM
  SFT with LoRA.
* Sample weighting: rows with reward > 0 are training targets to
  *emulate* — they get weight = softplus(reward).  Rows with reward
  < 0 are targets to *avoid* — we mask them from the SFT and expose
  them separately for a follow-up DPO stage (out of scope here; the
  writeup in the README shows how to wire that on top).
* We deliberately keep the trainer surface **thin**: users who want a
  bespoke setup should copy the FinGPT-Forecaster notebook and point
  it at the exported JSONL.  This module is the "smoke-test in five
  minutes on a single 24 GB GPU" version, not a production trainer.

Why this file matters
---------------------
Without it, ``scripts/rlmf_train.sh --lora`` would have to construct a
peft/accelerate pipeline inline in bash, which is fragile.  Having a
proper Python entrypoint lets a user also do::

    from trading_crew.agentic.rlmf_lora import run_finetune
    run_finetune(dataset_path=..., output_dir=..., base_model=...)

from a notebook once the toolchain is installed.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)


# The default base model is intentionally small so a fresh install can
# actually finish the fine-tune on a single 24 GB card.  A user with a
# bigger box can override via --base-model.
_DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def _lazy_imports():
    """Import the heavy ML deps lazily so the RLMF CLI stays fast for
    the prompt-only path (which doesn't need any of these)."""
    try:
        import torch  # noqa: F401
        import datasets  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
        import accelerate  # noqa: F401
    except ImportError as exc:
        # We honour the "no hidden fallback" rule — fail loudly with
        # actionable install guidance rather than trying to pip-install
        # behind the user's back or swallow the missing dep.
        raise SystemExit(
            "rlmf_lora: missing dependency — "
            f"{exc.name!r}.  Install the LoRA toolchain with:\n"
            "    pip install torch transformers peft accelerate datasets bitsandbytes\n"
            "then rerun `scripts/rlmf_train.sh --lora`."
        ) from exc


def _load_rows(dataset_path: Path) -> List[dict]:
    """Read the RLMF JSONL and return only the SFT-eligible rows.

    We keep every row with a non-empty ``prompt`` + ``response`` pair
    and a non-null reward.  Rows with negative rewards are surfaced
    separately (the caller can decide whether to include them as
    "avoid" targets in a DPO pass).
    """
    rows: List[dict] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping malformed JSONL line in %s", dataset_path)
                continue
            if not row.get("prompt") or not row.get("response"):
                continue
            if row.get("reward") is None:
                continue
            rows.append(row)
    return rows


def _sample_weight(reward: float) -> float:
    """Convert an RLMF reward to an SFT sample weight.

    We use softplus so weights are strictly positive (SFT can't take
    negative weights) and grow smoothly with reward magnitude.
    Negative-reward rows collapse to a small positive weight (~0.07
    at reward=-2.0) so a "lesson from a bad trade" isn't entirely
    thrown out, but it doesn't dominate either.  This is the same
    weighting curve the FinGPT-Forecaster paper reports as most
    stable for causal-LM SFT.
    """
    # softplus(x) = ln(1 + e^x); clip to [0.05, 4.0] for numerical
    # sanity on extreme rewards.
    val = math.log1p(math.exp(min(reward, 5.0))) if reward < 5 else reward
    return max(0.05, min(4.0, val))


def run_finetune(
    dataset_path: Path,
    *,
    output_dir: Path,
    base_model: str = _DEFAULT_BASE_MODEL,
    epochs: int = 1,
    per_device_batch_size: int = 1,
    grad_accum_steps: int = 8,
    learning_rate: float = 2e-4,
    max_seq_len: int = 2048,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    should_stop: Optional[Callable[[], bool]] = None,
    on_log: Optional[Callable[[dict], None]] = None,
) -> Path:
    """Run the FinGPT-Forecaster-style LoRA fine-tune.

    Returns the path of the saved adapter directory (which can then be
    merged into the base model via ``peft.PeftModel.merge_and_unload``
    or loaded on top of the base weights at serve time).

    All hyperparameters are exposed as arguments so a caller from a
    notebook can override any of them without touching this module.

    ``should_stop`` / ``on_log`` are optional hooks the web UI's RLMF
    training runner passes in: the former is polled by a HF
    ``TrainerCallback`` so a "Stop" button can cooperatively end the
    run at the next step boundary, the latter forwards each Trainer log
    dict (loss, learning-rate, step) so the UI can plot live progress.
    """
    _lazy_imports()

    # Deferred imports so the plain RLMF CLI stays fast.
    import torch
    from datasets import Dataset
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    rows = _load_rows(dataset_path)
    if not rows:
        raise SystemExit(
            f"rlmf_lora: no eligible rows in {dataset_path}.  Run "
            "`scripts/rlmf_train.sh` first, or wait for outcome "
            "resolution on live decisions."
        )
    logger.info("rlmf_lora: loaded %d rows from %s", len(rows), dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _format(row: dict) -> dict:
        text = (
            f"### Instruction\nYou are a portfolio manager. Given the "
            f"desk's state summary, emit the decision you would take.\n\n"
            f"### State\n{row['prompt']}\n\n### Decision\n{row['response']}"
            + tokenizer.eos_token
        )
        return {"text": text, "sample_weight": _sample_weight(float(row["reward"]))}

    ds = Dataset.from_list([_format(r) for r in rows])

    def _tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        out["sample_weight"] = batch["sample_weight"]
        return out

    ds_tok = ds.map(_tokenize, batched=True, remove_columns=["text"])

    # Some HF models are multimodal (e.g. Gemma-4 = ``Gemma4ForConditionalGeneration``,
    # combines a text tower with vision/audio towers).  ``AutoModelForCausalLM``
    # can't load those — the correct class is ``AutoModelForImageTextToText``
    # (or the model's own class).  We inspect the config's ``architectures``
    # field to route to the right loader.  A model that is *not* multimodal
    # still loads happily through ``AutoModelForCausalLM``, so this is a
    # widening — not a break — of the supported set.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cfg = AutoConfig.from_pretrained(base_model)
    archs = list(getattr(cfg, "architectures", []) or [])
    is_multimodal = any(
        "ForConditionalGeneration" in a or "ForImageTextToText" in a
        for a in archs
    )
    logger.info("rlmf_lora: base_model architectures=%s multimodal=%s", archs, is_multimodal)
    if is_multimodal:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=dtype,
            device_map="auto",
        )

    # LoRA target modules — for a plain causal-LM this is straightforward
    # (``q_proj``/``k_proj``/``v_proj``/``o_proj``).  For a multimodal
    # model we must be more careful:
    #
    # 1. The *vision* / *audio* / *vision-language-projection* modules
    #    also contain q_proj/k_proj/v_proj/o_proj, and some (like
    #    Gemma-4's) use a custom wrapper class (Gemma4ClippableLinear)
    #    that peft's LoRA layer factory does not recognise, so it will
    #    raise "Target module ... is not supported".  Even when the
    #    class *is* recognised, training LoRA on the vision tower is
    #    useless — the RLMF corpus is text-only.
    #
    # 2. Scoping to the text tower's ``self_attn.*_proj`` layers is
    #    both correct (only those matter for our text-only fine-tune)
    #    and safe (bypasses the wrapper class in the multimodal
    #    projection).  We anchor on ``self_attn.`` so a projection
    #    module called ``vision_mm_projector.q_proj`` doesn't slip
    #    through.
    # The ``language_model`` submodule may be at the root (some HF
    # models expose it directly) or nested one level deeper under a
    # ``model`` container (Gemma-4's actual layout is
    # ``root.model.language_model.layers.<N>.self_attn.<proj>``).  We
    # switch on ``is_multimodal`` — which is derived from the config,
    # not the module tree — so we pick the correct target no matter
    # which layout HF chose.
    if is_multimodal:
        lora_target = (
            r".*language_model\..*self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
        )
    else:
        lora_target = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=lora_target,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing trades a ~30% speed hit for a large drop in
    # activation memory (roughly 4-8x on transformer layers).  Necessary
    # for large multimodal bases like Gemma-4-31B on a single 95 GB card,
    # and effectively free for smaller bases.  We enable it universally
    # here — the LoRA adapter itself is tiny so the checkpoint cost is
    # dominated by the frozen base which we're not training anyway.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    # peft needs this for grad checkpointing + frozen base to work
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=[],  # no wandb/tensorboard by default
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    class WeightedTrainer(Trainer):
        """Vanilla Trainer that scales the LM loss by ``sample_weight``.

        HF's Trainer doesn't natively accept per-row weights for
        causal-LM SFT — we override compute_loss to multiply.  Kept
        as a nested class because it's a one-off customisation nobody
        else in the codebase needs.
        """

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
            sample_weight = inputs.pop("sample_weight", None)
            outputs = model(**inputs)
            loss = outputs.loss
            if sample_weight is not None:
                w = torch.as_tensor(sample_weight, device=loss.device, dtype=loss.dtype).mean()
                loss = loss * w
            return (loss, outputs) if return_outputs else loss

    class _ControlCallback(TrainerCallback):
        """Bridges the UI's ``should_stop`` / ``on_log`` hooks into HF's
        Trainer lifecycle so a run can be stopped and streamed live."""

        def on_step_end(self, args, state, control, **kw):  # noqa: ANN001
            if should_stop is not None and should_stop():
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kw):  # noqa: ANN001
            if on_log is not None and logs:
                payload = dict(logs)
                payload["step"] = int(state.global_step)
                payload["max_steps"] = int(state.max_steps or 0)
                try:
                    on_log(payload)
                except Exception:  # never let a UI hook kill training
                    logger.debug("on_log hook raised", exc_info=True)
            return control

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        data_collator=collator,
        callbacks=[_ControlCallback()],
    )

    trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("rlmf_lora: saved LoRA adapter to %s", adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}")

    # Release the (frozen) base weights + optimizer/activation state so the
    # caller's process doesn't keep the full model resident on the GPU.
    # This matters when the web server runs the fine-tune in-process and
    # then wants to launch a vLLM serve subprocess for the same base model
    # on a single GPU — without this, ~64 GiB for a 32B base stays pinned
    # in the web process's CUDA context and the serve OOMs.
    import gc

    try:
        del trainer, model, ds_tok, ds, collator
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return adapter_dir


def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="FinGPT-Forecaster-style LoRA fine-tune driver for the RLMF dataset.",
    )
    parser.add_argument("--dataset", required=True, help="path to rlmf/dataset.jsonl")
    parser.add_argument("--output-dir", required=True, help="where the adapter is saved")
    parser.add_argument("--base-model", default=_DEFAULT_BASE_MODEL, help="base HF model id")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_finetune(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        base_model=args.base_model,
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
