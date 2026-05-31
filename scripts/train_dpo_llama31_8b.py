#!/usr/bin/env python3
"""
DPO on Llama 3.1 8B with LoRA: maximize preference for Qwen-generated O2 over Llama 8B P1 baseline.

Expects a JSONL from prepare_dpo_qwen_vs_baseline.py with keys:
  prompt, chosen, rejected

Launch on 4 GPUs (example):
  torchrun --standalone --nproc_per_node=4 scripts/train_dpo_llama31_8b.py \\
    --dataset-jsonl data/dpo/qwen_chosen_vs_llama8b_rejected.jsonl \\
    --base-model $HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/... \\
    --output-dir models/llama31_dpo_qwen_vs_baseline

Requires: pip install trl datasets peft

Reference model: by default ``ref_model=None`` so TRL + LoRA does not load a second
full copy of the base weights (avoids OOM on multi-GPU DPO). Use ``--load-duplicate-ref-model``
only if you need the legacy two-model setup.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_dpo_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Bad JSON line {i} in {path}: {e}") from e
    return rows


def main() -> None:
    import traceback as _tb

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        print(
            "Missing dependency. Install with:\n"
            "  pip install trl datasets peft accelerate\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--dataset-jsonl", required=True, help="DPO pairs JSONL (prompt, chosen, rejected)")
    p.add_argument(
        "--base-model",
        default="$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        help="Local snapshot or HF id for Llama 3.1 8B Instruct",
    )
    p.add_argument("--output-dir", default="models/llama31_dpo_qwen_vs_baseline", help="Where to save adapter + tokenizer")
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-7, help="Typical DPO LR (small)")
    p.add_argument("--beta", type=float, default=0.1, help="DPO temperature / KL strength")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048, help="Max total sequence length for DPO (prompt+response)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--max-samples", type=int, default=0, help="If >0, train on first N rows only (debug)")
    p.add_argument("--bf16", action="store_true", help="Use bf16 if supported (recommended on A100/L40)")
    p.add_argument("--fp16", action="store_true", help="Use fp16 (use if bf16 unavailable)")
    p.add_argument(
        "--load-duplicate-ref-model",
        action="store_true",
        help="Load a separate frozen ref model (2x base VRAM per GPU; only if you know you need it)",
    )
    args = p.parse_args()

    ds_path = Path(args.dataset_jsonl)
    if not ds_path.is_file():
        raise SystemExit(f"Dataset not found: {ds_path}")

    rows = load_dpo_rows(ds_path)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    prompts, chosens, rejecteds = [], [], []
    for r in rows:
        pr = (r.get("prompt") or "").strip()
        ch = (r.get("chosen") or "").strip()
        rej = (r.get("rejected") or "").strip()
        if not pr or not ch or not rej:
            continue
        prompts.append(pr)
        chosens.append(ch)
        rejecteds.append(rej)

    if not prompts:
        raise SystemExit("No valid DPO rows after filtering.")

    # Keep ONLY the three columns DPOTrainer / DataCollatorForPreference expect.
    # Extra columns (p1, p2, category, p1_id) cause collator key errors in TRL >= 0.9.
    train_ds = Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})
    logger.info("Train rows: %d  columns: %s", len(train_ds), train_ds.column_names)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, legacy=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not torch.cuda.is_available():
        torch_dtype = torch.float32
    elif args.fp16:
        torch_dtype = torch.float16
    elif args.bf16 and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    else:
        # Default on GPU: fp16 if bf16 unavailable or not requested
        torch_dtype = torch.float16

    logger.info("Loading policy model (dtype=%s)...", torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch_dtype,          # `torch_dtype` was deprecated; `dtype` is the new kwarg
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # Build the LoRA config.  We pass it to DPOTrainer directly so that TRL
    # handles PEFT wrapping and the "ref" adapter (model.add_adapter("ref", ...))
    # internally with the matching PEFT version.  Pre-wrapping manually and then
    # passing an already-PeftModel can race with TRL's own add_adapter call.
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    ref_model = None
    if args.load_duplicate_ref_model:
        logger.info("Loading separate reference model (--load-duplicate-ref-model)...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch_dtype,
            trust_remote_code=True,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        ref_model.eval()
        for p_ in ref_model.parameters():
            p_.requires_grad = False

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dpo_args = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        bf16=(torch_dtype == torch.bfloat16),
        fp16=(torch_dtype == torch.float16),
        beta=args.beta,
        max_length=args.max_length,
        remove_unused_columns=False,
        report_to=[],
        gradient_checkpointing=True,   # already default=True in this TRL but explicit is fine
        optim="adamw_torch",
    )

    if ref_model is None:
        logger.info("DPO reference: TRL will use LoRA base weights via adapter disable (no extra copy in VRAM).")

    dpo_kw: dict = dict(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_ds,
        peft_config=peft_config,
    )
    try:
        logger.info("Constructing DPOTrainer (processing_class API)...")
        trainer = DPOTrainer(**dpo_kw, processing_class=tokenizer)
    except TypeError as e:
        logger.warning("processing_class kwarg rejected (%s); falling back to tokenizer=", e)
        trainer = DPOTrainer(**dpo_kw, tokenizer=tokenizer)
    logger.info("DPOTrainer constructed; trainable params: %s", sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))

    logger.info("Starting DPO training...")
    trainer.train()
    logger.info("Saving to %s", out_dir)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Done.")


if __name__ == "__main__":
    import traceback as _tb_main
    try:
        main()
    except Exception:
        # Print full traceback explicitly so it is not swallowed by torchrun's
        # multi-rank stderr interleaving.
        rank = int(os.environ.get("RANK", 0))
        print(f"\n[rank {rank}] === UNCAUGHT EXCEPTION ===", flush=True, file=sys.stderr)
        _tb_main.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)
