#!/usr/bin/env python3
"""
DPO on Llama-3.1-8B-Instruct starting from the SFT checkpoint, using LinguaSafe preference pairs.

Input: data/sft_dpo/dpo_train.jsonl  {prompt, chosen, rejected}
Output: output/DPO/{run_name}/model/

Usage:
  python scripts/train_dpo_linguasafe.py \
      --base-model output/SFT/run_001/model \
      --train-jsonl data/sft_dpo/dpo_train.jsonl \
      --val-jsonl   data/sft_dpo/dpo_val.jsonl \
      --output-dir  output/DPO/run_001/model
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as e:
        sys.exit(f"Missing dep: {e}. Run: pip install trl datasets peft accelerate")

    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model",    required=True,  help="SFT checkpoint (LoRA adapter dir or full model)")
    ap.add_argument("--train-jsonl",   default="data/sft_dpo/dpo_train.jsonl")
    ap.add_argument("--val-jsonl",     default="data/sft_dpo/dpo_val.jsonl")
    ap.add_argument("--output-dir",    default="output/DPO/run_001/model")
    ap.add_argument("--beta",          type=float, default=0.1,  help="KL penalty strength")
    ap.add_argument("--epochs",        type=float, default=1.0)
    ap.add_argument("--batch-size",    type=int,   default=2)
    ap.add_argument("--grad-accum",    type=int,   default=8)
    ap.add_argument("--lr",            type=float, default=5e-7)
    ap.add_argument("--max-length",    type=int,   default=2048)
    ap.add_argument("--max-prompt-length", type=int, default=512)
    ap.add_argument("--lora-r",        type=int,   default=16)
    ap.add_argument("--lora-alpha",    type=int,   default=32)
    ap.add_argument("--lora-dropout",  type=float, default=0.05)
    ap.add_argument("--lora-targets",  default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--logging-steps", type=int,   default=10)
    ap.add_argument("--eval-steps",    type=int,   default=50)
    ap.add_argument("--save-steps",    type=int,   default=50)
    ap.add_argument("--wandb-project", default="linguasafe-sft-dpo")
    ap.add_argument("--run-name",      default="dpo_linguasafe")
    ap.add_argument("--seed",          type=int,   default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print("=== DPO LinguaSafe ===")
    print(f"base:   {args.base_model}")
    print(f"train:  {args.train_jsonl}")
    print(f"val:    {args.val_jsonl}")
    print(f"output: {args.output_dir}")
    print(f"beta={args.beta}  epochs={args.epochs}  bs={args.batch_size}x{args.grad_accum}  lr={args.lr}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO needs left padding

    def format_dpo(row: dict) -> dict:
        prompt_msgs = [{"role": "user", "content": row["prompt"]}]
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt":   prompt_str,
            "chosen":   row["chosen"],
            "rejected": row["rejected"],
        }

    train_raw = load_jsonl(args.train_jsonl)
    val_raw   = load_jsonl(args.val_jsonl)
    train_ds  = Dataset.from_list([format_dpo(r) for r in train_raw])
    val_ds    = Dataset.from_list([format_dpo(r) for r in val_raw])
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type="sigmoid",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=args.logging_steps,
        report_to="wandb",
        run_name=args.run_name,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        seed=args.seed,
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # None + LoRA = implicit reference via adapter disable
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("\nStarting DPO training...")
    trainer.train()

    print("\nSaving model + tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
