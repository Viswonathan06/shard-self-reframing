#!/usr/bin/env python3
"""
v2 SFT for Llama-3.1-8B-Instruct on the DNA scratchpad dataset.

Differences vs scripts/sft_final.py (v1):
  * Uses Llama-3.1's NATIVE chat template via tokenizer.apply_chat_template()
    instead of the custom "### Human: / ### Assistant:" format.
  * PROMPT MASKING: labels are -100 on the prompt prefix and on padding, so
    loss is computed only on the assistant turn (the actual response we want
    to learn).
  * DYNAMIC PADDING via DataCollatorForSeq2Seq instead of pad-to-max-length.
    Prevents wasting gradient on long pad sequences.
  * Bigger LoRA: r=16 / alpha=32 on q,k,v,o_proj (v1 was r=4 on q,v).
  * Cosine LR schedule with warmup, eval on val set every epoch.
  * Reads stratified v2 dataset (data/sft_scratchpad_v2/{train,val}_sft.jsonl)
    where each row has a `messages` field.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.integrations import HfDeepSpeedConfig
from peft import LoraConfig, get_peft_model


class SafeTrainer(Trainer):
    """Compute loss externally so labels always reach the correct device.

    With device_map="auto", the model is pipelined across GPUs: logits land on
    whichever GPU hosts the lm_head while labels stay on GPU 0. The model's
    built-in CrossEntropyLoss then sees a device mismatch and the CUDA kernel
    reads garbage, triggering the nll_loss assertion. By popping labels before
    the forward and recomputing the shift+cross-entropy ourselves, we guarantee
    alignment regardless of how many GPUs the model spans — so this works for
    8B on 1 GPU and 70B on 8 GPUs without any change.

    Also handles:
      - OOB label IDs (Qwen3.5 EOS/PAD tokens beyond the lm_head output dim)
      - Fully-masked batches (all -100) → return zero loss instead of NaN
      - num_items_in_batch kwarg from transformers 5.x
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs.pop("num_items_in_batch", None)
        # Pop labels so the model skips its internal loss path and returns logits only.
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)

        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        logits = outputs.logits          # (B, T, V) — on whichever GPU hosts lm_head
        n_classes = logits.shape[-1]     # use actual output dim, not config.vocab_size

        # Clamp any label IDs that would be OOB for the lm_head (e.g. Qwen3.5 specials).
        oob = (labels != IGNORE_INDEX) & ((labels < 0) | (labels >= n_classes))
        if oob.any():
            labels = labels.masked_fill(oob, IGNORE_INDEX)

        # Standard causal-LM shift: predict token i+1 from position i.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Move labels to wherever logits live — this is the fix for device_map="auto".
        shift_labels = shift_labels.to(shift_logits.device)

        if (shift_labels == IGNORE_INDEX).all():
            loss = torch.tensor(0.0, requires_grad=True, device=shift_logits.device)
            return (loss, outputs) if return_outputs else loss

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, n_classes),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return (loss, outputs) if return_outputs else loss

IGNORE_INDEX = -100  # standard HF label-mask value

SYSTEM_PROMPT = (
    "You are a helpful and safe assistant. When a user asks about a sensitive topic, "
    "respond helpfully while avoiding harm. Always address the user's underlying need "
    "even if you cannot address the exact phrasing of their request."
)


class ChatSFTDataset(Dataset):
    """Pre-tokenised dataset that supports prompt masking under any chat template.

    For each record we:
      1. Render the prompt (only the user turn) with `add_generation_prompt=True`.
      2. Render the full conversation (user + assistant).
      3. Tokenise both. Use the prompt length P to set labels[:P] = -100,
         so only the assistant tokens contribute to the loss.
      4. Truncate from the right to `max_length` (assistant content is preferred
         over the prompt, so we keep the prompt and truncate the assistant
         response only when needed).

    Accepts two record formats:
      - `messages` list: [{"role": ..., "content": ...}, ...]
      - Raw fields: `p1` (user prompt) + `o2` (assistant response); a system
        message is injected automatically using SYSTEM_PROMPT.
    """

    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[Dict] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msgs = row.get("messages")
                if not msgs:
                    p1 = (row.get("p1") or "").strip()
                    o2 = (row.get("o2") or "").strip()
                    if p1 and o2:
                        msgs = [
                            {"role": "system",    "content": SYSTEM_PROMPT},
                            {"role": "user",      "content": p1},
                            {"role": "assistant", "content": o2},
                        ]
                        row = dict(row, messages=msgs)
                if not msgs or len(msgs) < 2:
                    continue
                self.data.append(row)

        print(f"Loaded {len(self.data)} records from {path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        msgs = self.data[idx]["messages"]
        # 1) prompt-only string + length
        prompt_str = self.tokenizer.apply_chat_template(
            [m for m in msgs if m["role"] != "assistant"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        # 2) full string (user + assistant)
        full_str = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = self.tokenizer(full_str, add_special_tokens=False)["input_ids"]

        # Append eos if the template didn't end the assistant turn with one
        eos = self.tokenizer.eos_token_id
        if eos is not None and full_ids[-1] != eos:
            full_ids = full_ids + [eos]

        # Truncate to max_length (right truncation: keep the prompt, drop tail)
        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]
        if len(prompt_ids) > self.max_length:
            prompt_ids = prompt_ids[: self.max_length]

        # 3) prompt masking
        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = IGNORE_INDEX

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-model",
        default="/playpen/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    )
    ap.add_argument("--train-jsonl", default="data/sft_scratchpad_v2/train_sft.jsonl")
    ap.add_argument("--val-jsonl", default="data/sft_scratchpad_v2/val_sft.jsonl")
    ap.add_argument("--output-dir", default="models/llama31_sft_v2")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per-device-batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)  # effective batch = 16
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora-targets",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names to attach LoRA to",
    )
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="linguasafe-sft-dpo")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--deepspeed", default=None, help="Path to DeepSpeed config JSON")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    print(f"=== SFT v2 ===")
    print(f"base: {args.base_model}")
    print(f"train: {args.train_jsonl}")
    print(f"val:   {args.val_jsonl}")
    print(f"out:   {args.output_dir}")
    print(
        f"epochs={args.epochs}  bs={args.per_device_batch_size}x{args.grad_accum}  "
        f"lr={args.learning_rate}  warmup={args.warmup_ratio}  max_len={args.max_length}"
    )
    print(
        f"LoRA: r={args.lora_r} alpha={args.lora_alpha} drop={args.lora_dropout} "
        f"targets={args.lora_targets}"
    )

    # HfDeepSpeedConfig must be alive before from_pretrained so that ZeRO-3's
    # deepspeed.zero.Init() context is active during weight loading. Without this,
    # the full model is loaded on every rank and ZeRO-3 can't repartition it in
    # time, causing each GPU to hold the complete parameter set (OOM for 27B+).
    #
    # deepspeed.zero.Init() calls DeepSpeedConfig.__init__ which runs
    # _configure_train_batch_size() — this asserts train_batch > 0 and fails
    # on "auto" strings. Resolve those values here using the known training args
    # before creating HfDeepSpeedConfig.
    _hf_ds_cfg = None
    if args.deepspeed:
        with open(args.deepspeed) as f:
            _ds_cfg_dict = json.load(f)
        if _ds_cfg_dict.get("zero_optimization", {}).get("stage", 0) == 3:
            import copy
            _init_cfg = copy.deepcopy(_ds_cfg_dict)
            mbs = args.per_device_batch_size
            gas = args.grad_accum
            _init_cfg["train_micro_batch_size_per_gpu"] = mbs
            _init_cfg["gradient_accumulation_steps"] = gas
            # zero.Init() runs before deepspeed.init_distributed(), so
            # get_world_size()=1 at this point — multiply by 1, not WORLD_SIZE.
            _init_cfg["train_batch_size"] = mbs * gas
            _init_cfg["bf16"] = {"enabled": True}
            _hf_ds_cfg = HfDeepSpeedConfig(_init_cfg)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right padding for causal LM training; left would mis-align labels.
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # Required so the input embeddings track grad through the LoRA path.
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_ds = ChatSFTDataset(args.train_jsonl, tokenizer, max_length=args.max_length)
    val_ds = ChatSFTDataset(args.val_jsonl, tokenizer, max_length=args.max_length)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        return_tensors="pt",
    )

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_steps=0,  # overridden below after dataset is loaded
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=args.run_name,
        # ZeRO-3 gathers weights during forward then immediately shards them back;
        # gradient checkpointing's recompute pass finds them already gone (shape [0]).
        # Disable gradient checkpointing under ZeRO-3 — param sharding already
        # reduces the per-GPU footprint enough that it isn't needed.
        gradient_checkpointing=_hf_ds_cfg is None,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        remove_unused_columns=False,
        deepspeed=args.deepspeed,
    )

    total_steps = (len(train_ds) // (args.per_device_batch_size * args.grad_accum)) * args.epochs
    targs.warmup_steps = max(1, round(args.warmup_ratio * total_steps))
    print(f"warmup_steps={targs.warmup_steps} (ratio={args.warmup_ratio}, total_steps={total_steps})")

    trainer = SafeTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("\nStarting training...")
    trainer.train()
    print("\nSaving best model + tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")
    Path(args.output_dir, "final_eval.json").write_text(json.dumps(metrics, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
