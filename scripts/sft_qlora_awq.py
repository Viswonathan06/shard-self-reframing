#!/usr/bin/env python3
"""
QLoRA-style SFT for AWQ pre-quantized models (e.g. Llama-3.3-70B-Instruct-AWQ-4bit).

AWQ models are already INT4 on disk — no bitsandbytes needed.
We load the frozen AWQ base and train only the LoRA adapters (bf16).

Memory budget per A6000 (49 GiB) for Llama-3.3-70B AWQ-4bit:
  AWQ INT4 weights: ~18 GiB  (denser packing than NF4)
  LoRA adapters + activations: ~8 GiB
  Total: ~26 GiB → fits comfortably, DDP across 4 GPUs.

Launch with torchrun:
  torchrun --nproc_per_node=4 scripts/sft_qlora_awq.py \
    --base-model /path/to/awq_model \
    --train-jsonl path/to/train.jsonl \
    --val-jsonl   path/to/val.jsonl \
    --output-dir  output/SFT/...
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
from peft import LoraConfig, get_peft_model

IGNORE_INDEX = -100

SYSTEM_PROMPT = (
    "You are a helpful and safe assistant. When a user asks about a sensitive topic, "
    "respond helpfully while avoiding harm. Always address the user's underlying need "
    "even if you cannot address the exact phrasing of their request."
)


class SafeTrainer(Trainer):
    """Cross-entropy computed outside the model forward so labels stay on the
    correct device regardless of how the model is sharded across GPUs."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        inputs.pop("num_items_in_batch", None)
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)

        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        logits = outputs.logits
        n_classes = logits.shape[-1]

        oob = (labels != IGNORE_INDEX) & ((labels < 0) | (labels >= n_classes))
        if oob.any():
            labels = labels.masked_fill(oob, IGNORE_INDEX)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)

        if (shift_labels == IGNORE_INDEX).all():
            loss = torch.tensor(0.0, requires_grad=True, device=shift_logits.device)
            return (loss, outputs) if return_outputs else loss

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, n_classes),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return (loss, outputs) if return_outputs else loss


class ChatSFTDataset(Dataset):
    """Accepts p1/o2 fields or pre-built messages list."""

    @staticmethod
    def _fold_system(msgs: List[Dict]) -> List[Dict]:
        result: List[Dict] = []
        pending_sys = ""
        for m in msgs:
            if m["role"] == "system":
                pending_sys = m["content"]
            elif m["role"] == "user" and pending_sys:
                result.append({"role": "user", "content": pending_sys + "\n\n" + m["content"]})
                pending_sys = ""
            else:
                result.append(m)
        return result

    def __init__(self, path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[Dict] = []

        self._supports_system = True
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            self._supports_system = False

        with open(path, encoding="utf-8") as f:
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
                        if self._supports_system:
                            msgs = [
                                {"role": "system",    "content": SYSTEM_PROMPT},
                                {"role": "user",      "content": p1},
                                {"role": "assistant", "content": o2},
                            ]
                        else:
                            msgs = [
                                {"role": "user",      "content": SYSTEM_PROMPT + "\n\n" + p1},
                                {"role": "assistant", "content": o2},
                            ]
                        row = dict(row, messages=msgs)
                elif not self._supports_system:
                    msgs = self._fold_system(msgs)
                    row = dict(row, messages=msgs)
                if not msgs or len(msgs) < 2:
                    continue
                self.data.append(row)

        print(f"Loaded {len(self.data)} records from {path} (system_role={'yes' if self._supports_system else 'folded into user'})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        msgs = self.data[idx]["messages"]

        prompt_str = self.tokenizer.apply_chat_template(
            [m for m in msgs if m["role"] != "assistant"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]

        full_str = self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_ids = self.tokenizer(full_str, add_special_tokens=False)["input_ids"]

        eos = self.tokenizer.eos_token_id
        if eos is not None and full_ids[-1] != eos:
            full_ids = full_ids + [eos]

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]
        if len(prompt_ids) > self.max_length:
            prompt_ids = prompt_ids[: self.max_length]

        labels = list(full_ids)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = IGNORE_INDEX

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model",            required=True)
    ap.add_argument("--train-jsonl",           required=True)
    ap.add_argument("--val-jsonl",             required=True)
    ap.add_argument("--output-dir",            required=True)
    ap.add_argument("--epochs",                type=int,   default=3)
    ap.add_argument("--per-device-batch-size", type=int,   default=1)
    ap.add_argument("--grad-accum",            type=int,   default=8)
    ap.add_argument("--learning-rate",         type=float, default=1e-4)
    ap.add_argument("--warmup-ratio",          type=float, default=0.05)
    ap.add_argument("--lora-r",                type=int,   default=16)
    ap.add_argument("--lora-alpha",            type=int,   default=32)
    ap.add_argument("--lora-targets",          default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--max-length",            type=int,   default=2048)
    ap.add_argument("--logging-steps",         type=int,   default=1)
    ap.add_argument("--seed",                  type=int,   default=42)
    ap.add_argument("--wandb-project",         default="linguasafe-sft-dpo")
    ap.add_argument("--run-name",              default="qlora_awq_sft")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["WANDB_PROJECT"] = args.wandb_project

    print(f"[rank {local_rank}] Loading AWQ base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # AWQ model is pre-quantized INT4 — load directly, no BitsAndBytesConfig needed.
    # device_map={"": local_rank} pins the full model to one GPU per DDP rank.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
    )

    # AWQ base layers are frozen INT4; only LoRA adapters are trained (bf16).
    # No prepare_model_for_kbit_training needed — that's bitsandbytes-specific.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_targets.split(","),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatSFTDataset(args.train_jsonl, tokenizer, args.max_length)
    val_dataset   = ChatSFTDataset(args.val_jsonl,   tokenizer, args.max_length)

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        fp16=False,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="wandb",
        run_name=args.run_name,
        optim="adamw_torch_fused",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print(f"[rank {local_rank}] Starting AWQ LoRA training...")
    trainer.train()

    if local_rank == 0:
        print("Saving LoRA adapter + tokenizer...")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"Done — adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
