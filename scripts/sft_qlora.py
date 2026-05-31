#!/usr/bin/env python3
"""
QLoRA SFT for large models (27B–70B) that don't fit in bf16 on 8x A6000 GPUs.

Uses 4-bit NF4 quantization (bitsandbytes) so each GPU holds a full copy of the
quantized model, enabling standard DDP data parallelism without DeepSpeed.

Memory budget per A6000 (49 GiB):
  Qwen3.5-27B   4-bit ~14 GiB + LoRA + activations → ~20 GiB
  Mistral-24B   4-bit ~12 GiB + LoRA + activations → ~18 GiB
  LLaMA-3.1-70B 4-bit ~35 GiB + LoRA + activations → ~40 GiB

Launch with torchrun (no DeepSpeed):
  torchrun --nproc_per_node=8 scripts/sft_qlora.py \
    --base-model /path/to/model \
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
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    """Same dataset format as sft_v2.py — accepts p1/o2 fields or messages list."""

    @staticmethod
    def _fold_system(msgs: List[Dict]) -> List[Dict]:
        """Fold any system message into the first user message (for tokenizers that reject system role)."""
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

        # Detect once whether this tokenizer accepts a system role (mistral_common rejects it).
        # Must use tokenize=True — mistral_common bypasses validation with tokenize=False.
        self._supports_system = True
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
                tokenize=True, add_generation_prompt=True,
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
        assistant_content = next(m["content"] for m in msgs if m["role"] == "assistant")

        # Use return_tensors="pt" so apply_chat_template always returns a BatchEncoding;
        # then extract .input_ids[0].tolist() to get a plain Python list regardless of
        # whether the backend is mistral_common or standard HF.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.tokenizer.apply_chat_template(
                [m for m in msgs if m["role"] != "assistant"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        prompt_ids = (out.input_ids if hasattr(out, "input_ids") else out)[0].tolist()

        response_ids = self.tokenizer(assistant_content, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + response_ids

        eos = self.tokenizer.eos_token_id
        if eos is not None and full_ids[-1] != eos:
            full_ids = full_ids + [eos]

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(n_prompt):
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
    ap.add_argument("--run-name",              default="qlora_sft")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["WANDB_PROJECT"] = args.wandb_project

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Mistral3 (e.g. Mistral-Small-3.2-24B) ships as ForConditionalGeneration
    # but is a causal LM; register it so AutoModelForCausalLM resolves it.
    try:
        from transformers.models.mistral3 import Mistral3Config, Mistral3ForConditionalGeneration
        AutoModelForCausalLM.register(Mistral3Config, Mistral3ForConditionalGeneration)
    except ImportError:
        pass

    print(f"[rank {local_rank}] Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True,
    )

    # Enable gradient checkpointing before kbit prep to save activation memory
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

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
        optim="paged_adamw_32bit",
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

    print(f"[rank {local_rank}] Starting QLoRA training...")
    trainer.train()

    if local_rank == 0:
        print("Saving best model + tokenizer...")
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        print(f"Done — adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
