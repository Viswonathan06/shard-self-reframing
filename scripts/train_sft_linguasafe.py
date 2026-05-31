#!/usr/bin/env python3
"""
SFT on Llama-3.1-8B-Instruct / Qwen3.5-9B using LinguaSafe teacher data.

Input:  output/SFT/.../sft_{non}reasoning/{train,val}.jsonl
Output: output/SFT/.../model/

Training:
  - Chat-template formatting via tokenizer.apply_chat_template
  - Prompt masking: loss only on assistant turn
  - LoRA (r=16, alpha=32) on q/k/v/o_proj + gate/up/down_proj
  - SafeTrainer: computes cross-entropy manually to avoid transformers 5.x bugs
  - W&B logging
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# gptqmodel's AWQ module is missing AwqGEMMQuantLinear in this build; patch
# peft.tuners.lora.model.dispatch_awq (the local binding used when building
# the dispatchers list) so it returns None instead of raising ImportError.
try:
    import peft.tuners.lora.model as _peft_lora_model
    _orig_dispatch_awq = _peft_lora_model.dispatch_awq
    def _safe_dispatch_awq(*args, **kwargs):
        try:
            return _orig_dispatch_awq(*args, **kwargs)
        except ImportError:
            return None
    _peft_lora_model.dispatch_awq = _safe_dispatch_awq
except Exception:
    pass


IGNORE_INDEX = -100


class SafeTrainer(Trainer):
    """Computes causal-LM loss manually, bypassing transformers 5.x num_items_in_batch bug.

    The model's own forward() in transformers 5.7 divides by num_items_in_batch
    which is 0 when all labels are -100, producing NaN → CUDA assert. We call
    forward() without labels and compute the loss ourselves with plain PyTorch
    CrossEntropyLoss, which handles -100 masking correctly.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        model_inputs = {k: v for k, v in inputs.items()
                        if k not in ("labels", "num_items_in_batch")}
        outputs = model(**model_inputs)
        logits = outputs.logits  # (B, T, V)

        # Causal LM shift: predict token t+1 from token t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().long()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return (loss, outputs) if return_outputs else loss


class SFTDataset(Dataset):
    """Tokenises (system + user + assistant) triples with prompt masking."""

    def __init__(self, path: str, tokenizer, max_length: int = 2048, base_model_path: str = ""):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._qwen3_no_think = False
        self.data: List[Dict] = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msgs = row.get("messages", [])
                if len(msgs) < 2:
                    continue
                self.data.append(row)

        print(f"  Loaded {len(self.data)} records from {path}")

    def __len__(self) -> int:
        return len(self.data)

    def _apply_template(self, msgs, add_generation_prompt: bool) -> list:
        kwargs: dict = dict(
            tokenize=True, add_generation_prompt=add_generation_prompt, return_tensors=None
        )
        if self._qwen3_no_think:
            try:
                kwargs["enable_thinking"] = False
            except Exception:
                pass
        result = self.tokenizer.apply_chat_template(msgs, **kwargs)
        if isinstance(result, list):
            return result
        ids = result["input_ids"]
        return ids.tolist() if hasattr(ids, "tolist") else list(ids)

    def __getitem__(self, idx: int) -> Dict:
        msgs = self.data[idx]["messages"]
        non_assistant = [m for m in msgs if m["role"] != "assistant"]

        prompt_ids = self._apply_template(non_assistant, add_generation_prompt=True)
        full_ids   = self._apply_template(msgs,          add_generation_prompt=False)

        eos = self.tokenizer.eos_token_id
        if eos is not None and (not full_ids or full_ids[-1] != eos):
            full_ids = full_ids + [eos]

        if len(full_ids) > self.max_length:
            full_ids = full_ids[: self.max_length]

        # Find the longest common prefix between prompt_ids and full_ids.
        # BPE tokenizers can merge the last template token(s) with the first
        # response token (e.g. "\n\n" + "I" → "\n\nI"), so prompt_ids is NOT
        # always an exact prefix of full_ids. Masking by len(prompt_ids) blindly
        # would mask into the response and leave loss near zero. Instead we mask
        # only the positions where the two sequences agree exactly, then start
        # computing loss from the first divergence point.
        mask_len = 0
        for i in range(min(len(prompt_ids), len(full_ids))):
            if full_ids[i] == prompt_ids[i]:
                mask_len = i + 1
            else:
                break

        labels = list(full_ids)
        for i in range(mask_len):
            labels[i] = IGNORE_INDEX

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model",    required=True)
    ap.add_argument("--train-jsonl",   default="data/sft_dpo/sft_train.jsonl")
    ap.add_argument("--val-jsonl",     default="data/sft_dpo/sft_val.jsonl")
    ap.add_argument("--output-dir",    default="output/SFT/run_001/model")
    ap.add_argument("--max-length",    type=int,   default=2048)
    ap.add_argument("--epochs",        type=int,   default=3)
    ap.add_argument("--batch-size",    type=int,   default=2)
    ap.add_argument("--grad-accum",    type=int,   default=8)
    ap.add_argument("--lr",            type=float, default=2e-5)
    ap.add_argument("--warmup-ratio",  type=float, default=0.05)
    ap.add_argument("--lora-r",        type=int,   default=16)
    ap.add_argument("--lora-alpha",    type=int,   default=32)
    ap.add_argument("--lora-dropout",  type=float, default=0.05)
    ap.add_argument("--lora-targets",  default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--logging-steps", type=int,   default=10)
    ap.add_argument("--save-steps",    type=int,   default=100)
    ap.add_argument("--wandb-project", default="linguasafe-sft-dpo")
    ap.add_argument("--run-name",      default="sft_linguasafe")
    ap.add_argument("--report-to",        default="wandb", help="'wandb', 'tensorboard', or 'none'")
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--fsdp",             default="", help="FSDP strategy, e.g. 'full_shard auto_wrap'")
    ap.add_argument("--fsdp-transformer-cls", default="", dest="fsdp_transformer_cls",
                    help="Transformer decoder layer class name for FSDP auto-wrap (e.g. Qwen3DecoderLayer)")
    ap.add_argument("--load-in-4bit",     action="store_true", dest="load_in_4bit",
                    help="QLoRA: load base model in 4-bit NF4 (requires bitsandbytes)")
    ap.add_argument("--fsdp-cpu-efficient", action="store_true", dest="fsdp_cpu_efficient",
                    help="FSDP: rank-0 loads model on CPU, other ranks use meta device (saves CPU RAM)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== SFT LinguaSafe ===")
    print(f"base:   {args.base_model}")
    print(f"train:  {args.train_jsonl}")
    print(f"val:    {args.val_jsonl}")
    print(f"output: {args.output_dir}")
    print(f"epochs={args.epochs}  bs={args.batch_size}x{args.grad_accum}  lr={args.lr}")

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    _local = os.path.isdir(args.base_model)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=_local
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map={"": local_rank},
            trust_remote_code=True,
            local_files_only=_local,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, lora)
    elif args.fsdp_cpu_efficient and local_rank != 0:
        # FSDP cpu-efficient: non-rank-0 creates model structure on meta device only.
        # FSDP sync_module_states will broadcast rank-0's weights before sharding.
        # LoRA adapters are also created inside the context so ALL parameters
        # (base + adapters) are on meta device — FSDP requires a consistent device
        # state across all params for sync_module_states to work correctly.
        from accelerate import init_empty_weights
        _cfg = AutoConfig.from_pretrained(
            args.base_model, trust_remote_code=True, local_files_only=_local
        )
        # VLMs (e.g. Qwen3.5) embed text_config inside the outer config; CausalLM
        # model classes expect the text-only sub-config, not the full VLM config.
        _causal_cfg = getattr(_cfg, "text_config", _cfg)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(_causal_cfg, trust_remote_code=True)
            model = get_peft_model(model, lora)
        # Cast to bfloat16 outside the context so the dtype is set on meta tensors
        # without init_empty_weights interfering with the .to() call.
        model = model.to(torch.bfloat16)
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        # Rank-0 (or non-FSDP): load model; for FSDP cpu-efficient, load onto CPU.
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu" if args.fsdp_cpu_efficient else None,
            trust_remote_code=True,
            local_files_only=_local,
        )
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model = get_peft_model(model, lora)

    if local_rank == 0:
        model.print_trainable_parameters()

    # FSDP + PEFT: update_fsdp_plugin_peft reads _no_split_modules to build the
    # wrap policy. VLMs list extra classes (e.g. Qwen3_5VisionBlock) that don't
    # exist in the text-only CausalLM, causing a lookup failure. Pin it to the
    # requested transformer class(es) so only present modules are listed.
    if bool(args.fsdp) and args.fsdp_transformer_cls:
        model._no_split_modules = [c.strip() for c in args.fsdp_transformer_cls.split(",")]

    train_ds = SFTDataset(args.train_jsonl, tokenizer, max_length=args.max_length, base_model_path=args.base_model)
    val_ds   = SFTDataset(args.val_jsonl,   tokenizer, max_length=args.max_length, base_model_path=args.base_model)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        return_tensors="pt",
    )

    use_fsdp = bool(args.fsdp)
    fsdp_config = {}
    if use_fsdp and args.fsdp_transformer_cls:
        fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = args.fsdp_transformer_cls
    if use_fsdp:
        fsdp_config["fsdp_state_dict_type"] = "FULL_STATE_DICT"
        fsdp_config["fsdp_activation_checkpointing"] = True
        if args.fsdp_cpu_efficient:
            fsdp_config["fsdp_cpu_ram_efficient_loading"] = True
            fsdp_config["fsdp_sync_module_states"] = True

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=not use_fsdp,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=not use_fsdp,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        fsdp=args.fsdp,
        fsdp_config=fsdp_config if fsdp_config else None,
    )

    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("\nStarting SFT training...")
    trainer.train()

    print("\nSaving model + tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")
    Path(args.output_dir, "final_eval.json").write_text(
        json.dumps(metrics, indent=2)
    )
    print(f"Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
