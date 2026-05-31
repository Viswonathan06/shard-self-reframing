#!/usr/bin/env python3
"""
Simplified SFT training for Llama 3.1 8B with scratchpad reasoning.
Focuses on compatibility and basic functionality.
"""

import json
import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, HfArgumentParser,
        Trainer
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@dataclass
class ScriptArguments:
    """Arguments for the SFT script."""
    
    # Model arguments - using local path
    model_name: str = field(default="/playpen/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659")
    
    # Data arguments  
    train_data: str = field(default="data/sft_scratchpad/train_sft_sharegpt.jsonl")
    val_data: str = field(default="data/sft_scratchpad/val_sft_sharegpt.jsonl")
    
    # Training arguments
    output_dir: str = field(default="models/llama31_8b_scratchpad_sft")
    num_train_epochs: int = field(default=1)  # Start with 1 epoch for testing
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=2e-4)
    max_seq_length: int = field(default=2048)  # Reduced for memory
    
    # LoRA arguments
    use_lora: bool = field(default=True)
    lora_r: int = field(default=8)  # Smaller for compatibility
    lora_alpha: int = field(default=16) 
    lora_dropout: float = field(default=0.05)
    
    # Other
    logging_steps: int = field(default=10)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=50)
    warmup_steps: int = field(default=20)


class SFTDataset(torch.utils.data.Dataset):
    """Simple dataset for SFT training."""
    
    def __init__(self, file_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(file_path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                
                # Convert ShareGPT to training format
                text = ""
                for msg in record['conversations']:
                    if msg['from'] == 'human':
                        text += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['value']}<|eot_id|>"
                    elif msg['from'] == 'gpt':
                        text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['value']}<|eot_id|>"
                
                self.data.append(text)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # For causal LM, input_ids = labels
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


def setup_model_and_tokenizer(args):
    """Set up the model and tokenizer."""
    
    print(f"📦 Loading model from: {args.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
        # Removed local_files_only for compatibility
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with minimal configuration - NO local_files_only to avoid path issues
    print(f"🔍 Attempting to load model from: {args.model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16,  # Use float16 for compatibility
        trust_remote_code=True
        # Removed local_files_only - causing path recognition issues
    )
    
    # Apply LoRA if enabled
    if args.use_lora:
        print(f"🔧 Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        
        # Standard Llama target modules
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    if not HAS_TRANSFORMERS:
        print("❌ Required packages not available")
        return
    
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    print("🚀 Starting Simple Llama 3.1 8B SFT Training")
    print("=" * 50)
    print(f"📦 Model: {args.model_name}")
    print(f"📚 Training data: {args.train_data}")
    print(f"🔍 Validation data: {args.val_data}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🧠 LoRA enabled: {args.use_lora}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up model and tokenizer
    print("\n🤖 Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args)
    
    # Load datasets
    print("\n📊 Loading datasets...")
    train_dataset = SFTDataset(args.train_data, tokenizer, args.max_seq_length)
    eval_dataset = SFTDataset(args.val_data, tokenizer, args.max_seq_length)
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_torch",
        learning_rate=args.learning_rate,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,  # Use fp16 instead of bf16
    )
    
    # Standard Trainer (not SFTTrainer for compatibility)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    # Start training
    print("\n🏋️  Starting training...")
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training info
    training_info = {
        "model_name": args.model_name,
        "training_samples": len(train_dataset),
        "validation_samples": len(eval_dataset),
        "epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "max_seq_length": args.max_seq_length,
        "lora_config": {
            "enabled": args.use_lora,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout
        } if args.use_lora else None
    }
    
    info_file = Path(args.output_dir) / "training_info.json"
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Model saved to: {args.output_dir}")
    print(f"📄 Training info: {info_file}")


if __name__ == "__main__":
    main()