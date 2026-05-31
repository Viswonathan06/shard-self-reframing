#!/usr/bin/env python3
"""
Working SFT script for Llama 3.1 8B with scratchpad reasoning.
Simplified and tested configuration.
"""

import json
import os
import torch
from pathlib import Path

# Set offline mode to use cached models
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        TrainingArguments, Trainer
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ScratchpadDataset(torch.utils.data.Dataset):
    """Dataset for scratchpad SFT training."""
    
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"📚 Loading data from {file_path}")
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 50:  # Limit to 50 samples for quick testing
                    break
                    
                record = json.loads(line.strip())
                
                # Simple format: user prompt + assistant scratchpad+response
                user_msg = ""
                assistant_msg = ""
                
                for msg in record['conversations']:
                    if msg['from'] == 'human':
                        user_msg = msg['value']
                    elif msg['from'] == 'gpt':
                        assistant_msg = msg['value']
                
                if user_msg and assistant_msg:
                    # Create training text
                    text = f"Human: {user_msg}\n\nAssistant: {assistant_msg}{tokenizer.eos_token}"
                    self.data.append(text)
        
        print(f"✅ Loaded {len(self.data)} training examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Don't pad individual examples
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()  # For causal LM, labels = input_ids
        }


def setup_model_and_tokenizer():
    """Set up model and tokenizer with working configuration."""
    
    # Use the standard model name - should use cached version
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    print(f"🤖 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"🤖 Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Apply LoRA
    print("🔧 Applying LoRA configuration")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def collate_fn(examples):
    """Custom collate function for batching."""
    
    # Get max length in batch
    max_len = max(len(ex['input_ids']) for ex in examples)
    max_len = min(max_len, 1024)  # Cap at 1024
    
    batch = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for ex in examples:
        input_ids = ex['input_ids'][:max_len]  # Truncate if needed
        attention_mask = ex['attention_mask'][:max_len]
        
        # Pad to max_len
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            pad_token = 0  # Usually 0 for padding
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len)])
        
        batch['input_ids'].append(input_ids)
        batch['attention_mask'].append(attention_mask)
        batch['labels'].append(input_ids.clone())
    
    # Stack into tensors
    batch['input_ids'] = torch.stack(batch['input_ids'])
    batch['attention_mask'] = torch.stack(batch['attention_mask'])
    batch['labels'] = torch.stack(batch['labels'])
    
    return batch


def main():
    if not HAS_TRANSFORMERS:
        print("❌ Required packages not available")
        return
    
    print("🚀 Starting Working SFT Training")
    print("=" * 40)
    
    # Paths
    train_data = "data/sft_scratchpad/train_sft_sharegpt.jsonl"
    val_data = "data/sft_scratchpad/val_sft_sharegpt.jsonl"
    output_dir = "models/llama31_sft_working"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load datasets
    print("\n📊 Loading datasets...")
    train_dataset = ScratchpadDataset(train_data, tokenizer)
    eval_dataset = ScratchpadDataset(val_data, tokenizer)
    
    # Training arguments - conservative settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        evaluation_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        load_best_model_at_end=False,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,
        dataloader_drop_last=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn
    )
    
    # Train
    print("\n🏋️  Starting training...")
    trainer.train()
    
    # Save
    print("\n💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n🎉 Training completed! Model saved to {output_dir}")


if __name__ == "__main__":
    main()