#!/usr/bin/env python3
"""
Final working SFT script using the exact local model path.
"""

import json
import os
import torch
from pathlib import Path

try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer
    )
    from peft import LoraConfig, get_peft_model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class ScratchpadDataset(torch.utils.data.Dataset):
    """Simple dataset for scratchpad training."""
    
    def __init__(self, file_path, tokenizer, max_length=1024):  # Increase max length
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"📚 Loading data from {file_path}")
        
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                # Remove limit - use all samples
                pass
                    
                record = json.loads(line.strip())
                
                # Extract conversation
                user_msg = ""
                assistant_msg = ""
                
                for msg in record['conversations']:
                    if msg['from'] == 'human':
                        user_msg = msg['value']
                    elif msg['from'] == 'gpt':
                        assistant_msg = msg['value']
                
                if user_msg and assistant_msg:
                    # Simple format
                    text = f"### Human: {user_msg}\n\n### Assistant: {assistant_msg}"
                    self.data.append(text)
        
        print(f"✅ Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


def main():
    if not HAS_TRANSFORMERS:
        print("❌ Transformers not available")
        return
    
    print("🚀 Final SFT Training")
    print("=" * 30)
    
    # Use the exact path you found
    model_path = "/playpen/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    
    print(f"📦 Loading from: {model_path}")
    
    # Load tokenizer directly from local path
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
        print("✅ Tokenizer loaded")
    except Exception as e:
        print(f"❌ Tokenizer failed: {e}")
        # Try with legacy=True as fallback
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
            print("✅ Tokenizer loaded (legacy mode)")
        except Exception as e2:
            print(f"❌ Tokenizer failed completely: {e2}")
            return
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model directly from local path
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Model failed: {e}")
        return
    
    # Apply minimal LoRA
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    print("✅ LoRA applied")
    model.print_trainable_parameters()
    
    # Load dataset
    train_data = "data/sft_scratchpad/train_sft_sharegpt.jsonl"
    dataset = ScratchpadDataset(train_data, tokenizer)
    
    # Full dataset training config
    training_args = TrainingArguments(
        output_dir="models/llama31_full_sft",
        num_train_epochs=2,  # Increase to 2 epochs for full dataset
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # Increase for full dataset
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=100,
        save_strategy="steps",
        report_to=None,
        fp16=True,
        remove_unused_columns=False,
        eval_steps=50,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )
    
    print("🏋️ Starting training...")
    trainer.train()
    
    print("💾 Saving...")
    trainer.save_model()
    tokenizer.save_pretrained("models/llama31_final")
    
    print("🎉 Done!")


if __name__ == "__main__":
    main()