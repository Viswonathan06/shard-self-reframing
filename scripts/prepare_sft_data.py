#!/usr/bin/env python3
"""
Prepare scratchpad data for Supervised Fine-Tuning (SFT) on Llama 3.1 8B.
Converts Qwen-generated scratchpad data into training format.
"""

import json
import argparse
import re
from pathlib import Path


def _wrap_reasoning_and_answer(inner: str, answer: str, tag: str) -> str:
    t = tag.strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", t):
        raise ValueError(f"Invalid assistant reasoning XML tag: {tag!r}")
    return f"<{t}>\n{inner}\n</{t}>\n\n{answer}"


def format_sft_conversation(record, reasoning_tag: str = "scratchpad"):
    """Convert a scratchpad record into SFT conversation format."""
    
    p1 = record['p1']
    generated_scratchpad = record['generated_scratchpad']
    generated_o2 = record['generated_o2']
    category = record['category']
    
    # Create the conversation format for Llama 3.1 8B SFT
    # Format: User prompt -> Assistant with scratchpad reasoning + response
    
    conversation = [
        {
            "role": "user", 
            "content": p1
        },
        {
            "role": "assistant",
            "content": _wrap_reasoning_and_answer(
                generated_scratchpad, generated_o2, reasoning_tag
            ),
        }
    ]
    
    return {
        "id": f"dna_sft_{record['p1_id']}",
        "conversations": conversation,
        "category": category,
        "source": "qwen480b_scratchpad_generation"
    }


def format_sft_instruction_response(record, reasoning_tag: str = "scratchpad"):
    """Alternative format: instruction-response pairs for SFT."""
    
    p1 = record['p1']
    generated_scratchpad = record['generated_scratchpad'] 
    generated_o2 = record['generated_o2']
    category = record['category']
    
    return {
        "instruction": p1,
        "input": "",  # No additional input context
        "output": _wrap_reasoning_and_answer(
            generated_scratchpad, generated_o2, reasoning_tag
        ),
        "id": f"dna_sft_{record['p1_id']}",
        "category": category,
        "source": "qwen480b_scratchpad_generation"
    }


def format_sft_sharegpt(record, reasoning_tag: str = "scratchpad"):
    """ShareGPT format for compatibility with various SFT frameworks."""
    
    p1 = record['p1']
    generated_scratchpad = record['generated_scratchpad']
    generated_o2 = record['generated_o2']
    category = record['category']
    
    return {
        "id": f"dna_sft_{record['p1_id']}",
        "conversations": [
            {"from": "human", "value": p1},
            {
                "from": "gpt",
                "value": _wrap_reasoning_and_answer(
                    generated_scratchpad, generated_o2, reasoning_tag
                ),
            }
        ],
        "category": category,
        "source": "qwen480b_scratchpad_generation"
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare scratchpad data for SFT")
    parser.add_argument('--input', required=True, help='Input JSONL file with scratchpad data')
    parser.add_argument('--output-dir', required=True, help='Output directory for SFT datasets')
    parser.add_argument('--format', choices=['conversation', 'instruction', 'sharegpt'], 
                       default='conversation', help='Output format')
    parser.add_argument('--train-split', type=float, default=0.9, 
                       help='Fraction of data for training (rest for validation)')
    parser.add_argument(
        '--assistant-reasoning-tag',
        type=str,
        default='scratchpad',
        help='XML tag name wrapping STEPs in assistant text, e.g. scratchpad (default) or reasoning. '
        'If you change this, re-run SFT; inference parses both scratchpad and reasoning.',
    )

    args = parser.parse_args()
    
    print(f"🚀 Preparing SFT data from: {args.input}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📋 Format: {args.format}")
    print(f"📊 Train/Val split: {args.train_split:.1%}/{1-args.train_split:.1%}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and process data
    records = []
    with open(args.input, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            
            # Skip records with errors
            if 'error' in record:
                continue
                
            # Choose formatting function
            rt = args.assistant_reasoning_tag
            if args.format == 'conversation':
                sft_record = format_sft_conversation(record, rt)
            elif args.format == 'instruction':
                sft_record = format_sft_instruction_response(record, rt)
            elif args.format == 'sharegpt':
                sft_record = format_sft_sharegpt(record, rt)
            
            records.append(sft_record)
    
    print(f"✅ Processed {len(records)} valid records")
    
    # Split data
    split_idx = int(len(records) * args.train_split)
    train_records = records[:split_idx]
    val_records = records[split_idx:]
    
    print(f"📚 Training records: {len(train_records)}")
    print(f"🔍 Validation records: {len(val_records)}")
    
    # Write training data
    train_file = output_path / f"train_sft_{args.format}.jsonl"
    with open(train_file, 'w') as f:
        for record in train_records:
            f.write(json.dumps(record) + '\n')
    
    # Write validation data
    val_file = output_path / f"val_sft_{args.format}.jsonl"
    with open(val_file, 'w') as f:
        for record in val_records:
            f.write(json.dumps(record) + '\n')
    
    # Create dataset info
    info = {
        "dataset_name": "dna_scratchpad_sft",
        "description": "Scratchpad reasoning dataset for DNA safety SFT",
        "format": args.format,
        "source_model": "qwen3-coder-480b",
        "target_model": "llama-3.1-8b",
        "total_records": len(records),
        "assistant_reasoning_tag": args.assistant_reasoning_tag,
        "train_records": len(train_records),
        "val_records": len(val_records),
        "categories": list(set(r.get('category', 'Unknown') for r in records)),
        "files": {
            "train": str(train_file.name),
            "validation": str(val_file.name)
        }
    }
    
    info_file = output_path / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n🎉 SFT dataset prepared!")
    print(f"📄 Training: {train_file}")
    print(f"📄 Validation: {val_file}")
    print(f"📄 Info: {info_file}")
    
    # Show sample
    print(f"\n📝 Sample training record:")
    print(json.dumps(train_records[0], indent=2)[:500] + "...")


if __name__ == '__main__':
    main()