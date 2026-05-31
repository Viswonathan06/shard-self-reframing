#!/usr/bin/env python3
"""
P2 Baseline pipeline: Generate model responses from safe rewrites (P2).
Same pipeline as P1 baseline, but uses filtered P2 prompts instead of P1.
Input: p1_p2_outputs_filtered.jsonl (from filter_p2_refusals.py)
Output: baseline_outputs.jsonl with p1_id, p1, p2, output, lang, category, level
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

load_dotenv()

from src.baselines.baseline_common import (
    load_jsonl,
    append_jsonl,
    load_checkpoint,
    generate_model_response,
)

USE_LOCAL_MODEL_ENV = os.getenv("USE_LOCAL_MODEL", "false").strip().lower()
USE_LOCAL_MODEL = USE_LOCAL_MODEL_ENV == "true" or USE_LOCAL_MODEL_ENV == "1"
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def load_p2_prompts(jsonl_path: str, max_records: Optional[int] = None) -> List[Dict]:
    """
    Load P1-P2 pairs from filtered JSONL (from filter_p2_refusals.py).
    P2 is used as the prompt for baseline generation.

    Returns:
        List of dicts with 'p1', 'p2', 'p1_id', 'lang', 'category', 'level', 'id'
    """
    records = load_jsonl(jsonl_path)
    out = []
    for rec in records:
        out.append({
            "id": str(rec.get("p1_id", "")),
            "p1_id": rec.get("p1_id"),
            "p1": rec.get("p1", ""),
            "p2": rec.get("p2", ""),
            "lang": rec.get("lang", ""),
            "category": rec.get("category", ""),
            "level": rec.get("level"),
            "type": rec.get("category", ""),
        })
    if max_records is not None and max_records > 0:
        out = out[:max_records]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="P2 Baseline: Generate model responses from safe rewrites (P2 prompts)"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to filtered P2 JSONL (p1_p2_outputs_filtered.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records to process",
    )
    parser.add_argument(
        "--use-local-model",
        action="store_true",
        default=False,
        help="Use local Hugging Face model instead of API",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Hugging Face model path or name for local inference",
    )

    args = parser.parse_args()

    # Override environment variables with command line args
    global USE_LOCAL_MODEL, LOCAL_MODEL_NAME
    if args.use_local_model:
        USE_LOCAL_MODEL = True
        print("✓ Local model enabled via --use-local-model flag")
    if args.local_model:
        LOCAL_MODEL_NAME = args.local_model
        os.environ["USE_LOCAL_MODEL"] = "true"
        os.environ["LOCAL_MODEL"] = args.local_model
        print(f"✓ Local model set to: {LOCAL_MODEL_NAME}")

    if USE_LOCAL_MODEL:
        print(f"Using local model: {LOCAL_MODEL_NAME}")
    else:
        print("Using API model (Azure OpenAI)")

    # Load P2 prompts
    print(f"\n{'='*80}")
    print("Loading P1-P2 pairs from JSONL (using P2 as prompt)")
    print(f"{'='*80}")
    prompts = load_p2_prompts(args.input_jsonl, args.max_records)
    print(f"Loaded {len(prompts)} P1-P2 pairs from {args.input_jsonl}")

    if not prompts:
        print("No prompts to process. Exiting.")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output_dir / "checkpoint.jsonl"
    all_results_file = output_dir / "baseline_outputs.jsonl"
    lock_file = output_dir / "write.lock"
    lock_file.touch(exist_ok=True)

    processed_ids = load_checkpoint(checkpoint_file)

    print(f"\n{'='*80}")
    print("Sequential processing mode")
    print(f"{'='*80}")

    successful = 0
    failed = 0

    for idx, rec in enumerate(prompts, 1):
        p1_id = rec.get("id", str(rec.get("p1_id", f"p1_{idx}")))

        if str(p1_id) in processed_ids:
            print(f"[{idx}/{len(prompts)}] Row p1_id={p1_id} already done (P2→output), skipping.")
            continue

        p1 = rec.get("p1", "")
        p2 = rec.get("p2")
        p2_text = str(p2).strip() if p2 is not None else ""
        category = rec.get("category", rec.get("type", ""))
        level = rec.get("level")

        print(f"\n[{idx}/{len(prompts)}] Row p1_id={p1_id} — generating response to P2 (safe prompt)")
        print(f"  Category: {category}")
        print(f"  Level: {level}")
        if not p2_text:
            print("  ❌ Missing/empty P2; checkpointing as failed and continuing.")
            failed += 1
            result = {
                "p1_id": p1_id,
                "p1": p1,
                "p2": p2,
                "output": None,
                "error": "Missing or empty P2",
                "is_baseline": True,
                "is_p2_baseline": True,
                **rec,
            }
            processed_ids.add(str(p1_id))
            append_jsonl(str(all_results_file), [result], lock_file)
            append_jsonl(str(checkpoint_file), [result], lock_file)
            continue

        print(f"  P2: {p2_text[:80]}...")

        print("  Calling model with P2 as user prompt...")
        response = generate_model_response(p2_text)

        if not response:
            print("    ❌ Failed to generate response")
            failed += 1
            result = {
                "p1_id": p1_id,
                "p1": p1,
                "p2": p2,
                "output": None,
                "error": "Response generation failed",
                "is_baseline": True,
                "is_p2_baseline": True,
                **rec,
            }
        else:
            print(f"    ✓ Generated response ({len(response)} chars)")
            result = {
                "p1_id": p1_id,
                "p1": p1,
                "p2": p2,
                "output": response,
                "is_baseline": True,
                "is_p2_baseline": True,
                **rec,
            }
            successful += 1

        processed_ids.add(str(p1_id))
        append_jsonl(str(all_results_file), [result], lock_file)
        append_jsonl(str(checkpoint_file), [result], lock_file)
        print(f"    ✓ Checkpointed (total successful: {successful})")

    print(f"\n{'='*80}")
    print("Processing complete")
    print(f"{'='*80}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {successful + failed}")
    print(f"\nResults saved to: {all_results_file}")
    print(f"Checkpoint saved to: {checkpoint_file}")


if __name__ == "__main__":
    main()
