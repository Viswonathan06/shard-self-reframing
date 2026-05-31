#!/usr/bin/env python3
"""
Baseline pipeline: Generate model responses directly from P1 prompts
(No P1 to P2 conversion - just ask the model P1 directly)
"""

import json
import argparse
import os
import fcntl
from pathlib import Path
from typing import Dict, List, Optional, Set
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.baselines.baseline_common import (
    load_jsonl,
    append_jsonl,
    claim_next_index,
    load_checkpoint,
    generate_model_response,
)

# Local model options (can be overridden by command line or environment)
USE_LOCAL_MODEL_ENV = os.getenv("USE_LOCAL_MODEL", "false").strip().lower()
USE_LOCAL_MODEL = USE_LOCAL_MODEL_ENV == "true" or USE_LOCAL_MODEL_ENV == "1"
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def load_p1_from_csv(csv_path: str, language: str, max_p1: Optional[int] = None) -> List[Dict]:
    """
    Load P1 prompts from CSV file and filter by language.

    Args:
        csv_path: Path to linguasafe.csv
        language: Language code to filter (e.g., 'en', 'zh', 'ar')
        max_p1: Maximum number of prompts to load (None = all)

    Returns:
        List of dicts with 'p1' (prompt text) and metadata
    """
    df = pd.read_csv(csv_path)

    # Filter by language
    df_filtered = df[df["lang"] == language].copy()

    if len(df_filtered) == 0:
        print(f"Warning: No prompts found for language '{language}' in {csv_path}")
        return []

    # Limit if specified
    if max_p1 is not None and max_p1 > 0:
        df_filtered = df_filtered.head(max_p1)

    # Convert to list of dicts
    records = []
    for _, row in df_filtered.iterrows():
        rec = {
            "id": str(row.get("id", "")),
            "p1": str(row.get("prompt", "")),
            "lang": str(row.get("lang", language)),
            "level": row.get("level", None),
            "subtype": str(row.get("subtype", "")) if pd.notna(row.get("subtype")) else "",
            "type": str(row.get("type", "")) if pd.notna(row.get("type")) else "",
            "source": str(row.get("source", "")) if pd.notna(row.get("source")) else "",
        }
        records.append(rec)

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Baseline: Generate model responses directly from P1 (no P2 conversion)"
    )
    parser.add_argument(
        "--csv-path", type=str, required=True, help="Path to linguasafe.csv dataset"
    )
    parser.add_argument(
        "--language", type=str, required=True, help="Language code (e.g., 'en', 'zh', 'ar')"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--max-p1", type=int, default=None, help="Maximum number of P1 prompts to process"
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
        help="Hugging Face model name for local inference",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model name to use (overrides AZURE_OPENAI_MODEL)"
    )
    parser.add_argument(
        "--queue-file",
        type=str,
        default=None,
        help="Queue file (JSONL) - if provided, uses queue-based processing",
    )
    parser.add_argument(
        "--cursor-file",
        type=str,
        default=None,
        help="Cursor file for queue-based processing",
    )
    parser.add_argument(
        "--write-lock-file",
        type=str,
        default=None,
        help="Lock file for safe appending",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Checkpoint file to track processed P1s",
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

    # Log which model is being used
    if USE_LOCAL_MODEL:
        print(f"Using local model: {LOCAL_MODEL_NAME}")
    else:
        print("Using API model (Azure OpenAI)")

    # Load P1 prompts
    print(f"\n{'='*80}")
    print("Loading P1 prompts")
    print(f"{'='*80}")
    p1_prompts = load_p1_from_csv(args.csv_path, args.language, args.max_p1)
    print(f"Loaded {len(p1_prompts)} P1 prompts for language '{args.language}'")

    if not p1_prompts:
        print("No prompts to process. Exiting.")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup checkpoint and queue files
    checkpoint_file = (
        Path(args.checkpoint_file) if args.checkpoint_file else output_dir / "checkpoint.jsonl"
    )

    # Load checkpoint
    processed_ids = load_checkpoint(checkpoint_file)

    # Setup output files
    all_results_file = output_dir / "baseline_outputs.jsonl"

    # Determine processing mode: queue-based or sequential
    use_queue = args.queue_file and args.cursor_file and args.write_lock_file

    if use_queue:
        # Queue-based processing (for parallel workers)
        print(f"\n{'='*80}")
        print("Queue-based processing mode")
        print(f"{'='*80}")

        queue_file = Path(args.queue_file)
        cursor_file = Path(args.cursor_file)
        lock_file = Path(args.write_lock_file)

        # Load queue
        queue_records = []
        if queue_file.exists():
            queue_records = load_jsonl(str(queue_file))
        else:
            print(f"Creating queue file: {queue_file}")
            with open(queue_file, "w", encoding="utf-8") as f:
                for rec in p1_prompts:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            queue_records = p1_prompts

        print(f"Queue loaded: {len(queue_records)} records")

        # Initialize cursor if needed
        if not cursor_file.exists():
            cursor_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cursor_file, "w") as f:
                f.write("0\n")

        # Initialize lock file if needed
        if not lock_file.exists():
            lock_file.parent.mkdir(parents=True, exist_ok=True)
            lock_file.touch()

        # Process queue
        pid = os.getpid()
        successful = 0
        failed = 0

        while True:
            print(f"Worker {pid} waiting for next index...")
            idx = claim_next_index(cursor_file, len(queue_records))
            if idx is None:
                print(f"Worker {pid}: Queue exhausted, exiting.")
                break

            p1_record = queue_records[idx]
            p1_id = p1_record.get("id", f"p1_{idx}")

            # Skip if already processed (checkpoint)
            if str(p1_id) in processed_ids:
                print(f"Worker {pid}: P1 {p1_id} already processed (checkpoint), skipping.")
                continue

            p1 = p1_record["p1"]
            category = p1_record.get("type", "")
            level = p1_record.get("level")

            print(f"Worker {pid} processing index {idx} (P1 ID: {p1_id})")
            print(f"  Category: {category}")
            print(f"  Level: {level}")
            print(f"  P1: {p1[:80]}...")

            # Generate baseline response (directly from P1, no conversion)
            print("  Generating baseline response...")
            response = generate_model_response(p1)

            if not response:
                print("    ❌ Failed to generate response")
                failed += 1
                result = {
                    "p1_id": p1_id,
                    "p1": p1,
                    "output": None,
                    "error": "Response generation failed",
                    "is_baseline": True,
                    **p1_record,
                }
                append_jsonl(str(all_results_file), [result], lock_file)
                append_jsonl(str(checkpoint_file), [result], lock_file)
                continue

            print(f"    ✓ Generated response ({len(response)} chars)")
            result = {
                "p1_id": p1_id,
                "p1": p1,
                "output": response,
                "is_baseline": True,
                **p1_record,
            }
            successful += 1
            append_jsonl(str(all_results_file), [result], lock_file)
            append_jsonl(str(checkpoint_file), [result], lock_file)
            print(f"    ✓ Saved result (total successful: {successful})")

        print(f"\n{'='*80}")
        print("Queue processing complete")
        print(f"{'='*80}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {successful + failed}")
        print(f"\nResults saved to: {all_results_file}")
        print(f"Checkpoint saved to: {checkpoint_file}")

    else:
        # Sequential processing
        print(f"\n{'='*80}")
        print("Sequential processing mode")
        print(f"{'='*80}")

        results = []
        successful = 0
        failed = 0

        for idx, p1_record in enumerate(p1_prompts, 1):
            p1_id = p1_record.get("id", f"p1_{idx}")

            # Skip if already processed (checkpoint)
            if str(p1_id) in processed_ids:
                print(f"[{idx}/{len(p1_prompts)}] P1 {p1_id} already processed, skipping.")
                continue

            p1 = p1_record["p1"]
            category = p1_record.get("type", "")
            level = p1_record.get("level")

            print(f"\n[{idx}/{len(p1_prompts)}] Processing P1 ID: {p1_id}")
            print(f"  Category: {category}")
            print(f"  Level: {level}")
            print(f"  P1: {p1[:80]}...")

            # Generate baseline response (directly from P1, no conversion)
            print("  Generating baseline response...")
            response = generate_model_response(p1)

            if not response:
                print("    ❌ Failed to generate response")
                failed += 1
                result = {
                    "p1_id": p1_id,
                    "p1": p1,
                    "output": None,
                    "error": "Response generation failed",
                    "is_baseline": True,
                    **p1_record,
                }
                results.append(result)
            else:
                print(f"    ✓ Generated response ({len(response)} chars)")
                result = {
                    "p1_id": p1_id,
                    "p1": p1,
                    "output": response,
                    "is_baseline": True,
                    **p1_record,
                }
                results.append(result)
                successful += 1

            # Checkpoint after each result so job can resume if interrupted
            processed_ids.add(str(p1_id))
            append_jsonl(str(all_results_file), [result], output_dir / "write.lock")
            append_jsonl(str(checkpoint_file), [result], output_dir / "write.lock")
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
