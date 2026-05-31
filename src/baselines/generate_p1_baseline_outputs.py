#!/usr/bin/env python3
"""
Generate O1 baseline outputs: send p1 directly to the model.

Input:  JSONL with p1_id, p1 (and any other fields — only chosen keys are written).
Output: p1_baseline_outputs.jsonl with p1_id, p1, output (=O1), etc.; use --omit-p2-in-output
        to drop p2 from each row (P1-only artifact).

Works with both Azure API and local models (vLLM / HuggingFace).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.baselines.baseline_common import load_jsonl, append_jsonl, load_checkpoint
from src.utils.model_client import call_model, batch_call_model, ContentFilteredError
from src.utils.wandb_utils import init_wandb, log_metrics, finish_run

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Provide a clear, helpful response to the user's question."
)

_logged_inference_error = False


def generate_response(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    use_hf: bool,
    model_name: Optional[str],
) -> Optional[str]:
    try:
        text = call_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            use_hf=use_hf,
            model_name=model_name,
        )
        return text.strip() if text is not None else None
    except ContentFilteredError:
        return None
    except Exception as e:
        global _logged_inference_error
        if not _logged_inference_error:
            _logged_inference_error = True
            logger.exception("First inference failure: %s", e)
        else:
            logger.debug("Inference failure: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate O1 baseline outputs by sending p1 directly to the model."
    )
    parser.add_argument("--input-jsonl", required=True, type=str,
                        help="JSONL with p1_id, p1 fields (e.g. the same p1_p2 file used for O2).")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--system-prompt-file", type=str, default=None,
                        help="Optional system prompt file. Default: simple helpful-assistant prompt.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--use-local-model", action="store_true")
    parser.add_argument("--local-model", type=str, default=None)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--exit-on-null", action="store_true",
                        help="Abort on first null inference (row not checkpointed).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of records per vLLM batch call (default: 32).")
    parser.add_argument(
        "--omit-p2-in-output",
        action="store_true",
        help="Do not write a p2 field in each output row (input may still contain p2 for alignment).",
    )
    args = parser.parse_args()

    if os.environ.get("EXIT_ON_NULL", "").strip().lower() in ("1", "true", "yes"):
        args.exit_on_null = True

    if args.use_local_model and args.local_model:
        os.environ["USE_LOCAL_MODEL"] = "true"
        os.environ["LOCAL_MODEL"] = args.local_model

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    records: List[Dict] = load_jsonl(args.input_jsonl)
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        print("No records found.")
        return

    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.jsonl"
    outputs_file = output_dir / "p1_baseline_outputs.jsonl"
    lock_file = output_dir / "write.lock"
    lock_file.touch(exist_ok=True)

    processed_ids = load_checkpoint(checkpoint_file)
    success = 0
    failed = 0

    init_wandb(
        job_name="p1_baseline_outputs",
        config={
            "input_jsonl": str(args.input_jsonl),
            "output_dir": str(output_dir),
            "system_prompt_file": args.system_prompt_file,
            "max_records": args.max_records,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "use_local_model": bool(args.use_local_model),
            "local_model": args.local_model,
            "use_hf": bool(args.use_hf),
            "exit_on_null": bool(args.exit_on_null),
            "omit_p2_in_output": bool(args.omit_p2_in_output),
            "total_records": len(records),
        },
        enabled=args.wandb,
    )

    model_name = args.local_model if args.use_local_model else None
    batch_size = max(1, args.batch_size)

    pending = []
    for idx, rec in enumerate(records, 1):
        p1_id = str(rec.get("p1_id", rec.get("id", f"p1_{idx}")))
        if p1_id in processed_ids:
            continue
        p1 = rec.get("p1", "")
        if not p1.strip():
            logger.warning("Empty p1 for p1_id=%s, skipping.", p1_id)
            continue
        pending.append((p1_id, rec))

    total = len(pending)
    print(f"Pending: {total} records (batch_size={batch_size})")
    done = 0

    for batch_start in range(0, total, batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        prompts = [(system_prompt, rec.get("p1", "")) for _, rec in batch]
        responses = batch_call_model(
            prompts,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_hf=args.use_hf,
            model_name=model_name,
        )

        for (p1_id, rec), response in zip(batch, responses):
            if args.exit_on_null and response is None:
                logger.error(
                    "exit-on-null: null response for p1_id=%s. Aborting.", p1_id,
                )
                finish_run()
                sys.exit(1)

            out = {
                "p1_id": p1_id,
                "p1": rec.get("p1"),
                "output": response if response is not None else None,
                "is_baseline": True,
                "is_p1_baseline": True,
                "category": rec.get("category", ""),
                "lang": rec.get("lang", ""),
                "level": rec.get("level"),
            }
            if not args.omit_p2_in_output:
                out["p2"] = rec.get("p2")
            append_jsonl(str(outputs_file), [out], lock_file)
            append_jsonl(str(checkpoint_file), [out], lock_file)
            processed_ids.add(p1_id)

            if response is not None:
                success += 1
            else:
                failed += 1

        done += len(batch)
        print(f"[{done}/{total}] success={success}, failed={failed}")
        log_metrics(
            {"success": success, "failed": failed,
             "processed": done,
             "processed_frac": done / max(1, total)},
            step=done,
        )

    print(f"Done. success={success}, failed={failed}")
    print(f"Outputs: {outputs_file}")
    log_metrics({"success": success, "failed": failed, "processed": success + failed}, step=len(records))
    finish_run()


if __name__ == "__main__":
    main()
