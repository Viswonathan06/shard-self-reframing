#!/usr/bin/env python3
"""
Generate model outputs from P1/P2 records using control signals.

Input JSONL rows are expected to include:
- p1
- p2
- response_mode

Optional: reformulated_component, preserved_benign_need. mention_level is derived
from response_mode when saving outputs (for downstream compatibility).

The script constructs a user prompt containing those fields and calls the model
with a provided system prompt.
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
from src.utils.response_controls import effective_mention_level
from src.utils.model_client import call_model, batch_call_model, batch_call_model_with_thinking, ContentFilteredError
from src.utils.wandb_utils import init_wandb, log_metrics, finish_run

logger = logging.getLogger(__name__)
_logged_inference_error = False


def build_controlled_user_prompt(rec: Dict) -> str:
    p1 = rec.get("p1", "")
    p2 = rec.get("p2", "")
    response_mode = rec.get("response_mode", "")
    reformulated_component = rec.get("reformulated_component", "")
    preserved_benign_need = rec.get("preserved_benign_need", "")
    return (
        f"Original user prompt (P1):\n{p1}\n\n"
        f"Safe reformulated prompt (P2):\n{p2}\n\n"
        f"response_mode: {response_mode}\n"
        f"reformulated_component: {reformulated_component}\n"
        f"preserved_benign_need: {preserved_benign_need}\n"
    )


def build_refinement_user_prompt(rec: Dict) -> str:
    p1 = rec.get("p1", "")
    p2 = rec.get("p2", "")
    baseline_output = rec.get("baseline_output", "")
    return (
        f"User's original request: {p1}\n\n"
        f"Assuming the user has benign intent, a safer interpretation of their request: {p2}\n\n"
        f"Here is the draft response to refine: {baseline_output}\n\n"
        f"Return only the more helpful response text."
    )


def build_benign_intent_user_prompt(rec: Dict) -> str:
    p1 = rec.get("p1", "")
    baseline_output = rec.get("baseline_output", rec.get("output", ""))
    return (
        f"User request:\n{p1}\n\n"
        f"Draft response:\n{baseline_output}\n"
    )


def generate_controlled_response(
    system_prompt: str,
    rec: Dict,
    temperature: float,
    max_tokens: int,
    use_hf: bool,
    model_name: Optional[str],
    refinement_mode: bool = False,
    benign_intent_mode: bool = False,
) -> Optional[str]:
    if benign_intent_mode:
        user_prompt = build_benign_intent_user_prompt(rec)
    elif refinement_mode:
        user_prompt = build_refinement_user_prompt(rec)
    else:
        user_prompt = build_controlled_user_prompt(rec)
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
            logger.exception(
                "First inference failure (later rows will be null without repeating this traceback). "
                "Common causes: vLLM failed to load with VLLM_ONLY=1, or empty generations. Error: %s",
                e,
            )
        else:
            logger.debug("Inference failure for a row: %s", e)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate controlled outputs from p1_p2 outputs JSONL using response_mode (bundled mention behavior)."
    )
    parser.add_argument("--input-jsonl", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--system-prompt-file", required=True, type=str)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--use-local-model", action="store_true")
    parser.add_argument("--local-model", type=str, default=None)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument(
        "--refinement-mode",
        action="store_true",
        help="Use refinement prompt (requires baseline_output field in input records).",
    )
    parser.add_argument(
        "--benign-intent-mode",
        action="store_true",
        help="Use benign-intent prompt: only P1 + draft response, no P2. "
             "Reads baseline from baseline_output or output field.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (set WANDB_API_KEY or use .wandb_key).",
    )
    parser.add_argument(
        "--exit-on-null",
        action="store_true",
        help="Exit with code 1 on first null/empty inference (no checkpoint line for that row).",
    )
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of records per vLLM batch call (default: 32).")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode and save <think> tokens to 'thinking' field in output.",
    )
    args = parser.parse_args()
    if os.environ.get("EXIT_ON_NULL", "").strip().lower() in ("1", "true", "yes"):
        args.exit_on_null = True

    if args.use_local_model and args.local_model:
        os.environ["USE_LOCAL_MODEL"] = "true"
        os.environ["LOCAL_MODEL"] = args.local_model

    records: List[Dict] = load_jsonl(args.input_jsonl)
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        print("No records found.")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "checkpoint.jsonl"
    outputs_file = output_dir / "baseline_outputs.jsonl"
    lock_file = output_dir / "write.lock"
    lock_file.touch(exist_ok=True)

    processed_ids = load_checkpoint(checkpoint_file)
    success = 0
    failed = 0

    init_wandb(
        job_name=f"p2_controlled_outputs",
        config={
            "input_jsonl": str(args.input_jsonl),
            "output_dir": str(output_dir),
            "system_prompt_file": str(args.system_prompt_file),
            "max_records": args.max_records,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "use_local_model": bool(args.use_local_model),
            "local_model": args.local_model,
            "use_hf": bool(args.use_hf),
            "exit_on_null": bool(args.exit_on_null),
            "total_records": len(records),
        },
        enabled=args.wandb,
    )

    model_name = args.local_model if args.use_local_model else None
    batch_size = max(1, args.batch_size)

    def _build_user_prompt(rec: Dict) -> str:
        if args.benign_intent_mode:
            return build_benign_intent_user_prompt(rec)
        if args.refinement_mode:
            return build_refinement_user_prompt(rec)
        return build_controlled_user_prompt(rec)

    pending = []
    for idx, rec in enumerate(records, 1):
        p1_id = str(rec.get("p1_id", rec.get("id", f"p1_{idx}")))
        if p1_id not in processed_ids:
            pending.append((p1_id, rec))

    total = len(pending)
    print(f"Pending: {total} records (batch_size={batch_size})")
    done = 0

    for batch_start in range(0, total, batch_size):
        batch = pending[batch_start: batch_start + batch_size]
        prompts = [(system_prompt, _build_user_prompt(rec)) for _, rec in batch]
        if args.enable_thinking:
            raw_pairs = batch_call_model_with_thinking(
                prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                use_hf=args.use_hf,
                model_name=model_name,
            )
            thinking_list = [t for t, _ in raw_pairs]
            responses = [r if r else None for _, r in raw_pairs]
        else:
            thinking_list = [None] * len(batch)
            responses = batch_call_model(
                prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                use_hf=args.use_hf,
                model_name=model_name,
            )

        for (p1_id, rec), response, thinking in zip(batch, responses, thinking_list):
            if args.exit_on_null and response is None:
                logger.error(
                    "exit-on-null: inference returned null for p1_id=%s. Aborting without checkpoint append.",
                    p1_id,
                )
                finish_run()
                sys.exit(1)

            out = {
                **rec,
                "p1_id": p1_id,
                "p1": rec.get("p1"),
                "p2": rec.get("p2"),
                "response_mode": rec.get("response_mode"),
                "mention_level": effective_mention_level(rec),
                "reformulated_component": rec.get("reformulated_component"),
                "preserved_benign_need": rec.get("preserved_benign_need"),
                "baseline_output": rec.get("baseline_output") or rec.get("output"),
                "output": response if response is not None else None,
                "is_baseline": True,
                "is_p2_controlled": True,
            }
            if thinking is not None:
                out["thinking"] = thinking
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
            {
                "success": success,
                "failed": failed,
                "processed": done,
                "processed_frac": done / max(1, total),
            },
            step=done,
        )

    print(f"Done. success={success}, failed={failed}")
    print(f"Outputs: {outputs_file}")
    log_metrics({"success": success, "failed": failed, "processed": success + failed}, step=len(records))
    finish_run()


if __name__ == "__main__":
    main()
