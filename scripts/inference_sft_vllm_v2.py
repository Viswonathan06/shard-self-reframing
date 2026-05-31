#!/usr/bin/env python3
"""
v2 inference for the SFT v2 LoRA adapter.

Differences vs scripts/inference_sft_vllm.py (v1):
  * Uses Llama-3.1's NATIVE chat template via tokenizer.apply_chat_template
    (must match the template the v2 SFT script trained on).
  * Drops the custom "### Human: / ### Assistant:" stop-token list.
  * Defaults repetition_penalty to 1.0 (off). v1's 1.15 made greedy decoding
    drift on prompts that re-use vocabulary heavily.
  * Defaults --lora-rank to 16 (matches sft_v2.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:
    sys.exit(f"vLLM not found: {e}\n  pip install vllm")

from transformers import AutoTokenizer

BASE_MODEL_DEFAULT = (
    "/playpen/models--meta-llama--Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
SFT_ADAPTER_DEFAULT = "models/llama31_sft_v2"
INPUT_DEFAULT = "data/linguasafe_test_unseen.jsonl"
OUTPUT_DEFAULT = "output/linguasafe_evaluation/sft_v2_llama8b_inference.jsonl"


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_final_answer(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    for tag in ("scratchpad", "reasoning"):
        end_tag = f"</{tag}>"
        idx = t.lower().find(end_tag)
        if idx != -1:
            return t[idx + len(end_tag):].strip()
    m = re.match(r"^\s*<([a-zA-Z_][\w\-]*)>.*?</\1>\s*(.*)$", t, flags=re.DOTALL)
    if m:
        return (m.group(2) or "").strip()
    return t


def build_prompts(records: list[dict], tokenizer) -> list[str]:
    prompts = []
    for r in records:
        msgs = [{"role": "user", "content": (r.get("p1") or "").strip()}]
        prompts.append(
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--sft-adapter", default=SFT_ADAPTER_DEFAULT)
    ap.add_argument("--input-data", default=INPUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--lora-rank", type=int, default=16)
    args = ap.parse_args()

    records = load_records(Path(args.input_data))
    if args.max_records > 0:
        records = records[: args.max_records]
    records = [r for r in records if (r.get("p1") or "").strip()]
    print(f"Prompts to run: {len(records)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = build_prompts(records, tokenizer)
    print("\nExample rendered prompt (first 200 chars):")
    print(prompts[0][:200].replace("\n", "\\n"))

    print(f"\nBase model: {args.base_model}")
    print(f"SFT adapter: {args.sft_adapter}")
    print(f"tensor_parallel={args.tensor_parallel_size}  gpu_mem={args.gpu_memory_utilization}")

    llm = LLM(
        model=args.base_model,
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    lora_request = LoRARequest(
        lora_name="sft_v2",
        lora_int_id=1,
        lora_path=str(Path(args.sft_adapter).resolve()),
    )

    sampling_params = SamplingParams(
        temperature=max(0.0, args.temperature),
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        # Stop on Llama-3.1 turn boundaries instead of custom markers.
        stop_token_ids=[tid for tid in [tokenizer.eos_token_id,
                                        tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                        if tid is not None and tid >= 0],
    )

    print(f"\nGenerating {len(prompts)} responses...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    print("Done.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for rec, out in zip(records, outputs):
            response = (out.outputs[0].text or "").strip()
            final_response = extract_final_answer(response)
            row = {
                "p1_id": str(rec.get("p1_id", "")),
                "p1": rec["p1"],
                "p2": rec.get("p2", ""),
                "category": rec.get("category", ""),
                "lang": rec.get("lang", ""),
                "level": rec.get("level", ""),
                "sft_generated_response": response,
                "sft_final_response": final_response,
                "model": "llama31_sft_v2",
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    main()
