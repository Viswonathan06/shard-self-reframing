#!/usr/bin/env python3
"""
Fast SFT Llama 3.1 8B inference using vLLM with native LoRA support.

Reads a JSONL of prompts (must have a `p1` field), runs them through the SFT
LoRA adapter, and writes `sft_generated_response` to the output JSONL.

Usage:
  python scripts/inference_sft_vllm.py \\
    --input-data data/linguasafe_test_unseen.jsonl \\
    --output output/linguasafe_evaluation/sft_llama8b_inference.jsonl \\
    --tensor-parallel-size 4
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


BASE_MODEL_DEFAULT = (
    "$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
SFT_ADAPTER_DEFAULT = "models/llama31_full_sft"
INPUT_DEFAULT = "data/linguasafe_test_unseen.jsonl"
OUTPUT_DEFAULT = "output/linguasafe_evaluation/sft_llama8b_inference.jsonl"


def build_prompt(p1: str) -> str:
    """Matches the SFT training format exactly."""
    return f"### Human: {p1.strip()}\n\n### Assistant: "


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_final_answer(text: str) -> str:
    """
    Strip model-visible reasoning tags and keep only the final user-facing answer.
    Falls back to raw text if no recognized tag structure is found.
    """
    t = (text or "").strip()
    if not t:
        return ""

    # Match tags used in this project (primary: scratchpad; alternate: reasoning).
    for tag in ("scratchpad", "reasoning"):
        end_tag = f"</{tag}>"
        idx = t.lower().find(end_tag)
        if idx != -1:
            return t[idx + len(end_tag):].strip()

    # Generic fallback: drop any leading XML-like block if present.
    m = re.match(r"^\s*<([a-zA-Z_][\w\-]*)>.*?</\1>\s*(.*)$", t, flags=re.DOTALL)
    if m:
        return (m.group(2) or "").strip()
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description="SFT inference via vLLM (batched, tensor parallel)")
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--sft-adapter", default=SFT_ADAPTER_DEFAULT)
    ap.add_argument("--input-data", default=INPUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument("--max-records", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument(
        "--repetition-penalty", type=float, default=1.15,
        help="Penalise repeated tokens (1.0 = off); 1.1-1.2 recommended for greedy LoRA",
    )
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument(
        "--lora-rank", type=int, default=16,
        help="Must match --lora-r used at SFT training time",
    )
    args = ap.parse_args()

    # Load input records
    records = load_records(Path(args.input_data))
    if args.max_records > 0:
        records = records[: args.max_records]
    records = [r for r in records if (r.get("p1") or "").strip()]
    print(f"Prompts to run: {len(records)}")

    # Build prompt strings
    prompts = [build_prompt(r["p1"]) for r in records]

    print(f"\nLoading model:    {args.base_model}")
    print(f"SFT adapter:      {args.sft_adapter}")
    print(f"tensor_parallel_size={args.tensor_parallel_size}  gpu_memory_utilization={args.gpu_memory_utilization}")

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
        lora_name="sft_adapter",
        lora_int_id=1,
        lora_path=str(Path(args.sft_adapter).resolve()),
    )

    sampling_params = SamplingParams(
        temperature=max(0.0, args.temperature),
        max_tokens=args.max_new_tokens,
        stop=["### Human:", "### Assistant:", "###\n", "\n###"],
        repetition_penalty=args.repetition_penalty,
    )

    print(f"\nRunning batched generation on {len(prompts)} prompts …")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    print("Generation complete.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for rec, out in zip(records, outputs):
            response = out.outputs[0].text.strip()
            final_response = extract_final_answer(response)
            row: dict = {
                "p1_id": str(rec.get("p1_id", "")),
                "p1": rec["p1"],
                "p2": rec.get("p2", ""),
                "category": rec.get("category", ""),
                "lang": rec.get("lang", ""),
                "level": rec.get("level", ""),
                "sft_generated_response": response,
                "sft_final_response": final_response,
                "model": "llama31_sft",
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} rows → {out_path}")


if __name__ == "__main__":
    main()
