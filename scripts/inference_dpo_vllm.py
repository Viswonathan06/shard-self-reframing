#!/usr/bin/env python3
"""
Fast DPO Llama 3.1 8B inference using vLLM with native LoRA support.

Batches all P1 prompts at once with tensor parallelism — ~50x faster than
sequential HuggingFace inference.

Usage:
  python scripts/inference_dpo_vllm.py \\
    --input-data output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/llama8b/baseline_outputs.jsonl \\
    --output output/dpo_evaluation/dpo_llama8b_inference.jsonl \\
    --tensor-parallel-size 4
"""
from __future__ import annotations

import argparse
import json
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
DPO_ADAPTER_DEFAULT = "models/llama31_dpo_qwen_vs_baseline"
INPUT_DEFAULT = (
    "output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel"
    "/llama8b/baseline_outputs.jsonl"
)
OUTPUT_DEFAULT = "output/dpo_evaluation/dpo_llama8b_inference.jsonl"
P1_BASELINE_DEFAULT = (
    "output/DNA_New_Experiments/dna_baseline_multimodel/llama8b/p1_baseline_outputs.jsonl"
)


def build_prompt(p1: str) -> str:
    """Matches DPO training format exactly."""
    return f"### Human: {p1.strip()}\n\n### Assistant: "


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_p1_baseline_map(path: Path) -> dict:
    if not path.is_file():
        return {}
    m = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = str(r.get("p1_id", ""))
            out = r.get("output") or r.get("baseline_output")
            if pid and out:
                m[pid] = out
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO inference via vLLM (batched, tensor parallel)")
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--dpo-adapter", default=DPO_ADAPTER_DEFAULT)
    ap.add_argument("--input-data", default=INPUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument("--p1-baseline-jsonl", default=P1_BASELINE_DEFAULT)
    ap.add_argument("--max-records", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--repetition-penalty", type=float, default=1.15,
                    help="Penalise repeated tokens (1.0 = off); 1.1-1.2 recommended for greedy LoRA")
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--lora-rank", type=int, default=16, help="Must match --lora-r used at DPO training time")
    args = ap.parse_args()

    # Load baseline map for comparison column
    p1_baseline_map = load_p1_baseline_map(Path(args.p1_baseline_jsonl))
    print(f"Loaded {len(p1_baseline_map)} P1 baseline rows for comparison")

    # Load input records
    records = load_records(Path(args.input_data))
    if args.max_records > 0:
        records = records[: args.max_records]
    # Filter empty p1s
    records = [r for r in records if (r.get("p1") or "").strip()]
    print(f"Prompts to run: {len(records)}")

    # Build prompt strings
    prompts = [build_prompt(r["p1"]) for r in records]

    # ── vLLM setup ────────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.base_model}")
    print(f"LoRA adapter:  {args.dpo_adapter}")
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
        lora_name="dpo_adapter",
        lora_int_id=1,
        lora_path=str(Path(args.dpo_adapter).resolve()),
    )

    # Stop before the model hallucinates a new ### Human: turn.
    # repetition_penalty > 1 suppresses looping common with greedy + LoRA.
    sampling_params = SamplingParams(
        temperature=max(0.0, args.temperature),
        max_tokens=args.max_new_tokens,
        stop=["### Human:", "### Assistant:", "###\n", "\n###"],
        repetition_penalty=args.repetition_penalty,
    )

    # ── Batched generation ────────────────────────────────────────────────────
    print(f"\nRunning batched generation on {len(prompts)} prompts ...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    print("Generation complete.")

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for rec, out in zip(records, outputs):
            response = out.outputs[0].text.strip()
            pid = str(rec.get("p1_id", ""))
            row: dict = {
                "p1_id": pid,
                "p1": rec["p1"],
                "p2": rec.get("p2", ""),
                "category": rec.get("category", ""),
                "dpo_generated_response": response,
                "model": "llama31_dpo_qwen_vs_baseline",
            }
            if pid in p1_baseline_map:
                row["p1_baseline_output"] = p1_baseline_map[pid]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} rows to {out_path}")


if __name__ == "__main__":
    main()
