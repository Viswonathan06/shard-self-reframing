#!/usr/bin/env python3
"""
Generate baseline (Llama-3.1-8B-Instruct) and SFT-v2 responses on the DNA v2
held-out validation set, and write a single refinement-style JSONL that the
existing judge script can consume.

Output JSONL fields per row (matches what evaluate_safety_helpfulness_single_random.py
expects with --refinement-jsonl):
  p1_id, p1, p2, category,
  baseline_output  -> o1   (Llama-3.1-8B-Instruct, no LoRA)
  output           -> o2   (Llama-3.1-8B-Instruct + sft_v2 LoRA, scratchpad stripped)
  raw_sft_output   (full SFT generation incl. <scratchpad>)
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
    sys.exit(f"vLLM not found: {e}")

from transformers import AutoTokenizer

BASE_MODEL_DEFAULT = (
    "/playpen/models--meta-llama--Llama-3.1-8B-Instruct"
    "/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
ADAPTER_DEFAULT = "models/llama31_sft_v2"
VAL_DEFAULT = "data/sft_scratchpad_v2/val_sft.jsonl"
SOURCE_DEFAULT = "output/qwen480b_scratchpad_llama8b_full.jsonl"
OUT_DEFAULT = "output/sft_evaluation/dna_val_v2_baseline_vs_sft_v2.jsonl"


def load_val_records(val_jsonl: Path, source_jsonl: Path) -> list[dict]:
    """Combine the v2 val split (which only carries `messages`) with the source
    file (which carries `p1`, `p2`-ish placeholders, `category`).
    """
    src_by_id = {}
    if source_jsonl.exists():
        with source_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if "p1_id" in r:
                    src_by_id[str(r["p1_id"])] = r

    rows = []
    with val_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            # `id` looks like "dna_sft_<p1_id>"
            stub = (r.get("id") or "").split("dna_sft_", 1)[-1]
            src = src_by_id.get(stub, {})
            msgs = r.get("messages") or []
            user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
            rows.append({
                "p1_id": stub,
                "p1": user_msg or src.get("p1", ""),
                "p2": src.get("p1", ""),  # no separate p2 in DNA — reuse p1 as a placeholder
                "category": r.get("category") or src.get("category", ""),
            })
    return rows


def render_prompt(p1: str, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": p1}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_final_answer(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    for tag in ("scratchpad", "reasoning"):
        end_tag = f"</{tag}>"
        idx = t.lower().find(end_tag)
        if idx != -1:
            return t[idx + len(end_tag):].strip()
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--sft-adapter", default=ADAPTER_DEFAULT)
    ap.add_argument("--val-jsonl", default=VAL_DEFAULT)
    ap.add_argument("--source-jsonl", default=SOURCE_DEFAULT)
    ap.add_argument("--output", default=OUT_DEFAULT)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--lora-rank", type=int, default=16)
    args = ap.parse_args()

    val_path = Path(args.val_jsonl)
    src_path = Path(args.source_jsonl)
    rows = load_val_records(val_path, src_path)
    if args.max_records:
        rows = rows[: args.max_records]
    rows = [r for r in rows if (r.get("p1") or "").strip()]
    print(f"Val records: {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [render_prompt(r["p1"], tokenizer) for r in rows]

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

    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    stop_ids = [tid for tid in [tokenizer.eos_token_id, eot] if isinstance(tid, int) and tid >= 0]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop_token_ids=stop_ids)

    print("Generating BASELINE responses (no LoRA)...")
    base_out = llm.generate(prompts, sp)

    print("Generating SFT v2 responses (with LoRA)...")
    lora_request = LoRARequest(
        lora_name="sft_v2", lora_int_id=1, lora_path=str(Path(args.sft_adapter).resolve())
    )
    sft_out = llm.generate(prompts, sp, lora_request=lora_request)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for rec, b, s in zip(rows, base_out, sft_out):
            base_text = (b.outputs[0].text or "").strip()
            sft_raw = (s.outputs[0].text or "").strip()
            sft_final = extract_final_answer(sft_raw)
            row = {
                "p1_id": str(rec["p1_id"]),
                "p1": rec["p1"],
                "p2": rec.get("p2", ""),
                "category": rec.get("category", ""),
                "baseline_output": base_text,
                "output": sft_final,
                "raw_sft_output": sft_raw,
                "baseline_model": "llama31-8b-instruct",
                "sft_model": "llama31_sft_v2",
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
