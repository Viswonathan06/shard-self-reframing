#!/usr/bin/env python3
"""
vLLM baseline inference: send p1 prompts through a model with no LoRA adapter.

Input JSONL must have p1_id and p1 fields.
Output matches the p1_baseline_outputs.jsonl format used across the pipeline:
  p1_id, p1, output, is_baseline, is_p1_baseline, category, lang, level, p2

Usage:
  python scripts/run_baseline_inference_vllm.py \
      --model /path/to/model \
      --input  output/DNA_Per_Model_Experiments/refinement_from_baseline_multimodel/llama8b/refinement_input.jsonl \
      --output output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/phi4/p1_baseline_outputs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    sys.exit(f"vLLM not found: {e}")

from transformers import AutoTokenizer

SYSTEM_PROMPT = "You are a helpful assistant."


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",                required=True)
    ap.add_argument("--input",                required=True)
    ap.add_argument("--output",               required=True)
    ap.add_argument("--system-prompt",        default=SYSTEM_PROMPT)
    ap.add_argument("--max-new-tokens",       type=int,   default=2048)
    ap.add_argument("--temperature",          type=float, default=0.3)
    ap.add_argument("--tensor-parallel-size", type=int,   default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len",        type=int,   default=4096)
    ap.add_argument("--max-records",          type=int,   default=0)
    ap.add_argument("--batch-size",           type=int,   default=256,
                    help="Prompts per llm.generate() call. vLLM batches within each call.")
    args = ap.parse_args()

    records = load_records(Path(args.input))
    if args.max_records > 0:
        records = records[:args.max_records]
    records = [r for r in records if (r.get("p1") or "").strip()]
    print(f"Loaded {len(records)} records from {args.input}")

    out_path = Path(args.output)
    # Resume: skip already-done p1_ids
    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(str(json.loads(line)["p1_id"]))
        print(f"Resuming: {len(done_ids)} already done, {len(records) - len(done_ids)} remaining")
    pending = [r for r in records if str(r["p1_id"]) not in done_ids]
    if not pending:
        print("All records already processed.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = []
    for r in pending:
        msgs = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user",   "content": r["p1"].strip()},
        ]
        prompts.append(
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        )

    example = prompts[0][:200].replace("\n", "\\n")
    print(f"Example prompt (first 200 chars): {example}")
    print(f"Model: {args.model}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    stop_ids = [tokenizer.eos_token_id]
    for tok in ("<|im_end|>", "<|eot_id|>", "<|end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)
    stop_ids = list(set(filter(None, stop_ids)))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop_token_ids=stop_ids,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = len(done_ids)
    batch_size = args.batch_size
    print(f"\nGenerating {len(pending)} responses in batches of {batch_size}...")

    with out_path.open("a", encoding="utf-8") as fout:
        for batch_start in range(0, len(pending), batch_size):
            batch_recs    = pending[batch_start : batch_start + batch_size]
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            for rec, out in zip(batch_recs, outputs):
                text = (out.outputs[0].text or "").strip()
                row = {
                    "p1_id":          str(rec["p1_id"]),
                    "p1":             rec["p1"],
                    "output":         text,
                    "is_baseline":    True,
                    "is_p1_baseline": True,
                    "category":       rec.get("category", ""),
                    "lang":           rec.get("lang", ""),
                    "level":          rec.get("level"),
                    "p2":             rec.get("p2", ""),
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_written += len(batch_recs)
            print(f"  [{total_written}/{len(done_ids) + len(pending)}] batches done")

    print(f"Wrote {len(pending)} new rows ({total_written} total) → {out_path}")


if __name__ == "__main__":
    main()
