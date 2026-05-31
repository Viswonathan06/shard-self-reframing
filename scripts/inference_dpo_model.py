#!/usr/bin/env python3
"""
Run inference on the DPO-trained Llama 3.1 8B adapter.

The DPO adapter was trained with:
  prompt  = "### Human: {p1}\n\n### Assistant: "
  chosen  = Qwen 480B generated response
  rejected = Llama 8B direct P1 baseline response

At inference we feed the same "### Human: {p1}\n\n### Assistant: " prefix
and let the model continue.  No scratchpad/CoT structure is expected.

Usage:
  python scripts/inference_dpo_model.py \\
    --input-data output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/llama8b/baseline_outputs.jsonl \\
    --output output/dpo_evaluation/dpo_llama8b_inference.jsonl \\
    --max-records 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n  pip install transformers peft accelerate")


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


def load_model(base_model_path: str, dpo_adapter_path: str, num_gpus: int = 1):
    n_avail = torch.cuda.device_count()
    n_use = min(num_gpus, n_avail) if n_avail > 0 else 1
    if n_avail == 0:
        device_map: str | dict = "cpu"
    elif n_use == 1:
        device_map = "cuda:0"
    else:
        device_map = "auto"   # spread across all visible GPUs
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading base model: {base_model_path}  (gpus={n_use}/{n_avail}, dtype={dtype}, device_map={device_map})")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"Loading DPO adapter: {dpo_adapter_path}")
    model = PeftModel.from_pretrained(base, dpo_adapter_path, device_map=device_map)
    model.eval()

    print(f"Loading tokenizer: {dpo_adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(dpo_adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


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
            if pid and out is not None:
                m[pid] = out
    return m


def build_prompt(p1: str) -> str:
    """Match DPO training prompt format exactly."""
    return f"### Human: {p1.strip()}\n\n### Assistant: "


def generate(
    model,
    tokenizer,
    p1: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    prompt = build_prompt(p1)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    gen_kwargs: dict = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO Llama 3.1 8B inference on P1 prompts")
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--dpo-adapter", default=DPO_ADAPTER_DEFAULT)
    ap.add_argument("--input-data", default=INPUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument(
        "--p1-baseline-jsonl",
        default=P1_BASELINE_DEFAULT,
        help="Llama 8B P1 baseline JSONL to attach as p1_baseline_output for comparison",
    )
    ap.add_argument("--max-records", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to shard model across (device_map=auto when >1)")
    args = ap.parse_args()

    p1_baseline_map = load_p1_baseline_map(Path(args.p1_baseline_jsonl))
    if p1_baseline_map:
        print(f"Loaded {len(p1_baseline_map)} P1 baseline rows for comparison")
    else:
        print(f"Warning: no P1 baseline map loaded from {args.p1_baseline_jsonl}")

    model, tokenizer = load_model(args.base_model, args.dpo_adapter, num_gpus=args.num_gpus)

    in_path = Path(args.input_data)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        total = sum(1 for l in f if l.strip())
    limit = total if args.max_records <= 0 else min(args.max_records, total)
    print(f"\nProcessing {limit}/{total} rows  →  {out_path}")

    done = errors = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if done + errors >= limit:
                break
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            p1 = (rec.get("p1") or "").strip()
            if not p1:
                continue

            verbose = i < 2 or (i + 1) % 50 == 0 or done + errors + 1 == limit
            if verbose:
                print(f"[{done + errors + 1}/{limit}] {rec.get('category','')} | {p1[:70]}...")

            try:
                response = generate(
                    model,
                    tokenizer,
                    p1,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                if verbose:
                    print(f"  -> {len(response)} chars: {response[:120]}...")

                row: dict = {
                    "p1_id": str(rec.get("p1_id", i)),
                    "p1": p1,
                    "p2": rec.get("p2", ""),
                    "category": rec.get("category", ""),
                    "dpo_generated_response": response,
                    "model": "llama31_dpo_qwen_vs_baseline",
                }
                pid = str(rec.get("p1_id", i))
                if pid in p1_baseline_map:
                    row["p1_baseline_output"] = p1_baseline_map[pid]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                done += 1

            except Exception as e:
                print(f"  ERROR row {i}: {e}")
                fout.write(json.dumps({
                    "p1_id": str(rec.get("p1_id", i)),
                    "p1": p1,
                    "error": str(e),
                }) + "\n")
                fout.flush()
                errors += 1

    print(f"\nDone. Successful={done}  Errors={errors}  Output={out_path}")


if __name__ == "__main__":
    main()
