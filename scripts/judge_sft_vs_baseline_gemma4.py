#!/usr/bin/env python3
"""
Compare SFT heldout responses vs baseline responses using Gemma4 as judge.

Evaluates both harmlessness and helpfulness using the existing judge prompts.
Randomly assigns SFT/baseline to A/B positions to avoid position bias.

Usage:
  python scripts/judge_sft_vs_baseline_gemma4.py \
      --sft-responses  output/SFT/sft_Llama-3.1-8B-Instruct_13198/eval/heldout_responses.jsonl \
      --baseline       output/Linguasafe_Experiments/helpful_assistant_multimodel/llama8b/p1_baseline_outputs.jsonl \
      --output-dir     output/SFT/sft_Llama-3.1-8B-Instruct_13198/eval/judge_vs_baseline \
      --judge-model    /path/to/gemma4 \
      --model-tag      llama8b_sft
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tqdm import tqdm


PROMPT_DIR = Path(__file__).resolve().parent.parent / "src" / "judge_prompts"


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def parse_verdict(text: str) -> str | None:
    m = re.search(r"\[\[([ABC])\]\]", text or "")
    if m:
        c = m.group(1).upper()
        return "A" if c == "A" else ("B" if c == "B" else "Tie")
    return None


def build_prompt(template: str, question: str, resp_a: str, resp_b: str) -> str:
    # harmfulness template uses {response_a}/{response_b}, helpfulness uses {answer_a}/{answer_b}
    return (
        template
        .replace("{question}", question)
        .replace("{response_a}", resp_a)
        .replace("{response_b}", resp_b)
        .replace("{answer_a}", resp_a)
        .replace("{answer_b}", resp_b)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-responses", required=True)
    ap.add_argument("--baseline",      required=True, help="p1_baseline_outputs.jsonl")
    ap.add_argument("--output-dir",    required=True)
    ap.add_argument("--judge-model",   required=True)
    ap.add_argument("--model-tag",     default="sft_model")
    ap.add_argument("--sft-label",     default="SFT",      help="Label for sft-responses in output (e.g. large_model)")
    ap.add_argument("--baseline-label", default="Baseline", help="Label for baseline in output (e.g. self_model)")
    ap.add_argument("--batch-size",    type=int,   default=8)
    ap.add_argument("--max-new-tokens", type=int,  default=256)
    ap.add_argument("--temperature",   type=float, default=0.1)
    ap.add_argument("--max-records",        type=int,   default=None)
    ap.add_argument("--max-response-chars", type=int,   default=4000,
                    help="Truncate each response to this many chars before judging")
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--resume",             action="store_true",
                    help="Skip p1_ids already in output file and append new results")
    args = ap.parse_args()

    random.seed(args.seed)

    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as e:
        sys.exit(f"Missing dep: {e}")

    # --- Load & join data ---
    sft_rows = load_jsonl(args.sft_responses)
    base_rows = load_jsonl(args.baseline)

    def norm_id(v) -> str:
        try:
            return str(int(float(str(v))))
        except (ValueError, TypeError):
            return str(v)

    # baseline indexed by p1_id (stored as string or float — normalise)
    base_by_id = {norm_id(r["p1_id"]): r for r in base_rows}
    # fallback index by normalised p1 text (handles cross-dataset p1_id collisions)
    base_by_p1 = {re.sub(r"\s+", " ", r["p1"]).strip(): r for r in base_rows}

    joined = []
    id_matched, p1_matched = 0, 0
    for r in sft_rows:
        pid = norm_id(r["p1_id"])
        p1_key = re.sub(r"\s+", " ", r["p1"]).strip()
        if pid in base_by_id and re.sub(r"\s+", " ", base_by_id[pid]["p1"]).strip() == p1_key:
            joined.append((r, base_by_id[pid]))
            id_matched += 1
        elif p1_key in base_by_p1:
            joined.append((r, base_by_p1[p1_key]))
            p1_matched += 1

    print(f"SFT rows: {len(sft_rows)} | Baseline rows: {len(base_rows)} | Matched: {len(joined)} (by id: {id_matched}, by p1 text: {p1_matched})")
    if not joined:
        sys.exit("ERROR: No matching records between SFT and baseline. Check p1_id formats and p1 text.")

    if args.max_records:
        joined = joined[: args.max_records]

    # Resume: skip already-judged p1_ids
    out_dir = Path(args.output_dir)
    out_file = out_dir / f"sft_vs_baseline_{args.model_tag}.jsonl"
    existing_results = []
    if args.resume and out_file.exists():
        existing_results = load_jsonl(str(out_file))
        done_ids = {norm_id(r["p1_id"]) for r in existing_results}
        before = len(joined)
        joined = [(s, b) for s, b in joined if norm_id(s["p1_id"]) not in done_ids]
        print(f"Resume: skipping {before - len(joined)} already-judged rows, {len(joined)} remaining")
        if not joined:
            print("All rows already judged — nothing to do.")
            sys.exit(0)

    harm_tmpl = load_prompt("eval_harmfulness.txt")
    help_tmpl = load_prompt("eval_helpfulness.txt")

    print(f"Loading judge: {args.judge_model}")
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=int(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE", "4")),
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
        max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "4096")),
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)

    def to_vllm_prompt(system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def split_template(template: str) -> tuple[str, str]:
        """Return (system_text, user_text) by splitting on [User Question]."""
        if "[System]" in template and "[User Question]" in template:
            system_part, user_part = template.split("[User Question]", 1)
            return system_part.replace("[System]", "").strip(), "[User Question]" + user_part
        return "", template

    harm_system, harm_user_tmpl = split_template(harm_tmpl)
    help_system, help_user_tmpl = split_template(help_tmpl)

    # --- Judge ---
    results = []
    batches = list(range(0, len(joined), args.batch_size))

    for i in tqdm(batches, desc="Judging", unit="batch"):
        batch = joined[i : i + args.batch_size]

        harm_prompts, help_prompts, assignments = [], [], []
        for sft_r, base_r in batch:
            sft_resp  = sft_r.get("response") or sft_r.get("sft_output") or sft_r.get("sft_generated_response") or ""
            base_resp = base_r.get("output") or base_r.get("baseline_output") or base_r.get("dpo_generated_response") or base_r.get("response") or base_r.get("sft_output") or ""
            question  = sft_r["p1"]
            # Truncate to keep only the final answer — strips runaway thinking blobs
            sft_resp  = sft_resp[:args.max_response_chars]
            base_resp = base_resp[:args.max_response_chars]

            # Randomise position
            if random.random() < 0.5:
                a, b, sft_pos = sft_resp, base_resp, "A"
            else:
                a, b, sft_pos = base_resp, sft_resp, "B"
            assignments.append(sft_pos)

            harm_prompts.append(to_vllm_prompt(
                harm_system,
                build_prompt(harm_user_tmpl, question, a, b),
            ))
            help_prompts.append(to_vllm_prompt(
                help_system,
                build_prompt(help_user_tmpl, question, a, b),
            ))

        harm_outs = llm.generate(harm_prompts, sampling)
        help_outs = llm.generate(help_prompts, sampling)

        for (sft_r, base_r), sft_pos, harm_out, help_out in zip(
            batch, assignments, harm_outs, help_outs
        ):
            harm_verdict_raw = parse_verdict(harm_out.outputs[0].text)
            help_verdict_raw = parse_verdict(help_out.outputs[0].text)

            def map_verdict(raw, sft_pos):
                if raw is None:
                    return None
                if raw == "Tie":
                    return "Tie"
                if raw == sft_pos:
                    return args.sft_label
                return args.baseline_label

            results.append({
                "p1_id":             sft_r.get("p1_id"),
                "p1":                sft_r["p1"],
                "level":             sft_r.get("level"),
                "type":              sft_r.get("type"),
                "sft_response":      sft_r.get("response") or sft_r.get("sft_output") or "",
                "baseline_response": base_r.get("output") or base_r.get("baseline_output") or base_r.get("response") or base_r.get("sft_output") or "",
                "sft_position":      sft_pos,
                # harmlessness: who is MORE harmful — for SFT "win" we want baseline more harmful
                "harm_verdict_raw":  harm_verdict_raw,
                "harm_winner":       map_verdict(harm_verdict_raw, sft_pos),
                "harm_judge_text":   harm_out.outputs[0].text,
                # helpfulness: who is MORE helpful
                "help_verdict_raw":  help_verdict_raw,
                "help_winner":       map_verdict(help_verdict_raw, sft_pos),
                "help_judge_text":   help_out.outputs[0].text,
            })

    # --- Write ---
    out_dir.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if (args.resume and existing_results) else "w"
    with open(out_file, write_mode, encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Combine existing + new for summary stats
    all_results = existing_results + results

    # --- Stats ---
    valid_harm = [r for r in all_results if r["harm_winner"] is not None]
    valid_help = [r for r in all_results if r["help_winner"] is not None]

    def win_stats(rows, key):
        counts = defaultdict(int)
        for r in rows:
            counts[r[key]] += 1
        total = len(rows)
        return {k: (v, f"{100*v/total:.1f}%") for k, v in counts.items()}

    print(f"\n{'='*55}")
    print(f"Results: {args.model_tag}  ({len(all_results)} comparisons, {len(results)} new)")
    print(f"\nHARMLESSNESS (who is LESS harmful — higher SFT = better):")
    for k, (n, pct) in win_stats(valid_harm, "harm_winner").items():
        label = f"← {args.sft_label} safer" if k == args.baseline_label else (f"← {args.sft_label} more harmful" if k == args.sft_label else "")
        print(f"  {k:10s}: {n:4d}  {pct}  {label}")
    print(f"\nHELPFULNESS (who is MORE helpful — higher {args.sft_label} = better):")
    for k, (n, pct) in win_stats(valid_help, "help_winner").items():
        label = f"← {args.sft_label} wins" if k == args.sft_label else (f"← {args.baseline_label} wins" if k == args.baseline_label else "")
        print(f"  {k:10s}: {n:4d}  {pct}  {label}")

    # By level breakdown
    levels = sorted({r["level"] for r in all_results if r.get("level") is not None})
    if levels:
        print(f"\nBy risk level:")
        for lv in levels:
            grp_harm = [r for r in valid_harm if r.get("level") == lv]
            grp_help = [r for r in valid_help if r.get("level") == lv]
            if not grp_harm:
                continue
            sft_safer = sum(1 for r in grp_harm if r["harm_winner"] == args.baseline_label)
            sft_helpful = sum(1 for r in grp_help if r["help_winner"] == args.sft_label)
            print(f"  Level {lv} (n={len(grp_harm)}): "
                  f"SFT safer={sft_safer}({100*sft_safer/len(grp_harm):.0f}%)  "
                  f"SFT helpful={sft_helpful}({100*sft_helpful/len(grp_help):.0f}%)")

    # Save summary
    summary = {
        "model_tag":    args.model_tag,
        "n_compared":   len(all_results),
        "harmlessness": win_stats(valid_harm, "harm_winner"),
        "helpfulness":  win_stats(valid_help, "help_winner"),
    }
    (out_dir / f"summary_{args.model_tag}.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nResults → {out_file}")
    print(f"Summary → {out_dir}/summary_{args.model_tag}.json")


if __name__ == "__main__":
    main()
