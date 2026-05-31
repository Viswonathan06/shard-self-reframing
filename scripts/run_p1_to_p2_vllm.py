#!/usr/bin/env python3
"""
Batch P1→P2 safety reframing using vLLM directly (no model_client.py).

Loads vLLM once, builds all prompts, runs in batches. Supports resume.

Usage:
  python scripts/run_p1_to_p2_vllm.py \
      --model /path/to/model \
      --jsonl-path dataset/donotanswer_no_outputs.jsonl \   # mutually exclusive with --csv-path
      --csv-path  dataset/linguasafe.csv \                  #
      --language en \
      --guidelines-dir output/guidelines/guidelines_openai_one \
      --universal-guidelines src/prompts/guidelines.txt \
      --output output/DNA_New_Experiments/p2s/_work_phi4/p1_p2_outputs_openai_one.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    sys.exit(f"vLLM not found: {e}")

from transformers import AutoTokenizer
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

REWRITE_SYSTEM = (
    "Output ONLY the rewritten safe prompt text, with no preface, no apologies, "
    "no explanations, no labels, no extra quotes or markdown."
)


def sanitize_category_filename(category: str) -> str:
    name = re.sub(r"[^\w\s\-]", "", category)
    name = re.sub(r"\s+", "_", name.strip())
    return name or "unknown"


def load_universal_guidelines(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_category_guidelines(guidelines_dir: Path, category: str) -> str:
    fname = sanitize_category_filename(category) + ".txt"
    p = guidelines_dir / fname
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def build_user_prompt(template: str, universal: str, cat_guidelines: str,
                      category: str, p1: str) -> str:
    return (
        template
        .replace("{INSERT_UNIVERSAL_GUIDELINES_HERE}", universal)
        .replace("{INSERT_CATEGORY_SPECIFIC_GUIDELINES_HERE}", cat_guidelines or "(No category-specific guidelines.)")
        .replace("{INSERT_CATEGORY_NAME}", category)
        .replace("{INSERT_UNSAFE_PROMPT_HERE}", p1)
        .replace("{INSERT_PROMPT_HERE}", p1)
    )


def parse_p2(raw: str) -> str:
    """Extract the reformulated prompt from model output (JSON or plain text)."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        lines = lines[1:] if lines[0].strip().startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            p2 = obj.get("the reformulated prompt") or obj.get("p2")
            if p2:
                return str(p2).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    return s


def load_records_jsonl(path: Path, language: str) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p1_id = str(r.get("p1_id", r.get("id", "")))
        p1 = str(r.get("p1", r.get("prompt", ""))).strip()
        if not p1:
            continue
        records.append({
            "p1_id":    p1_id,
            "p1":       p1,
            "category": str(r.get("type", r.get("subtype", "Unknown"))),
            "lang":     str(r.get("lang", language)),
            "level":    r.get("level"),
        })
    return records


def load_records_csv(path: Path, language: str) -> list[dict]:
    df = pd.read_csv(path)
    df = df[df["lang"] == language].copy()
    records = []
    for _, row in df.iterrows():
        p1 = str(row.get("prompt", "")).strip()
        if not p1:
            continue
        records.append({
            "p1_id":    str(row.get("id", "")),
            "p1":       p1,
            "category": str(row.get("type", "Unknown")) if pd.notna(row.get("type")) else "Unknown",
            "lang":     str(row.get("lang", language)),
            "level":    row.get("level"),
        })
    return records


def load_done_ids(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    pid = str(r.get("p1_id", ""))
                    p2 = r.get("p2") or ""
                    if pid and str(p2).strip() and str(p2).lower() != "null":
                        done.add(pid)
                except json.JSONDecodeError:
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",                  required=True)
    ap.add_argument("--jsonl-path",             default=None)
    ap.add_argument("--csv-path",               default=None)
    ap.add_argument("--language",               default="en")
    ap.add_argument("--guidelines-dir",         required=True)
    ap.add_argument("--universal-guidelines",   default="src/prompts/guidelines.txt")
    ap.add_argument("--rewriter-prompt",        default="src/prompts/safety_rewriter_prompt.txt")
    ap.add_argument("--output",                 required=True)
    ap.add_argument("--temperature",            type=float, default=0.3)
    ap.add_argument("--max-tokens",             type=int,   default=512)
    ap.add_argument("--tensor-parallel-size",   type=int,   default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--max-model-len",          type=int,   default=4096)
    ap.add_argument("--batch-size",             type=int,   default=256)
    args = ap.parse_args()

    if not args.jsonl_path and not args.csv_path:
        ap.error("Provide --jsonl-path or --csv-path")

    guidelines_dir = ROOT / args.guidelines_dir
    universal_text = load_universal_guidelines(ROOT / args.universal_guidelines)
    template = (ROOT / args.rewriter_prompt).read_text(encoding="utf-8").strip()

    if args.jsonl_path:
        records = load_records_jsonl(Path(args.jsonl_path), args.language)
    else:
        records = load_records_csv(Path(args.csv_path), args.language)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(out_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} already done")
    pending = [r for r in records if r["p1_id"] not in done_ids]
    if not pending:
        print("All records already processed.")
        return

    print(f"Records: {len(records)} total, {len(pending)} pending")

    # Build prompts — use apply_chat_template for proper formatting
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts: list[str] = []
    for r in pending:
        cat_guidelines = load_category_guidelines(guidelines_dir, r["category"])
        user_prompt = build_user_prompt(template, universal_text, cat_guidelines,
                                        r["category"], r["p1"])
        msgs = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]
        try:
            rendered = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            rendered = f"System: {REWRITE_SYSTEM}\n\nUser: {user_prompt}\n\nAssistant:"
        prompts.append(rendered)

    example = prompts[0][:300].replace("\n", "\\n")
    print(f"Example prompt (first 300 chars): {example}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_custom_all_reduce=(args.tensor_parallel_size > 1),
    )

    stop_ids = [tokenizer.eos_token_id]
    for tok in ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)
    stop_ids = list(set(filter(None, stop_ids)))

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop_token_ids=stop_ids,
    )

    success = len(done_ids)
    batch_size = args.batch_size
    total = len(done_ids) + len(pending)

    with out_path.open("a", encoding="utf-8") as fout:
        for start in range(0, len(pending), batch_size):
            batch_recs    = pending[start : start + batch_size]
            batch_prompts = prompts[start : start + batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            for rec, out in zip(batch_recs, outputs):
                raw = (out.outputs[0].text or "").strip()
                p2  = parse_p2(raw) if raw else ""
                row = {
                    "p1_id":    rec["p1_id"],
                    "p1":       rec["p1"],
                    "p2":       p2 or None,
                    "category": rec["category"],
                    "lang":     rec["lang"],
                    "level":    rec["level"],
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                if p2:
                    success += 1
            fout.flush()
            done_so_far = len(done_ids) + start + len(batch_recs)
            print(f"  [{done_so_far}/{total}] batches done, success so far: {success}")

    print(f"Done. Wrote {len(pending)} rows → {out_path}")


if __name__ == "__main__":
    main()
