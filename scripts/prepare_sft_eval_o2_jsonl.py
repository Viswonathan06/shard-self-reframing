#!/usr/bin/env python3
"""
Build an O2 JSONL for pairwise DNA evaluation: each row mirrors baseline multimodel rows
but `output` is the SFT model's user-facing completion.

Reads output/sft_inference_results.jsonl (or --input-jsonl).

Example:
  python scripts/prepare_sft_eval_o2_jsonl.py \\
    --input-jsonl output/sft_inference_results.jsonl \\
    --output-jsonl output/sft_evaluation/sft_llama8b_outputs_for_gemma_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def extract_final_after_reasoning_tag(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    for tag in ("scratchpad", "reasoning"):
        end_tag = f"</{tag}>"
        idx = t.lower().find(end_tag)
        if idx != -1:
            return t[idx + len(end_tag) :].strip()
    m = re.match(r"^\s*<([a-zA-Z_][\w\-]*)>.*?</\1>\s*(.*)$", t, flags=re.DOTALL)
    if m:
        return (m.group(2) or "").strip()
    return t


def pick_response_text(r: dict, primary_field: str) -> str:
    """Prefer primary field; if missing/empty, fall back to sft_final_response then scratchpad strip."""
    order = [primary_field, "sft_final_response", "output"]
    seen = set()
    for key in order:
        if not key or key in seen:
            continue
        seen.add(key)
        raw = r.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    gen = r.get("sft_generated_response")
    if isinstance(gen, str) and gen.strip():
        extracted = extract_final_after_reasoning_tag(gen)
        if extracted:
            return extracted
        return gen.strip()
    return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-jsonl",
        default="output/sft_inference_results.jsonl",
        help="SFT inference results (must include p1_id, p1, sft_generated_response)",
    )
    p.add_argument(
        "--output-jsonl",
        required=True,
        help="Writer: one JSON object per line with output = SFT response",
    )
    p.add_argument(
        "--text-field",
        default="sft_generated_response",
        help="Which field holds the judged SFT text (default: sft_generated_response)",
    )
    args = p.parse_args()

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, raw in enumerate(fin, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"Invalid JSON on line {line_no} of {in_path}") from None
            pid = r.get("p1_id")
            if pid is None:
                raise SystemExit(f"Missing p1_id on line {line_no}")

            txt = pick_response_text(r, args.text_field)

            row = {
                "p1_id": pid,
                "p1": r.get("p1", ""),
                "p2": r.get("p2", ""),
                "category": r.get("category", ""),
                "output": txt,
            }
            for k in ("lang", "level"):
                if k in r:
                    row[k] = r[k]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"Wrote {n_ok} rows to {out_path}")


if __name__ == "__main__":
    main()
