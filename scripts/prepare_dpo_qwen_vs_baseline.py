#!/usr/bin/env python3
"""
Build a DPO JSONL for Llama 3.1 8B: chosen = Qwen 480B teacher O2, rejected = Llama 8B P1 baseline.

Prompt format matches SFT/DNA P1-only style:
  ### Human: {p1}\\n\\n### Assistant:

Each output line:
  {"p1_id", "p1", "p2", "category", "prompt", "chosen", "rejected"}

Defaults:
  --qwen-jsonl  output/qwen480b_scratchpad_llama8b_full.jsonl  (uses generated_o2)
  --baseline-jsonl  output/DNA_New_Experiments/dna_baseline_multimodel/llama8b/p1_baseline_outputs.jsonl  (uses output)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def human_assistant_prefix(p1: str) -> str:
    return f"### Human: {p1}\n\n### Assistant: "


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--qwen-jsonl",
        default="output/qwen480b_scratchpad_llama8b_full.jsonl",
        help="Qwen teacher file with generated_o2",
    )
    ap.add_argument(
        "--baseline-jsonl",
        default="output/DNA_New_Experiments/dna_baseline_multimodel/llama8b/p1_baseline_outputs.jsonl",
        help="Llama 8B P1 baseline file with output",
    )
    ap.add_argument(
        "--output-jsonl",
        default="data/dpo/qwen_chosen_vs_llama8b_rejected.jsonl",
        help="Output DPO dataset JSONL",
    )
    args = ap.parse_args()

    qwen_path = Path(args.qwen_jsonl)
    base_path = Path(args.baseline_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_by_id: dict[str, dict] = {}
    for r in load_jsonl(base_path):
        pid = str(r.get("p1_id", "")).strip()
        if pid:
            baseline_by_id[pid] = r

    n_in = 0
    n_out = 0
    n_skip_no_baseline = 0
    n_skip_empty = 0
    n_skip_identical = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for r in load_jsonl(qwen_path):
            n_in += 1
            pid = str(r.get("p1_id", "")).strip()
            if not pid or pid not in baseline_by_id:
                n_skip_no_baseline += 1
                continue

            p1 = (r.get("p1") or baseline_by_id[pid].get("p1") or "").strip()
            p2 = (r.get("p2") or baseline_by_id[pid].get("p2") or "").strip()
            category = (r.get("category") or baseline_by_id[pid].get("category") or "").strip()

            chosen = strip_code_fences(r.get("generated_o2") or "")
            rejected = strip_code_fences(baseline_by_id[pid].get("output") or "")

            if not chosen or not rejected:
                n_skip_empty += 1
                continue
            if chosen == rejected:
                n_skip_identical += 1
                continue

            row = {
                "p1_id": pid,
                "p1": p1,
                "p2": p2,
                "category": category,
                "prompt": human_assistant_prefix(p1),
                "chosen": chosen,
                "rejected": rejected,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Qwen rows read:        {n_in}")
    print(f"DPO pairs written:     {n_out}")
    print(f"Skipped (no baseline): {n_skip_no_baseline}")
    print(f"Skipped (empty text):    {n_skip_empty}")
    print(f"Skipped (identical):     {n_skip_identical}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
