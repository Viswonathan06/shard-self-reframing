#!/usr/bin/env python3
"""
Build val-only evaluation inputs for SFT.

Outputs:
  1) val prompt file for inference
  2) val baseline file (same rows) for O1 in pairwise eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_val_prompts(val_sharegpt: Path) -> set[str]:
    prompts: set[str] = set()
    with val_sharegpt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for msg in row.get("conversations", []):
                if msg.get("from") == "human":
                    prompts.add((msg.get("value") or "").strip())
                    break
    return prompts


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare val-only eval files")
    p.add_argument(
        "--baseline-jsonl",
        default="output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/llama8b/baseline_outputs.jsonl",
    )
    p.add_argument("--val-sharegpt", default="data/sft_scratchpad/val_sft_sharegpt.jsonl")
    p.add_argument("--val-input-out", default="data/sft_val_only_prompts.jsonl")
    p.add_argument("--val-baseline-out", default="output/sft_evaluation/o1_baseline_val_only.jsonl")
    args = p.parse_args()

    val_prompts = get_val_prompts(Path(args.val_sharegpt))
    baseline_rows = read_jsonl(Path(args.baseline_jsonl))

    val_rows = [r for r in baseline_rows if (r.get("p1") or "").strip() in val_prompts]

    Path(args.val_input_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_baseline_out).parent.mkdir(parents=True, exist_ok=True)

    with Path(args.val_input_out).open("w", encoding="utf-8") as f_in:
        for r in val_rows:
            f_in.write(json.dumps(r, ensure_ascii=False) + "\n")

    with Path(args.val_baseline_out).open("w", encoding="utf-8") as f_base:
        for r in val_rows:
            f_base.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"val prompts in split: {len(val_prompts)}")
    print(f"val rows matched in baseline: {len(val_rows)}")
    print(f"wrote: {args.val_input_out}")
    print(f"wrote: {args.val_baseline_out}")


if __name__ == "__main__":
    main()
