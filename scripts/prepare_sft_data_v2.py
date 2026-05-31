#!/usr/bin/env python3
"""
v2 of scratchpad SFT data preparation.

Differences vs prepare_sft_data.py (v1):
  * Stratified train/val split by `category` so every category appears in both
    splits (v1 produced val with only 3 categories, none of them in train).
  * Native role-based output (`messages`) so downstream training can apply the
    Llama-3.1 chat template directly. ShareGPT format is also written for
    backward-compat with anything that already consumes it.
  * Deterministic per-category shuffling under a configurable --seed.

Output directory layout:
  data/sft_scratchpad_v2/
    train_sft.jsonl              # native messages format
    val_sft.jsonl                # native messages format
    train_sft_sharegpt.jsonl     # legacy compat
    val_sft_sharegpt.jsonl       # legacy compat
    dataset_info.json
    split_report.json            # per-category counts + seed
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


def _wrap_reasoning_and_answer(scratchpad: str, answer: str, tag: str) -> str:
    t = tag.strip().lower()
    if not re.fullmatch(r"[a-z][a-z0-9_]*", t):
        raise ValueError(f"Invalid reasoning tag: {tag!r}")
    return f"<{t}>\n{scratchpad}\n</{t}>\n\n{answer}"


def stratified_split(records, split, seed):
    """Split records into (train, val) keeping the category mix balanced.

    Each category is shuffled independently with `seed`, then the first
    `round(n * split)` rows go to train (with at least 1 in each split when n>=2).
    """
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.get("category", "Unknown")].append(r)

    rng = random.Random(seed)
    train, val = [], []
    report = {}
    for cat, rows in sorted(by_cat.items()):
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        rows = [rows[i] for i in idxs]
        n = len(rows)
        n_train = round(n * split)
        if n >= 2:
            n_train = max(1, min(n - 1, n_train))   # guarantee at least 1 train + 1 val
        train.extend(rows[:n_train])
        val.extend(rows[n_train:])
        report[cat] = {"total": n, "train": n_train, "val": n - n_train}
    return train, val, report


def to_messages(record, reasoning_tag: str):
    """Native chat format (used by tokenizer.apply_chat_template)."""
    return {
        "id": f"dna_sft_{record['p1_id']}",
        "category": record.get("category", "Unknown"),
        "source": "qwen480b_scratchpad_generation",
        "messages": [
            {"role": "user", "content": record["p1"]},
            {
                "role": "assistant",
                "content": _wrap_reasoning_and_answer(
                    record["generated_scratchpad"],
                    record["generated_o2"],
                    reasoning_tag,
                ),
            },
        ],
    }


def to_sharegpt(record, reasoning_tag: str):
    """Legacy compat (matches v1 ShareGPT shape)."""
    return {
        "id": f"dna_sft_{record['p1_id']}",
        "conversations": [
            {"from": "human", "value": record["p1"]},
            {
                "from": "gpt",
                "value": _wrap_reasoning_and_answer(
                    record["generated_scratchpad"],
                    record["generated_o2"],
                    reasoning_tag,
                ),
            },
        ],
        "category": record.get("category", "Unknown"),
        "source": "qwen480b_scratchpad_generation",
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Source JSONL with p1/generated_scratchpad/generated_o2/category")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--train-split", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reasoning-tag", default="scratchpad")
    args = ap.parse_args()

    records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "error" in r:
                continue
            if not (r.get("p1") and r.get("generated_scratchpad") and r.get("generated_o2")):
                continue
            records.append(r)
    print(f"Loaded {len(records)} valid records")

    train, val, report = stratified_split(records, args.train_split, args.seed)
    print(f"Train: {len(train)}  Val: {len(val)}  (split={args.train_split}, seed={args.seed})")
    print("\nPer-category split:")
    for cat, r in sorted(report.items()):
        print(f"  {r['train']:4d} / {r['val']:4d}   {cat}")

    out = Path(args.output_dir)
    write_jsonl(out / "train_sft.jsonl", [to_messages(r, args.reasoning_tag) for r in train])
    write_jsonl(out / "val_sft.jsonl",   [to_messages(r, args.reasoning_tag) for r in val])
    write_jsonl(out / "train_sft_sharegpt.jsonl", [to_sharegpt(r, args.reasoning_tag) for r in train])
    write_jsonl(out / "val_sft_sharegpt.jsonl",   [to_sharegpt(r, args.reasoning_tag) for r in val])

    info = {
        "dataset_name": "dna_scratchpad_sft_v2",
        "description": "Stratified train/val split of DNA scratchpad SFT data",
        "source_model": "qwen3-coder-480b",
        "target_model": "llama-3.1-8b-instruct",
        "total_records": len(records),
        "train_records": len(train),
        "val_records": len(val),
        "train_split": args.train_split,
        "seed": args.seed,
        "reasoning_tag": args.reasoning_tag,
        "categories": sorted(report.keys()),
        "files": {
            "train": "train_sft.jsonl",
            "val": "val_sft.jsonl",
            "train_sharegpt": "train_sft_sharegpt.jsonl",
            "val_sharegpt": "val_sft_sharegpt.jsonl",
        },
    }
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    (out / "split_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
