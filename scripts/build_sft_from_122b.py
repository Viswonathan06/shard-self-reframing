#!/usr/bin/env python3
"""
Build SFT datasets from qwen35_122b refinement outputs (teacher distillation).

qwen35_122b is used as the sole teacher: its refined response (O2) is the
training target for non-reasoning models, and its thinking + O2 is the target
for reasoning models (Qwen3.5-9B, Qwen3.5-27B, etc.).

Output layout (--output-dir):
  sft_nonreasoning/
    train.jsonl   -- {messages: [system, user(P1), assistant(O2)]}
    val.jsonl
  sft_reasoning/
    train.jsonl   -- {messages: [system, user(P1), assistant(<think>...O2)]}
    val.jsonl
  stats.json

Records with null output or empty thinking are excluded from the reasoning split
(they hit the max_tokens limit mid-think and have no clean response).

Usage:
  python scripts/build_sft_from_122b.py \\
      --input output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_multimodel/qwen35_122b/baseline_outputs.jsonl \\
      --output-dir output/SFT/qwen35_122b_teacher \\
      --val-frac 0.1 \\
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a helpful and safe assistant. When a user asks about a sensitive "
    "topic, respond helpfully while avoiding harm. Always address the user's "
    "underlying need even if you cannot address the exact phrasing of their request."
)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  wrote {len(data):>5} rows → {path}")


def make_messages(p1: str, assistant_content: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": p1},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def stratified_split(
    rows: list, val_frac: float, test_frac: float, seed: int
) -> tuple[list, list, list]:
    """Stratify by 'level', then split into train/val/test preserving distribution."""
    from collections import defaultdict
    rng = random.Random(seed)
    by_level: dict[str, list] = defaultdict(list)
    for r in rows:
        by_level[r.get("level", "unknown")].append(r)

    train, val, test = [], [], []
    for level, group in sorted(by_level.items()):
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * test_frac))
        n_val  = max(1, round(n * val_frac))
        test  += group[:n_test]
        val   += group[n_test: n_test + n_val]
        train += group[n_test + n_val:]

    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_multimodel/qwen35_122b/baseline_outputs.jsonl",
    )
    ap.add_argument("--output-dir", default="output/SFT/qwen35_122b_teacher")
    ap.add_argument("--val-frac",  type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} records from {args.input}")

    out_dir = Path(args.output_dir)

    nonreasoning_rows = []
    reasoning_rows = []
    skipped_null = 0

    for r in records:
        p1 = (r.get("p1") or "").strip()
        output = (r.get("output") or "").strip()
        thinking = (r.get("thinking") or "").strip()

        if not p1 or not output or not thinking:
            skipped_null += 1
            continue

        pid = r.get("p1_id", "")
        category = r.get("category", "")
        level = r.get("level")

        meta = {"p1_id": pid, "category": category, "level": level}

        # Non-reasoning: just P1 → O2
        nonreasoning_rows.append({**make_messages(p1, output), **meta})

        # Reasoning: P1 → <think>[thinking]</think>\n\nO2
        reasoning_target = f"<think>\n{thinking}\n</think>\n\n{output}"
        reasoning_rows.append({**make_messages(p1, reasoning_target), **meta})

    print(f"Clean records: {len(nonreasoning_rows)} rows  (skipped: {skipped_null})")

    # Stratified train/val/test splits
    nr_train, nr_val, nr_test = stratified_split(
        nonreasoning_rows, args.val_frac, args.test_frac, args.seed
    )
    r_train, r_val, r_test = stratified_split(
        reasoning_rows, args.val_frac, args.test_frac, args.seed
    )

    write_jsonl(nr_train, out_dir / "sft_nonreasoning" / "train.jsonl")
    write_jsonl(nr_val,   out_dir / "sft_nonreasoning" / "val.jsonl")
    write_jsonl(nr_test,  out_dir / "sft_nonreasoning" / "test.jsonl")
    write_jsonl(r_train,  out_dir / "sft_reasoning"    / "train.jsonl")
    write_jsonl(r_val,    out_dir / "sft_reasoning"    / "val.jsonl")
    write_jsonl(r_test,   out_dir / "sft_reasoning"    / "test.jsonl")

    # Level distribution in test set
    from collections import Counter
    test_levels = Counter(r.get("level") for r in nr_test)

    stats = {
        "source": args.input,
        "total_records": len(records),
        "skipped_null_or_no_thinking": skipped_null,
        "clean_records": len(nonreasoning_rows),
        "nonreasoning": {"train": len(nr_train), "val": len(nr_val), "test": len(nr_test)},
        "reasoning":    {"train": len(r_train),  "val": len(r_val),  "test": len(r_test)},
        "test_level_distribution": dict(sorted(test_levels.items())),
    }
    stats_path = out_dir / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nStats → {stats_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
