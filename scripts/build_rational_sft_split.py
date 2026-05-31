#!/usr/bin/env python3
"""
Build Rational-SFT training splits for Qwen 27B.

Uses the SAME train/val/test p1_id assignments as the self-model SFT splits
so that results are directly comparable (identical test set).

For each row the assistant turn is:
  <think>\n{rational_full_output}        if </think> already present (81%)
  <think>\n{rational_full_output}\n</think>   otherwise

Output: output/SFT/rational_sft/qwen35_27b/{train,val,test,all}.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF_MODEL_DIR = ROOT / "output/SFT/self_model/qwen35_27b"
OUT_DIR = ROOT / "output/SFT/rational_sft/qwen35_27b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RATIONAL_FILES = [
    ROOT / "output/rational_baseline/qwen35_27b/dna/inference_compliance.jsonl",
    ROOT / "output/rational_baseline/qwen35_27b/linguasafe/inference_compliance.jsonl",
]

SYSTEM_PROMPT = (
    "You are a helpful and safe assistant. When a user asks about a sensitive topic, "
    "respond helpfully while avoiding harm. Always address the user's underlying need "
    "even if you cannot address the exact phrasing of their request."
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def format_assistant(rational_full_output: str) -> str:
    """Keep only the reasoning chain, not the final answer.

    Qwen outputs: [reasoning chain] </think>\n\n[final answer]
    We train on:  <think>\n[reasoning chain]\n</think>
    so the model learns to reason via the SCR chain and then
    generates its own final answer at inference time.
    """
    if "</think>" in rational_full_output:
        reasoning = rational_full_output[:rational_full_output.index("</think>")]
        return "<think>\n" + reasoning.strip() + "\n</think>"
    # No delimiter found — treat the entire output as the reasoning chain
    return "<think>\n" + rational_full_output.strip() + "\n</think>"


def build_messages(p1: str, rational_full_output: str) -> list[dict]:
    return [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": p1},
        {"role": "assistant", "content": format_assistant(rational_full_output)},
    ]


def main():
    # Load all rational outputs indexed by p1_id
    rational_by_pid: dict[str, dict] = {}
    for path in RATIONAL_FILES:
        for row in load_jsonl(path):
            pid = str(row["p1_id"])
            rational_by_pid[pid] = row
    print(f"Loaded {len(rational_by_pid)} rational rows")

    # Build splits matching self-model assignments
    stats: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        self_rows = load_jsonl(SELF_MODEL_DIR / f"{split}.jsonl")
        out_rows = []
        missing = 0
        for r in self_rows:
            pid = str(r["p1_id"])
            rat = rational_by_pid.get(pid)
            if rat is None:
                missing += 1
                continue
            out_rows.append({
                "p1_id":   pid,
                "dataset": r.get("dataset", rat.get("dataset", "")),
                "level":   r.get("level",   rat.get("level")),
                "category":r.get("category",rat.get("category", "")),
                "lang":    r.get("lang",    ""),
                "p1":      r["p1"],
                "o2":      format_assistant(rat["rational_full_output"]),
                "messages": build_messages(r["p1"], rat["rational_full_output"]),
            })
        if missing:
            print(f"  WARNING: {split} missing {missing} pids in rational data")
        out_path = OUT_DIR / f"{split}.jsonl"
        with out_path.open("w") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats[split] = {"total": len(out_rows)}
        print(f"  {split}: {len(out_rows)} rows → {out_path}")

    # Also write all.jsonl (train+val+test combined)
    all_rows = []
    for split in ("train", "val", "test"):
        all_rows.extend(load_jsonl(OUT_DIR / f"{split}.jsonl"))
    with (OUT_DIR / "all.jsonl").open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_stats = {
        "model": "qwen35_27b",
        "method": "rational_sft",
        "note": "same p1_id splits as self_model/qwen35_27b for fair comparison",
        **{k: v["total"] for k, v in stats.items()},
        "total": len(all_rows),
    }
    (OUT_DIR / "split_stats.json").write_text(json.dumps(split_stats, indent=2))
    print(f"\nDone. Stats: {split_stats}")


if __name__ == "__main__":
    main()
