#!/usr/bin/env python3
"""
Prepare LinguaSafe unseen test set + O1 baseline JSONL for evaluation.

Reads:
  - output/deepseek/p2_llama8b/baseline_outputs.jsonl  (Llama 8B answering P2 rewrites)
  - data/sft_scratchpad/train_sft_sharegpt.jsonl        (SFT training split)
  - data/sft_scratchpad/val_sft_sharegpt.jsonl          (SFT validation split)

Writes:
  - data/linguasafe_test_unseen.jsonl          → SFT/DPO inference input (p1 prompts only)
  - output/linguasafe_evaluation/o1_llama8b_p2_baseline.jsonl → O1 for pairwise eval

Note on experimental setup:
  O1 = Llama 8B responding to the *safe rewrite* P2 (what the model can say when
       the intent is made explicit and benign).
  O2 = SFT/DPO model responding directly to the original *unsafe* P1.

  This tests whether fine-tuning allows the model to answer P1 as helpfully and
  safely as the baseline answers P2 — without needing the explicit rewrite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEEPSEEK_BASELINE = Path("output/deepseek/p2_llama8b/baseline_outputs.jsonl")
SFT_TRAIN = Path("data/sft_scratchpad/train_sft_sharegpt.jsonl")
SFT_VAL = Path("data/sft_scratchpad/val_sft_sharegpt.jsonl")
UNSEEN_OUT = Path("data/linguasafe_test_unseen.jsonl")
O1_OUT = Path("output/linguasafe_evaluation/o1_llama8b_p2_baseline.jsonl")


def read_multiline_jsonl(path: Path) -> list[dict]:
    """
    Robustly reads files where each JSON object may span multiple lines,
    separated by the pattern }\n{ between records.
    """
    text = path.read_text(encoding="utf-8")
    rows: list[dict] = []
    # Try each line as a complete JSON first (standard JSONL)
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if rows:
        return rows
    # Fallback: split on }\n{ boundaries
    for chunk in text.split("\n}\n{"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.startswith("{"):
            chunk = "{" + chunk
        if not chunk.endswith("}"):
            chunk = chunk + "}"
        try:
            rows.append(json.loads(chunk))
        except json.JSONDecodeError:
            pass
    return rows


def load_sft_prompts(*paths: Path) -> set[str]:
    """Extract the human turn (P1 text) from ShareGPT-format SFT splits."""
    prompts: set[str] = set()
    for p in paths:
        if not p.is_file():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                convos = d.get("conversations", [])
                if isinstance(convos, str):
                    convos = json.loads(convos)
                for c in convos:
                    if c.get("from") == "human":
                        prompts.add(c["value"].strip())
                        break
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare LinguaSafe unseen test set")
    ap.add_argument("--baseline-jsonl", default=str(DEEPSEEK_BASELINE))
    ap.add_argument("--sft-train", default=str(SFT_TRAIN))
    ap.add_argument("--sft-val", default=str(SFT_VAL))
    ap.add_argument("--unseen-out", default=str(UNSEEN_OUT))
    ap.add_argument("--o1-out", default=str(O1_OUT))
    args = ap.parse_args()

    print("Loading SFT training prompts …")
    sft_prompts = load_sft_prompts(Path(args.sft_train), Path(args.sft_val))
    print(f"  SFT prompts loaded: {len(sft_prompts)}")

    print(f"Loading baseline JSONL: {args.baseline_jsonl}")
    rows = read_multiline_jsonl(Path(args.baseline_jsonl))
    print(f"  Total rows: {len(rows)}")

    unseen = [r for r in rows if r.get("p1", "").strip() not in sft_prompts]
    print(f"  Unseen (not in SFT training): {len(unseen)}")

    # Write test input JSONL (for SFT/DPO inference)
    unseen_path = Path(args.unseen_out)
    unseen_path.parent.mkdir(parents=True, exist_ok=True)
    with unseen_path.open("w", encoding="utf-8") as f:
        for r in unseen:
            row = {
                "p1_id": r.get("p1_id", r.get("id", "")),
                "p1": r.get("p1", ""),
                "p2": r.get("p2", ""),
                "category": r.get("category", r.get("type", "")),
                "lang": r.get("lang", "en"),
                "level": r.get("level", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(unseen)} rows → {unseen_path}")

    # Write O1 baseline JSONL (for pairwise evaluation)
    o1_path = Path(args.o1_out)
    o1_path.parent.mkdir(parents=True, exist_ok=True)
    with o1_path.open("w", encoding="utf-8") as f:
        for r in unseen:
            row = {
                "p1_id": r.get("p1_id", r.get("id", "")),
                "p1": r.get("p1", ""),
                "p2": r.get("p2", ""),
                "category": r.get("category", r.get("type", "")),
                "lang": r.get("lang", "en"),
                "level": r.get("level", ""),
                "output": r.get("output", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(unseen)} rows → {o1_path}")


if __name__ == "__main__":
    main()
