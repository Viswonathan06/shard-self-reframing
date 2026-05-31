#!/usr/bin/env python3
"""
Build an expanded SFT preferred dataset from baseline_vs_guidelines eval_results.

For each P1, collect every model whose O2 was judged more helpful than O1.
Keep the response from the largest qualifying model (70B > 27B > 9B > 8B > 7B).
Discard P1s where no model's O2 beats the baseline.

Use --exclude-model to drop the model being fine-tuned (don't train a model on
its own outputs).

Output: a JSONL file with one row per P1, ready for prepare_sft_dpo_data.py.

Usage:
  python scripts/build_sft_preferred.py \
      --output data/sft_clean_preferred_expanded.jsonl \
      --exclude-model qwen35_9b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Priority order: largest model first. Each entry is (model_key, size_B, eval_results_path).
MODEL_PRIORITY = [
    ("llama33_70b", 70,
     "output/Linguasafe_Experiments/evaluation/"
     "baseline_vs_guidelines_refinement_llama33_70b_gemma4_judge/eval_results.jsonl"),
    ("qwen35_27b", 27,
     "output/Linguasafe_Experiments/evaluation/"
     "baseline_vs_guidelines_refinement_qwen35_27b_gemma4_judge/eval_results.jsonl"),
    ("qwen35_9b", 9,
     "output/Linguasafe_Experiments/evaluation/"
     "baseline_vs_guidelines_refinement_qwen35_9b_gemma4_judge/eval_results.jsonl"),
    ("llama8b", 8,
     "output/Linguasafe_Experiments/evaluation/"
     "baseline_vs_guidelines_refinement_llama8b_gemma4_judge/eval_results.jsonl"),
    ("mistral_7b", 7,
     "output/Linguasafe_Experiments/evaluation/"
     "baseline_vs_guidelines_refinement_mistral_7b_gemma4_judge/eval_results.jsonl"),
]


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(data: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  wrote {len(data):>5} rows → {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/sft_clean_preferred_expanded.jsonl",
                    help="Output JSONL path")
    ap.add_argument("--exclude-model", default=None,
                    help="Model key to exclude (the one being SFT'd, e.g. qwen35_9b)")
    ap.add_argument("--linguasafe", default="dataset/linguasafe_train.jsonl",
                    help="Path to linguasafe_train.jsonl for level/type lookup")
    args = ap.parse_args()

    exclude = args.exclude_model
    if exclude:
        print(f"Excluding model: {exclude}")

    # Load level/type metadata from linguasafe_train
    meta: dict[str, dict] = {}
    try:
        for row in load_jsonl(args.linguasafe):
            meta[str(row["id"])] = {"level": row.get("level"), "type": row.get("type"),
                                     "subtype": row.get("subtype", "")}
    except FileNotFoundError:
        print(f"Warning: {args.linguasafe} not found — level/type will be missing")

    # For each p1_id, find the best qualifying O2 (largest model where help winner = O2)
    # We process models in priority order and stop at the first winner per p1.
    best: dict[str, dict] = {}          # p1_id → chosen row
    model_tally: dict[str, int] = {}    # how many p1s each model contributed

    active_models = [(key, size, path) for key, size, path in MODEL_PRIORITY
                     if key != exclude]
    print(f"Model priority (after exclusion): {[m[0] for m in active_models]}")

    # Load all models into memory grouped by p1_id, then pick best per p1
    # (process in priority order — first qualifying model wins)
    claimed: set[str] = set()
    for model_key, model_size, path in active_models:
        try:
            rows = load_jsonl(path)
        except FileNotFoundError:
            print(f"  Warning: {path} not found, skipping {model_key}")
            continue

        contributed = 0
        for row in rows:
            pid = str(row["p1_id"])
            if pid in claimed:
                continue
            if row["helpfulness"].get("winner") != "O2":
                continue
            # O2 wins helpfulness for this model — claim it
            m = meta.get(pid, {})
            best[pid] = {
                "p1_id":       pid,
                "p1":          row["p1"],
                "p2":          row["p2"],
                "o1":          row["o1"],
                "o2":          row["o2"],
                "helpfulness": row["helpfulness"],
                "harmlessness": row["harmlessness"],
                "_chosen":     "O2",
                "_filter":     "expanded",
                "_model":      model_key,
                "_model_size": model_size,
                "level":       m.get("level"),
                "type":        m.get("type"),
                "subtype":     m.get("subtype", ""),
            }
            claimed.add(pid)
            contributed += 1

        model_tally[model_key] = contributed
        print(f"  {model_key:>14}: {contributed:>4} p1s claimed  (cumulative: {len(claimed)})")

    result = list(best.values())
    print(f"\nTotal P1s with at least one qualifying O2: {len(result)}")

    # Collect discarded P1s (no model's O2 beat baseline) — natural heldout candidates
    all_eval_ids: set[str] = set()
    for _, _, path in active_models:
        try:
            for row in load_jsonl(path):
                all_eval_ids.add(str(row["p1_id"]))
        except FileNotFoundError:
            pass
    discarded_ids = all_eval_ids - claimed
    discarded = []
    for pid in discarded_ids:
        m = meta.get(pid, {})
        discarded.append({
            "p1_id": pid,
            "level": m.get("level"),
            "type":  m.get("type"),
        })
    print(f"P1s discarded (no model's O2 beat baseline): {len(discarded)}  "
          f"→ saved as heldout candidates")

    from collections import Counter
    lvl = Counter(r["level"] for r in result)
    typ = Counter(r["type"]  for r in result)
    mod = Counter(r["_model"] for r in result)
    print(f"\nContributing model breakdown: {dict(mod)}")
    print(f"Level dist: {dict(sorted((k, v) for k, v in lvl.items() if k is not None))}")
    print(f"Type dist:  {dict(typ.most_common())}")

    write_jsonl(result, args.output)

    # Write discarded P1s alongside the main output as heldout candidates
    heldout_path = str(Path(args.output).with_name(
        Path(args.output).stem + "_heldout_candidates.jsonl"
    ))
    write_jsonl(discarded, heldout_path)


if __name__ == "__main__":
    main()
