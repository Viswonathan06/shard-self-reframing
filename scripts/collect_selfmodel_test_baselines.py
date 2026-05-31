#!/usr/bin/env python3
"""
Collect per-model baseline outputs for the self-model SFT test splits.

For each model, pulls baseline outputs from the helpful_assistant_multimodel
directories (DNA + LinguaSafe) and matches them to the model's own test split
at output/SFT/self_model/{model}/test.jsonl.

Output: output/SFT/self_model/{model}/modelwise_test_baselines/{model}.jsonl

Usage:
  python scripts/collect_selfmodel_test_baselines.py --models llama8b qwen35_9b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET_BASELINE_DIRS = {
    "dna":        ROOT / "output/DNA_Per_Model_Experiments/helpful_assistant_multimodel",
    "linguasafe": ROOT / "output/Linguasafe_Experiments/helpful_assistant_multimodel",
}

MODEL_DIR_MAP = {
    "llama8b":    "llama8b",
    "llama70b":   "llama33_70b",
    "qwen35_9b":  "qwen35_9b",
    "qwen35_27b": "qwen35_27b",
    "mistral7b":  "mistral_7b",
    "mistral24b": "mistral_24b",
}

SUPPORTED_MODELS = list(MODEL_DIR_MAP.keys())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def process_model(model: str) -> None:
    model_dir = MODEL_DIR_MAP[model]
    self_model_dir = ROOT / "output" / "SFT" / "self_model" / model
    test_path = self_model_dir / "test.jsonl"

    if not test_path.exists():
        print(f"  SKIP {model}: test split not found at {test_path}")
        return

    test_rows = load_jsonl(test_path)
    # Group needed p1_ids by dataset
    needed: dict[str, set] = {"dna": set(), "linguasafe": set()}
    for r in test_rows:
        needed[r["dataset"]].add(str(r["p1_id"]))

    print(f"\n=== {model} — test set: {len(test_rows)} examples ===")

    # Load baseline outputs from both datasets
    baseline_by_id: dict[str, dict] = {}
    for dataset, base_dir in DATASET_BASELINE_DIRS.items():
        path = base_dir / model_dir / "p1_baseline_outputs.jsonl"
        if not path.exists():
            print(f"  WARNING: missing {path}")
            continue
        rows = load_jsonl(path)
        found = 0
        for r in rows:
            pid = str(r["p1_id"])
            if pid in needed[dataset]:
                baseline_by_id[pid] = r
                found += 1
        print(f"  {dataset}: {len(rows)} total → {found} matched to test set")

    # Build output records
    out_rows = []
    missing = 0
    for r in test_rows:
        pid = str(r["p1_id"])
        b = baseline_by_id.get(pid)
        if b is None:
            missing += 1
            continue
        out_rows.append({
            "p1_id":          pid,
            "dataset":        r["dataset"],
            "p1":             r["p1"],
            "baseline_output": b.get("baseline_output") or b.get("o1") or b.get("output", ""),
            "level":          r.get("level"),
            "category":       r.get("category"),
            "lang":           r.get("lang"),
        })

    out_dir = self_model_dir / "modelwise_test_baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model}.jsonl"
    with out_path.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  Matched: {len(out_rows)} / {len(test_rows)}  Missing: {missing}")
    print(f"  Wrote → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+", default=["llama8b", "qwen35_9b"],
        choices=SUPPORTED_MODELS,
    )
    args = ap.parse_args()
    for model in args.models:
        process_model(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
