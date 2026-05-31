#!/usr/bin/env python3
"""
For each model, collect baseline outputs on the test-split examples
(from build_combined_sft_split.py) and write one file per model.

Sources:
  DNA:        output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/{model}/p1_baseline_outputs.jsonl
              (fallback: output/DNA_New_Experiments/dna_baseline_multimodel/{model}/... for qwen35_122b)
  LinguaSafe: output/Linguasafe_Experiments/helpful_assistant_multimodel/{model}/p1_baseline_outputs.jsonl

Output:
  output/SFT/qwen35_122b_teacher/combined_dna_linguasafe/modelwise_test_baselines/{model}.jsonl
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMBINED_DIR = ROOT / "output" / "SFT" / "qwen35_122b_teacher" / "combined_dna_linguasafe"
OUT_DIR = COMBINED_DIR / "modelwise_test_baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_BASELINE_DIRS = {
    "dna":        ROOT / "output/DNA_Per_Model_Experiments/helpful_assistant_multimodel",
    "linguasafe": ROOT / "output/Linguasafe_Experiments/helpful_assistant_multimodel",
}

# qwen35_122b is absent from DNA_Per_Model_Experiments; fall back to DNA_New_Experiments
DNA_FALLBACK_DIR = ROOT / "output/DNA_New_Experiments/dna_baseline_multimodel"

MODELS = [
    "llama8b",
    "llama33_70b",
    "mistral_7b",
    "mistral_24b",
    "qwen35_9b",
    "qwen35_27b",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    # Load test set and build lookup: (dataset, p1_id) → test record
    test_rows = load_jsonl(COMBINED_DIR / "test.jsonl")
    test_index: dict[tuple, dict] = {
        (r["dataset"], str(r["p1_id"])): r for r in test_rows
    }
    print(f"Test set: {len(test_rows)} examples")

    # Group test p1_ids by dataset for efficient lookup
    test_ids_by_dataset: dict[str, set] = defaultdict(set)
    for ds, pid in test_index:
        test_ids_by_dataset[ds].add(pid)

    for model in MODELS:
        records: list[dict] = []
        missing: dict[str, int] = {}

        for dataset, base_dir in DATASET_BASELINE_DIRS.items():
            path = base_dir / model / "p1_baseline_outputs.jsonl"
            if not path.exists() and dataset == "dna":
                path = DNA_FALLBACK_DIR / model / "p1_baseline_outputs.jsonl"
                if path.exists():
                    print(f"  [{model}] DNA: using fallback {path.parent.parent.name}")
            if not path.exists():
                print(f"  WARNING: missing {path}")
                continue

            baseline_rows = load_jsonl(path)
            baseline_by_id = {str(r["p1_id"]): r for r in baseline_rows}

            wanted = test_ids_by_dataset[dataset]
            found = 0
            for pid in wanted:
                b = baseline_by_id.get(pid)
                test_rec = test_index[(dataset, pid)]
                records.append({
                    "p1_id":           pid,
                    "dataset":         dataset,
                    "p1":              test_rec["p1"],
                    "baseline_output": b["output"] if b else None,
                    "level":           test_rec["level"],
                    "category":        test_rec["category"],
                    "lang":            test_rec["lang"],
                })
                if b:
                    found += 1
            missing[dataset] = len(wanted) - found

        out_path = OUT_DIR / f"{model}.jsonl"
        with out_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        miss_str = ", ".join(f"{ds}:{n}" for ds, n in missing.items() if n > 0)
        print(f"  {model}: {len(records)} records written"
              + (f"  [missing: {miss_str}]" if miss_str else ""))

    print(f"\nDone → {OUT_DIR}")


if __name__ == "__main__":
    main()
