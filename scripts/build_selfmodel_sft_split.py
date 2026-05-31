#!/usr/bin/env python3
"""
Build per-model self-generated SFT splits.

For each model, select the instances where the model's own refinement (O2) won
the helpfulness evaluation (baseline_vs_refinement_<model>_gemma4_judge), then
produce stratified train/val/test splits identical in structure to the combined
qwen35_122b teacher splits.

Sources:
  DNA:        output/DNA_New_Experiments/evaluation/
                  baseline_vs_refinement_{model}_gemma4_judge/eval_results.jsonl
  LinguaSafe: output/Linguasafe_Experiments/evaluation/
                  baseline_vs_refinement_{model}_gemma4_judge/eval_results.jsonl

Filter:  helpfulness.winner == "O2"
Stratify by: (dataset, harm_level)

Output (output/SFT/self_model/{model}/):
  all.jsonl        -- full filtered pool with metadata
  train.jsonl      -- 72 % stratified  (80% × 90%)
  val.jsonl        -- 8 %              (80% × 10%)
  test.jsonl       -- 20 % stratified
  split_stats.json -- per-stratum counts

Usage:
  python scripts/build_selfmodel_sft_split.py --models llama8b qwen35_9b
  python scripts/build_selfmodel_sft_split.py   # all supported models
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
TEST_FRAC = 0.20
VAL_FRAC_OF_TRAIN = 0.10

ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_MODELS = ["llama8b", "llama70b", "qwen35_9b", "qwen35_27b", "mistral7b", "mistral24b", "phi4",
                    "qwen35_9b_thinking", "qwen35_27b_thinking"]

# Map script model tag → directory name used in evaluation paths
MODEL_DIR_MAP = {
    "llama8b":              "llama8b",
    "llama70b":             "llama33_70b",
    "qwen35_9b":            "qwen35_9b",
    "qwen35_27b":           "qwen35_27b",
    "mistral7b":            "mistral_7b",
    "mistral24b":           "mistral_24b",
    "phi4":                 "phi4",
    "qwen35_9b_thinking":   "qwen35_9b_thinking",
    "qwen35_27b_thinking":  "qwen35_27b_thinking",
}


def eval_path(dataset: str, model_dir: str, model: str) -> Path:
    if dataset == "dna":
        return (ROOT / "output/DNA_Per_Model_Experiments/evaluation"
                / f"refinement_from_baseline_multimodel_{model_dir}_gemma4_judge/eval_results.jsonl")
    return (ROOT / "output/Linguasafe_Experiments/evaluation"
            / f"baseline_vs_refinement_{model_dir}_gemma4_judge/eval_results.jsonl")


def meta_path(dataset: str, model_dir: str, model: str) -> Path:
    if dataset == "dna":
        return (ROOT / "output/DNA_Per_Model_Experiments/refinement_from_baseline_multimodel"
                / f"{model_dir}/refinement_input.jsonl")
    return (ROOT / "output/Linguasafe_Experiments"
            / f"linguasafe_refinement_from_baseline_multimodel/{model_dir}/refinement_input.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_pool(model: str, model_dir: str) -> list[dict]:
    pool: list[dict] = []
    for dataset in ("dna", "linguasafe"):
        ep = eval_path(dataset, model_dir, model)
        mp = meta_path(dataset, model_dir, model)
        if not ep.exists():
            print(f"  SKIP {dataset}: eval file not found: {ep}")
            continue
        eval_rows = load_jsonl(ep)
        meta_by_id = {}
        if mp.exists():
            for r in load_jsonl(mp):
                meta_by_id[str(r["p1_id"])] = r

        o2_wins = [r for r in eval_rows if r["helpfulness"].get("winner") == "O2"]
        for r in o2_wins:
            pid = str(r["p1_id"])
            m = meta_by_id.get(pid, {})
            pool.append({
                "p1_id":     pid,
                "dataset":   dataset,
                "p1":        r["p1"],
                "p2":        r.get("p2", ""),
                "o1":        r["o1"],
                "o2":        r["o2"],
                "level":     m.get("level"),
                "category":  m.get("category"),
                "lang":      m.get("lang"),
                "judge_helpfulness_winner":    r["helpfulness"].get("winner"),
                "judge_helpfulness_reasoning": r["helpfulness"].get("reasoning", ""),
                "judge_harmlessness_winner":   r["harmlessness"].get("winner"),
            })
        print(f"  {dataset}: {len(eval_rows)} total → {len(o2_wins)} O2 wins")
    return pool


def stratified_split(pool: list[dict], frac: float, seed: int):
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in pool:
        strata[(r["dataset"], r["level"])].append(r)
    a, b = [], []
    for key, items in sorted(strata.items()):
        rng.shuffle(items)
        n = max(1, round(len(items) * frac))
        b.extend(items[:n])
        a.extend(items[n:])
    return a, b


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_model(model: str) -> None:
    model_dir = MODEL_DIR_MAP[model]
    out_dir = ROOT / "output" / "SFT" / "self_model" / model
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {model} (dir={model_dir}) ===")
    pool = build_pool(model, model_dir)
    if not pool:
        print("  No O2 wins found — skipping.")
        return
    print(f"  Combined pool: {len(pool)} examples")

    dataset_counts = Counter(r["dataset"] for r in pool)
    level_counts = Counter((r["dataset"], r["level"]) for r in pool)
    print(f"  By dataset: {dict(dataset_counts)}")

    train_val, test = stratified_split(pool, TEST_FRAC, SEED)
    train, val = stratified_split(train_val, VAL_FRAC_OF_TRAIN, SEED + 1)
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    write_jsonl(out_dir / "all.jsonl", pool)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)

    stats = {
        "model": model,
        "seed": SEED,
        "test_frac": TEST_FRAC,
        "val_frac_of_train": VAL_FRAC_OF_TRAIN,
        "total": len(pool),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "by_dataset": {k: v for k, v in sorted(dataset_counts.items())},
        "by_stratum": {
            f"{ds}_level{lv}": {
                "total": n,
                "train": sum(1 for r in train if r["dataset"] == ds and r["level"] == lv),
                "val":   sum(1 for r in val   if r["dataset"] == ds and r["level"] == lv),
                "test":  sum(1 for r in test  if r["dataset"] == ds and r["level"] == lv),
            }
            for (ds, lv), n in sorted(level_counts.items())
        },
    }
    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  Wrote → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+", default=["llama8b", "qwen35_9b"],
        choices=SUPPORTED_MODELS,
        help="Which models to build splits for (default: llama8b qwen35_9b)",
    )
    args = ap.parse_args()

    for model in args.models:
        process_model(model)

    print("\nDone.")


if __name__ == "__main__":
    main()
