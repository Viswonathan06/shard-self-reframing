#!/usr/bin/env python3
"""
Merge O2-win pairs from DNA and LinguaSafe qwen35_122b refinement evals,
then produce a stratified train/test split.

Sources:
  DNA:        output/DNA_New_Experiments/evaluation/
                  baseline_vs_refinement_qwen35_122b_gemma4_judge/eval_results.jsonl
  LinguaSafe: output/Linguasafe_Experiments/evaluation/
                  baseline_vs_refinement_qwen35_122b_gemma4_judge/eval_results.jsonl

Filter:  helpfulness.winner == "O2"
Stratify by: (dataset, harm_level)

Output (output/SFT/qwen35_122b_teacher/combined_dna_linguasafe/):
  all.jsonl        -- full filtered pool with metadata
  train.jsonl      -- 80 % stratified
  test.jsonl       -- 20 % stratified
  split_stats.json -- per-stratum counts
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

SEED = 42
TEST_FRAC = 0.20
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "SFT" / "qwen35_122b_teacher" / "combined_dna_linguasafe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "dna": {
        "eval": ROOT / "output/DNA_New_Experiments/evaluation"
                       "/baseline_vs_refinement_qwen35_122b_gemma4_judge/eval_results.jsonl",
        "meta": ROOT / "output/DNA_New_Experiments"
                       "/dna_refinement_from_baseline_multimodel/qwen35_122b/refinement_input.jsonl",
    },
    "linguasafe": {
        "eval": ROOT / "output/Linguasafe_Experiments/evaluation"
                       "/baseline_vs_refinement_qwen35_122b_gemma4_judge/eval_results.jsonl",
        "meta": ROOT / "output/Linguasafe_Experiments"
                       "/linguasafe_refinement_from_baseline_multimodel/qwen35_122b/refinement_input.jsonl",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_pool() -> list[dict]:
    pool: list[dict] = []
    for dataset, paths in SOURCES.items():
        eval_rows = load_jsonl(paths["eval"])
        meta_rows = load_jsonl(paths["meta"])
        meta_by_id = {str(r["p1_id"]): r for r in meta_rows}

        o2_wins = [r for r in eval_rows if r["helpfulness"].get("winner") == "O2"]

        for r in o2_wins:
            pid = str(r["p1_id"])
            m = meta_by_id.get(pid, {})
            pool.append({
                "p1_id":      pid,
                "dataset":    dataset,
                "p1":         r["p1"],
                "p2":         r.get("p2", ""),
                "o1":         r["o1"],
                "o2":         r["o2"],
                "level":      m.get("level"),
                "category":   m.get("category"),
                "lang":       m.get("lang"),
                "judge_helpfulness_winner":   r["helpfulness"].get("winner"),
                "judge_helpfulness_reasoning": r["helpfulness"].get("reasoning", ""),
                "judge_harmlessness_winner":  r["harmlessness"].get("winner"),
            })

        print(f"  {dataset}: {len(eval_rows)} total → {len(o2_wins)} O2 wins")

    return pool


def stratified_split(pool: list[dict], test_frac: float, seed: int):
    rng = random.Random(seed)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in pool:
        key = (r["dataset"], r["level"])
        strata[key].append(r)

    train, test = [], []
    for key, items in sorted(strata.items()):
        rng.shuffle(items)
        n_test = max(1, round(len(items) * test_frac))
        test.extend(items[:n_test])
        train.extend(items[n_test:])

    return train, test


def carve_val(train: list[dict], val_frac: float, seed: int):
    """Stratified val split carved from train; train shrinks accordingly."""
    rng = random.Random(seed + 1)
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for r in train:
        key = (r["dataset"], r["level"])
        strata[key].append(r)

    new_train, val = [], []
    for key, items in sorted(strata.items()):
        rng.shuffle(items)
        n_val = max(1, round(len(items) * val_frac))
        val.extend(items[:n_val])
        new_train.extend(items[n_val:])

    return new_train, val


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    print("Loading and filtering...")
    pool = build_pool()
    print(f"Combined pool: {len(pool)} examples")

    dataset_counts = Counter(r["dataset"] for r in pool)
    level_counts = Counter((r["dataset"], r["level"]) for r in pool)
    print("By dataset:", dict(dataset_counts))
    print("By (dataset, level):", dict(sorted(level_counts.items())))

    print("\nSplitting (stratified by dataset × level)...")
    train_and_val, test = stratified_split(pool, TEST_FRAC, SEED)
    train, val = carve_val(train_and_val, val_frac=0.10, seed=SEED)
    print(f"  train: {len(train)}  val: {len(val)}  test: {len(test)}")

    write_jsonl(OUT_DIR / "all.jsonl", pool)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    stats = {
        "seed": SEED,
        "test_frac": TEST_FRAC,
        "val_frac_of_train": 0.10,
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
    (OUT_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nWrote → {OUT_DIR}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
