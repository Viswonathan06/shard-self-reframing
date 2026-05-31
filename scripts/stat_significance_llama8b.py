"""
Statistical significance tests for Llama8b: SFT vs DPO vs Rational SFT vs Baseline.

Method:
  - Split the 498 test comparisons into 3 equal parts (~166 each).
  - For each part compute helpfulness win rate for the method and for baseline.
  - Run a paired t-test (scipy) across the 3 split win rates.
  - Reports p-values for each method vs baseline and for each pair of methods.
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

ROOT = Path("$PROJECT_ROOT")

# ── Load judge results ─────────────────────────────────────────────────────────

def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def load_sft(path):
    """Named-field format: help_winner in {SFT, Baseline, Tie}."""
    records = load_jsonl(path)
    return [
        {"p1_id": r["p1_id"],
         "method_win": int(r["help_winner"] == "SFT"),
         "baseline_win": int(r["help_winner"] == "Baseline")}
        for r in records
    ]


def load_dpo(path):
    """o1/o2 format: o1=Baseline, o2=DPO. winner in {O1, O2, Tie}."""
    records = load_jsonl(path)
    return [
        {"p1_id": r["p1_id"],
         "method_win": int(r["helpfulness"]["winner"] == "O2"),
         "baseline_win": int(r["helpfulness"]["winner"] == "O1")}
        for r in records
    ]


data = {
    "SFT": load_sft(
        ROOT / "output/SFT/self_model/llama8b/judge/llama8b/sft_vs_baseline_llama8b.jsonl"
    ),
    "DPO": load_dpo(
        ROOT / "output/dpo_evaluation/dpo_vs_baseline_llama8b_gemma4_judge/eval_results.jsonl"
    ),
    "Rational SFT": load_sft(
        ROOT / "output/SFT/rational_sft/llama8b/judge/llama8b/sft_vs_baseline_llama8b_rational_sft.jsonl"
    ),
}

# ── Split into 3 equal parts and compute win rates ────────────────────────────

N_SPLITS = 3

def split_win_rates(records, key):
    n = len(records)
    size = n // N_SPLITS
    rates = []
    for i in range(N_SPLITS):
        chunk = records[i * size: (i + 1) * size]
        rates.append(np.mean([r[key] for r in chunk]))
    return np.array(rates)


print(f"{'Method':<14}  {'n':>4}  {'Split win rates (method)':<36}  {'Split win rates (baseline)'}")
print("-" * 85)

win_rates = {}
baseline_rates = {}
for name, records in data.items():
    wr = split_win_rates(records, "method_win")
    br = split_win_rates(records, "baseline_win")
    win_rates[name] = wr
    baseline_rates[name] = br
    print(f"{name:<14}  {len(records):>4}  {str(np.round(wr,3)):<36}  {np.round(br,3)}")

# ── T-tests: each method vs its own baseline rates ───────────────────────────

print()
print("=== Method vs Baseline (paired t-test across 3 splits) ===")
print(f"{'Method':<14}  {'Mean method WR':>14}  {'Mean baseline WR':>16}  {'t':>7}  {'p':>8}")
print("-" * 70)

for name in data:
    wr = win_rates[name]
    br = baseline_rates[name]
    t, p = stats.ttest_rel(wr, br)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"{name:<14}  {wr.mean():>14.3f}  {br.mean():>16.3f}  {t:>7.3f}  {p:>8.4f}  {sig}")

# ── T-tests: pairwise between methods (on method win rates) ──────────────────

print()
print("=== Pairwise method comparisons (paired t-test across 3 splits) ===")
print(f"{'Comparison':<24}  {'Mean A':>7}  {'Mean B':>7}  {'t':>7}  {'p':>8}")
print("-" * 60)

methods = list(data.keys())
for i in range(len(methods)):
    for j in range(i + 1, len(methods)):
        a, b = methods[i], methods[j]
        t, p = stats.ttest_rel(win_rates[a], win_rates[b])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        label = f"{a} vs {b}"
        print(f"{label:<24}  {win_rates[a].mean():>7.3f}  {win_rates[b].mean():>7.3f}  {t:>7.3f}  {p:>8.4f}  {sig}")
