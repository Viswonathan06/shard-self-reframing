#!/usr/bin/env python3
"""
Statistical significance tests for SFT vs baseline judge results.

Tests per condition:
  - Binomial test: is SFT win rate > 50% among decisive (non-tie) comparisons?

Tests between conditions (same base model, different supervision source):
  - Two-proportion z-test on SFT win rates (excluding ties)
  - McNemar's test on overlapping p1_ids

Usage:
  python scripts/significance_tests.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = [
    {
        "label":  "LLaMA-3.1-8B / Qwen3.5-122B supervision",
        "model":  "llama8b",
        "source": "teacher",
        "path":   ROOT / "output/SFT/qwen35_122b_teacher/combined_dna_linguasafe/judge/llama8b/sft_vs_baseline_llama8b.jsonl",
    },
    {
        "label":  "LLaMA-3.1-8B / Self-model supervision",
        "model":  "llama8b",
        "source": "self",
        "path":   ROOT / "output/SFT/self_model/llama8b/judge/llama8b/sft_vs_baseline_llama8b.jsonl",
    },
    {
        "label":  "Qwen3.5-9B / Qwen3.5-122B supervision",
        "model":  "qwen35_9b",
        "source": "teacher",
        "path":   ROOT / "output/SFT/qwen35_122b_teacher/combined_dna_linguasafe/judge/qwen35_9b/sft_vs_baseline_qwen35_9b.jsonl",
    },
    {
        "label":  "Qwen3.5-9B / Self-model supervision",
        "model":  "qwen35_9b",
        "source": "self",
        "path":   ROOT / "output/SFT/self_model/qwen35_9b/judge/qwen35_9b/sft_vs_baseline_qwen35_9b.jsonl",
    },
]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_outcomes(path: Path) -> dict[str, str]:
    """Return {p1_id: help_winner} for each row."""
    return {str(r["p1_id"]): r["help_winner"] for r in load_rows(path)}


def position_bias_test(rows: list[dict]) -> dict:
    """
    Check whether SFT win rate differs when SFT is shown as response A vs B.
    H0: P(SFT wins | pos=A) == P(SFT wins | pos=B).
    Uses a two-proportion z-test and Fisher's exact test.
    """
    pos_a = [r for r in rows if r["sft_position"] == "A"]
    pos_b = [r for r in rows if r["sft_position"] == "B"]

    def win_rate(subset):
        decisive = [r for r in subset if r["help_winner"] != "Tie"]
        wins = sum(1 for r in decisive if r["help_winner"] == "SFT")
        return wins, len(decisive)

    k_a, n_a = win_rate(pos_a)
    k_b, n_b = win_rate(pos_b)

    # Fisher's exact test on 2x2 contingency table
    # rows: position A / position B; cols: SFT wins / does not win
    table = [[k_a, n_a - k_a], [k_b, n_b - k_b]]
    _, p_fisher = stats.fisher_exact(table, alternative="two-sided")

    # Two-proportion z-test
    p_pool = (k_a + k_b) / (n_a + n_b) if (n_a + n_b) > 0 else 0
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) if (n_a and n_b) else 1
    z = (k_a / n_a - k_b / n_b) / se if se > 0 else 0.0
    p_z = 2 * stats.norm.sf(abs(z))

    return {
        "n_pos_a": len(pos_a), "wins_pos_a": k_a, "rate_pos_a": k_a / n_a if n_a else 0,
        "n_pos_b": len(pos_b), "wins_pos_b": k_b, "rate_pos_b": k_b / n_b if n_b else 0,
        "p_fisher": p_fisher,
        "p_z": p_z,
    }


def binomial_test(outcomes: dict[str, str]) -> dict:
    """One-sided binomial test: P(SFT wins > 50%) among decisive comparisons."""
    decisive = [v for v in outcomes.values() if v != "Tie"]
    n = len(decisive)
    k = sum(1 for v in decisive if v == "SFT")
    result = stats.binomtest(k, n, p=0.5, alternative="greater")
    return {
        "n_decisive": n,
        "n_sft_wins": k,
        "sft_win_rate": k / n if n else 0,
        "p_value": result.pvalue,
    }


def two_prop_ztest(outcomes_a: dict[str, str], outcomes_b: dict[str, str]) -> dict:
    """Two-proportion z-test comparing SFT win rates (among decisive) between two conditions."""
    decisive_a = [v for v in outcomes_a.values() if v != "Tie"]
    decisive_b = [v for v in outcomes_b.values() if v != "Tie"]
    k_a = sum(1 for v in decisive_a if v == "SFT")
    k_b = sum(1 for v in decisive_b if v == "SFT")
    n_a, n_b = len(decisive_a), len(decisive_b)

    p_a = k_a / n_a
    p_b = k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_a - p_b) / se if se > 0 else 0.0
    # Two-sided
    p_value = 2 * stats.norm.sf(abs(z))
    return {
        "rate_a": p_a,
        "rate_b": p_b,
        "z": z,
        "p_value": p_value,
    }


def mcnemar_test(outcomes_a: dict[str, str], outcomes_b: dict[str, str]) -> dict | None:
    """McNemar's test on overlapping p1_ids (paired comparisons)."""
    shared = set(outcomes_a) & set(outcomes_b)
    if len(shared) < 5:
        return None

    # Discordant cells: A wins / B doesn't, and vice versa
    ab = 0  # A=SFT, B=Baseline-or-Tie
    ba = 0  # A=Baseline-or-Tie, B=SFT
    for pid in shared:
        a_win = outcomes_a[pid] == "SFT"
        b_win = outcomes_b[pid] == "SFT"
        if a_win and not b_win:
            ab += 1
        elif b_win and not a_win:
            ba += 1

    n_discordant = ab + ba
    if n_discordant == 0:
        return {"shared": len(shared), "discordant": 0, "p_value": 1.0}

    result = stats.binomtest(ab, n_discordant, p=0.5, alternative="two-sided")
    return {
        "shared": len(shared),
        "discordant": n_discordant,
        "a_better": ab,
        "b_better": ba,
        "p_value": result.pvalue,
    }


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def main():
    print("=" * 70)
    print("Per-condition binomial tests (H0: SFT win rate = 50% among decisive)")
    print("=" * 70)
    all_outcomes = {}
    all_rows = {}
    for cond in CONDITIONS:
        rows = load_rows(cond["path"])
        all_rows[(cond["model"], cond["source"])] = rows
        outcomes = {str(r["p1_id"]): r["help_winner"] for r in rows}
        all_outcomes[(cond["model"], cond["source"])] = outcomes
        res = binomial_test(outcomes)
        print(f"\n{cond['label']}")
        print(f"  Decisive: {res['n_decisive']}  SFT wins: {res['n_sft_wins']}"
              f"  Rate: {res['sft_win_rate']:.1%}")
        print(f"  Binomial p = {res['p_value']:.2e}  {sig_stars(res['p_value'])}")

    print("\n" + "=" * 70)
    print("Position bias tests (H0: win rate identical when SFT is shown as A vs B)")
    print("=" * 70)
    for cond in CONDITIONS:
        rows = all_rows[(cond["model"], cond["source"])]
        pb = position_bias_test(rows)
        print(f"\n{cond['label']}")
        print(f"  SFT=A: {pb['wins_pos_a']}/{pb['n_pos_a']} wins ({pb['rate_pos_a']:.1%})"
              f"   SFT=B: {pb['wins_pos_b']}/{pb['n_pos_b']} wins ({pb['rate_pos_b']:.1%})")
        print(f"  Fisher's exact p = {pb['p_fisher']:.4f}  {sig_stars(pb['p_fisher'])}"
              f"   z-test p = {pb['p_z']:.4f}  {sig_stars(pb['p_z'])}")

    print("\n" + "=" * 70)
    print("Between-condition tests (teacher vs self-model, same base model)")
    print("=" * 70)

    for model in ("llama8b", "qwen35_9b"):
        oc_teacher = all_outcomes[(model, "teacher")]
        oc_self    = all_outcomes[(model, "self")]
        label = "LLaMA-3.1-8B" if model == "llama8b" else "Qwen3.5-9B"

        print(f"\n{label}: Qwen3.5-122B supervision vs Self-model supervision")

        zres = two_prop_ztest(oc_teacher, oc_self)
        print(f"  Two-proportion z-test:")
        print(f"    Teacher rate: {zres['rate_a']:.1%}  Self rate: {zres['rate_b']:.1%}")
        print(f"    z = {zres['z']:.3f}  p = {zres['p_value']:.4f}  {sig_stars(zres['p_value'])}")

        mres = mcnemar_test(oc_teacher, oc_self)
        if mres is None:
            print(f"  McNemar: fewer than 5 overlapping examples — skipped")
        else:
            print(f"  McNemar (on {mres['shared']} shared examples):")
            print(f"    Discordant pairs: {mres['discordant']}"
                  f"  Teacher-better: {mres.get('a_better', '?')}"
                  f"  Self-better: {mres.get('b_better', '?')}")
            print(f"    p = {mres['p_value']:.4f}  {sig_stars(mres['p_value'])}")

    print("\n* p<0.05  ** p<0.01  *** p<0.001  n.s. not significant")


if __name__ == "__main__":
    main()
