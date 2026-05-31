#!/usr/bin/env python3
"""
Statistical summaries across all Gemma judge eval_results.jsonl runs.

For each experiment directory (containing eval_results.jsonl):
  - Helpfulness / harmlessness: O1 vs O2 vs Tie counts, O2 win rate, Wilson 95% CI,
    exact binomial test vs p0=0.5 on non-tie subset (did O2 beat O1 more than chance?).
  - Relevance: p1 vs p2 vs equal counts and rates (from <more_relevant_to> tags).

Optional explicit paired comparison (same p1_id in both runs):
  --compare DIR1 DIR2
  → McNemar on helpfulness (discordant pairs), ties excluded; stored in paired_mcnemar_helpfulness.

By default, the script also runs McNemar for same-model pairs where O1 is baseline in both:
  baseline_vs_benign_intent_* vs baseline_vs_benign_intent_guidelines_*
  baseline_vs_refinement_* vs baseline_vs_refinement_guidelines_*
  (stored in paired_mcnemar_default_pairs). Use --no-default-mcnemar to skip.

Examples:
  python scripts/eval_stats_all.py
  python scripts/eval_stats_all.py --eval-root output/DNA_New_Experiments/evaluation
  python scripts/eval_stats_all.py --compare \\
      output/DNA_New_Experiments/evaluation/baseline_vs_benign_intent_llama8b_gemma4_judge \\
      output/DNA_New_Experiments/evaluation/baseline_vs_benign_intent_guidelines_llama8b_gemma4_judge \\
      --metric helpfulness
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from scipy.stats import binomtest
except ImportError:
    binomtest = None  # type: ignore


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion k/n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    rad = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return ((centre - rad) / denom, (centre + rad) / denom)


def binomial_two_sided_p_normal(k: int, n: int, p0: float = 0.5) -> float:
    """Two-sided normal approximation (large n); returns p in [0,1]."""
    if n <= 0:
        return float("nan")
    phat = k / n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se <= 0:
        return 0.0 if abs(phat - p0) < 1e-12 else 1.0
    z = (phat - p0) / se
    # Phi(|z|) via erf
    phi = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, 2 * (1.0 - phi))))


def binomial_two_sided_p(k: int, n: int, p0: float = 0.5) -> Optional[float]:
    if n <= 0:
        return None
    if binomtest is not None:
        return float(binomtest(k, n, p=p0, alternative="two-sided").pvalue)
    return binomial_two_sided_p_normal(k, n, p0)


def parse_relevance_bucket(raw: str) -> str:
    t = (raw or "").lower()
    if "<more_relevant_to>p1</more_relevant_to>" in t:
        return "p1"
    if "<more_relevant_to>p2</more_relevant_to>" in t:
        return "p2"
    if "<more_relevant_to>equal</more_relevant_to>" in t:
        return "equal"
    m = re.search(r"<more_relevant_to>\s*([^<]+)\s*</more_relevant_to>", t)
    if m:
        return m.group(1).strip().lower()
    return "unknown"


def winner_triplet(block: Optional[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Returns (o1_wins, o2_wins, ties) for one comparison block."""
    if not block or not isinstance(block, dict):
        return (0, 0, 0)
    w = (block.get("winner") or "").strip()
    if w == "O1":
        return (1, 0, 0)
    if w == "O2":
        return (0, 1, 0)
    if w == "Tie":
        return (0, 0, 1)
    return (0, 0, 0)


def load_eval_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_comparative(
    name: str, o1: int, o2: int, ties: int
) -> Dict[str, Any]:
    n = o1 + o2 + ties
    n_decisive = o1 + o2
    o2_rate_all = o2 / n if n else float("nan")
    o2_rate_decisive = o2 / n_decisive if n_decisive else float("nan")
    lo, hi = wilson_ci(o2, n_decisive) if n_decisive else (float("nan"), float("nan"))
    p_binom = binomial_two_sided_p(o2, n_decisive, 0.5)
    return {
        "metric": name,
        "n_total": n,
        "o1_wins": o1,
        "o2_wins": o2,
        "ties": ties,
        "o2_win_rate_overall": o2_rate_all,
        "o2_win_rate_given_decisive": o2_rate_decisive,
        "wilson_95ci_o2_given_decisive": [lo, hi],
        "binom_pvalue_h0_p05_decisive": p_binom,
    }


def summarize_relevance(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    p1 = p2 = eq = unk = 0
    for r in rows:
        rel = r.get("relevance") or {}
        raw = rel.get("raw") or rel.get("reasoning") or ""
        b = parse_relevance_bucket(str(raw))
        if b == "p1":
            p1 += 1
        elif b == "p2":
            p2 += 1
        elif b == "equal":
            eq += 1
        else:
            unk += 1
    n = p1 + p2 + eq + unk
    return {
        "metric": "relevance",
        "n_total": n,
        "p1_more_relevant": p1,
        "p2_more_relevant": p2,
        "equal": eq,
        "unknown": unk,
        "p1_rate": p1 / n if n else float("nan"),
        "p2_rate": p2 / n if n else float("nan"),
        "equal_rate": eq / n if n else float("nan"),
        "wilson_95ci_p2_vs_decisive_p1p2": wilson_ci(p2, p1 + p2) if (p1 + p2) else (float("nan"), float("nan")),
        "binom_pvalue_h0_p05_p2_among_p1p2": binomial_two_sided_p(p2, p1 + p2, 0.5),
    }


def eval_one_results_file(path: Path) -> Dict[str, Any]:
    rows = load_eval_rows(path)
    h1 = h2 = ht = 0
    s1 = s2 = st = 0
    for r in rows:
        a, b, c = winner_triplet(r.get("helpfulness"))
        h1, h2, ht = h1 + a, h2 + b, ht + c
        a, b, c = winner_triplet(r.get("harmlessness"))
        s1, s2, st = s1 + a, s2 + b, st + c

    rel_sum = summarize_relevance(rows)
    return {
        "eval_results_path": str(path),
        "experiment_dir": str(path.parent.name),
        "n_rows": len(rows),
        "helpfulness": summarize_comparative("helpfulness", h1, h2, ht),
        "harmlessness": summarize_comparative("harmlessness", s1, s2, st),
        "relevance": rel_sum,
    }


def helpfulness_o2_indicator(r: Dict[str, Any]) -> Optional[int]:
    """1 = O2 wins, 0 = O1 wins, None = tie / missing."""
    w = (r.get("helpfulness") or {}).get("winner")
    if w == "O2":
        return 1
    if w == "O1":
        return 0
    return None


def mcnemar_pair(rows_a: List[Dict], rows_b: List[Dict]) -> Dict[str, Any]:
    map_a = {str(r.get("p1_id")): r for r in rows_a if r.get("p1_id") is not None}
    map_b = {str(r.get("p1_id")): r for r in rows_b if r.get("p1_id") is not None}
    common = sorted(set(map_a) & set(map_b))
    both_o2 = both_o1 = a2_b1 = a1_b2 = 0
    skipped_tie = 0
    for pid in common:
        ya = helpfulness_o2_indicator(map_a[pid])
        yb = helpfulness_o2_indicator(map_b[pid])
        if ya is None or yb is None:
            skipped_tie += 1
            continue
        if ya == 1 and yb == 1:
            both_o2 += 1
        elif ya == 0 and yb == 0:
            both_o1 += 1
        elif ya == 1 and yb == 0:
            a2_b1 += 1
        else:
            a1_b2 += 1
    discordant = a2_b1 + a1_b2
    # McNemar (continuity correction): chi2 = (|b-c|-1)^2 / (b+c)
    if discordant > 0:
        chi2 = (abs(a2_b1 - a1_b2) - 1) ** 2 / discordant
        p_chi2 = 1.0 - _chi2_sf(chi2, 1)
    else:
        chi2 = float("nan")
        p_chi2 = float("nan")
    p_exact = None
    if binomtest is not None and discordant > 0:
        lo = min(a2_b1, a1_b2)
        p_exact = float(binomtest(lo, discordant, 0.5, alternative="two-sided").pvalue)
    return {
        "n_common_ids": len(common),
        "used_pairs_decisive_both_runs": both_o2 + both_o1 + a2_b1 + a1_b2,
        "skipped_tie_either_run": skipped_tie,
        "concordant_o2_both": both_o2,
        "concordant_o1_both": both_o1,
        "discordant_o2_A_o1_B": a2_b1,
        "discordant_o1_A_o2_B": a1_b2,
        "mcnemar_chi2_statistic": chi2,
        "mcnemar_chi2_pvalue_df1": p_chi2,
        "mcnemar_exact_binom_pvalue": p_exact,
    }


def _chi2_sf(x: float, df: int) -> float:
    """Survival function for chi-square with df=1 (no scipy.stats dependency)."""
    # P(Chi^2_1 > x) = 2 * (1 - Phi(sqrt(x)))
    if x <= 0:
        return 1.0
    z = math.sqrt(x)
    # Phi(z) approx via error function
    phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return max(0.0, min(1.0, 2 * (1.0 - phi)))


def default_mcnemar_pair_specs(by_exp_name: Dict[str, Path]) -> List[Tuple[str, str, str]]:
    """
    Return (label, dir_a_name, dir_b_name) for McNemar where both dirs exist.
    A = plain treatment vs baseline; B = guidelines treatment vs baseline (same tag).
    """
    out: List[Tuple[str, str, str]] = []
    for name, path in sorted(by_exp_name.items()):
        if name.startswith("baseline_vs_benign_intent_guidelines_"):
            continue
        if name.startswith("baseline_vs_benign_intent_"):
            tag = name[len("baseline_vs_benign_intent_") :]
            other = f"baseline_vs_benign_intent_guidelines_{tag}"
            if other in by_exp_name:
                out.append(("benign_intent_vs_benign_intent_guidelines", name, other))
            continue
        if name.startswith("baseline_vs_refinement_guidelines_"):
            continue
        if name.startswith("baseline_vs_refinement_"):
            tag = name[len("baseline_vs_refinement_") :]
            other = f"baseline_vs_refinement_guidelines_{tag}"
            if other in by_exp_name:
                out.append(("refinement_vs_refinement_guidelines", name, other))
    return out


def run_default_mcnemar_pairs(by_exp_name: Dict[str, Path]) -> List[Dict[str, Any]]:
    """McNemar for each discovered (A,B) pair with eval_results.jsonl in both dirs."""
    rows_out: List[Dict[str, Any]] = []
    for label, a_name, b_name in default_mcnemar_pair_specs(by_exp_name):
        pa, pb = by_exp_name[a_name], by_exp_name[b_name]
        fa, fb = pa / "eval_results.jsonl", pb / "eval_results.jsonl"
        if not fa.is_file() or not fb.is_file():
            continue
        block: Dict[str, Any] = {
            "comparison_label": label,
            "dir_a": str(pa),
            "dir_b": str(pb),
            "experiment_a": a_name,
            "experiment_b": b_name,
            **mcnemar_pair(load_eval_rows(fa), load_eval_rows(fb)),
        }
        rows_out.append(block)
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stats across eval_results.jsonl files")
    ap.add_argument(
        "--eval-root",
        nargs="*",
        default=[
            "output/DNA_New_Experiments/evaluation",
            "output/sft_evaluation",
            "output/dpo_evaluation",
        ],
        help="Root dirs to search for */eval_results.jsonl",
    )
    ap.add_argument(
        "--out-json",
        default="reports/eval_stats_all.json",
        help="Write full JSON report here",
    )
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("DIR_A", "DIR_B"),
        help="Two experiment dirs (each containing eval_results.jsonl) for McNemar helpfulness",
    )
    ap.add_argument(
        "--no-default-mcnemar",
        action="store_true",
        help="Do not run automatic McNemar pairs (intent vs intent+guidelines; refinement vs refinement+guidelines)",
    )
    args = ap.parse_args()

    if binomtest is None:
        print("Warning: scipy not found; install scipy for binomial p-values.", file=sys.stderr)

    roots = [Path(r) for r in args.eval_root]
    files: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        files.extend(sorted(root.glob("*/eval_results.jsonl")))

    if not files:
        print("No eval_results.jsonl found under:", args.eval_root)
        sys.exit(1)

    report: Dict[str, Any] = {
        "eval_roots": [str(r) for r in roots],
        "n_files": len(files),
        "per_experiment": [],
        "scipy_available": binomtest is not None,
    }

    for f in files:
        try:
            report["per_experiment"].append(eval_one_results_file(f))
        except Exception as e:
            report["per_experiment"].append({"eval_results_path": str(f), "error": str(e)})

    by_exp_name: Dict[str, Path] = {f.parent.name: f.parent for f in files}
    if not args.no_default_mcnemar:
        report["paired_mcnemar_default_pairs"] = run_default_mcnemar_pairs(by_exp_name)
    else:
        report["paired_mcnemar_default_pairs"] = []

    if args.compare:
        da, db = Path(args.compare[0]), Path(args.compare[1])
        fa, fb = da / "eval_results.jsonl", db / "eval_results.jsonl"
        if not fa.is_file() or not fb.is_file():
            print("Compare paths need eval_results.jsonl in each dir.", file=sys.stderr)
            sys.exit(1)
        report["paired_mcnemar_helpfulness"] = {
            "dir_a": str(da),
            "dir_b": str(db),
            **mcnemar_pair(load_eval_rows(fa), load_eval_rows(fb)),
        }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path} ({len(files)} experiments)")

    # Compact table: experiment, O2 helpful % (decisive), CI, p vs 0.5
    print("\nHelpfulness (O2 win rate among non-ties; H0 p=0.5 two-sided)")
    print("-" * 100)
    for block in report["per_experiment"]:
        if "error" in block:
            print(f"ERR {block.get('eval_results_path')}: {block['error']}")
            continue
        h = block["helpfulness"]
        lo, hi = h["wilson_95ci_o2_given_decisive"]
        p = h.get("binom_pvalue_h0_p05_decisive")
        pstr = f"{p:.4g}" if p is not None else "n/a"
        print(
            f"{block['experiment_dir'][:52]:52}  "
            f"n={h['n_total']:4}  O2%={100*h['o2_win_rate_given_decisive']:5.1f}  "
            f"95%CI[{100*lo:4.1f},{100*hi:4.1f}]  p={pstr}"
        )

    if report.get("paired_mcnemar_default_pairs"):
        print("\nDefault paired McNemar (helpfulness, ties excluded; same p1_id)")
        for m in report["paired_mcnemar_default_pairs"]:
            p = m.get("mcnemar_exact_binom_pvalue")
            pch = m.get("mcnemar_chi2_pvalue_df1")
            pstr = f"{p:.4g}" if p is not None else f"{pch:.4g}" if pch == pch else "n/a"
            print(
                f"  [{m['comparison_label']}] {m['experiment_a'][:40]:40} vs {m['experiment_b'][:40]:40}  "
                f"n={m['used_pairs_decisive_both_runs']:4}  p={pstr}"
            )

    if "paired_mcnemar_helpfulness" in report:
        m = report["paired_mcnemar_helpfulness"]
        print("\nCustom --compare McNemar (helpfulness, ties excluded)")
        print(json.dumps(m, indent=2, ensure_ascii=False))
    elif not args.no_default_mcnemar and not report.get("paired_mcnemar_default_pairs"):
        print(
            "\n(No default McNemar pairs matched directory names under eval roots; "
            "use --compare DIR_A DIR_B for an arbitrary pair.)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
