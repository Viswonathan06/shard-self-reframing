#!/usr/bin/env python3
"""
Rebuild eval_summary.json, eval_metrics_flat.jsonl, and eval_reasoning_by_prompt.jsonl
from an existing eval_results.jsonl (e.g. after a partial run or prompt change).

Requires the same O1/O2 (or combined input) JSONL used for the original eval so
flat rows can include p1, p2, o1, o2 text.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

# Repo root on PYTHONPATH
from src.baselines.baseline_common import load_jsonl
from src.evaluation_ranking.evaluate_safety_helpfulness import (
    compute_summary,
    write_derived_exports,
    load_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory containing eval_results.jsonl (writes summary + flat + reasoning here).",
    )
    parser.add_argument(
        "--eval-results",
        default=None,
        help="Path to eval_results.jsonl (default: <output-dir>/eval_results.jsonl).",
    )
    parser.add_argument("--input-jsonl", default=None, help="Combined records with o1, o2, p1, p2.")
    parser.add_argument("--o2-jsonl", default=None, help="O2 outputs JSONL (with p1_id, p1, p2, output).")
    parser.add_argument("--o1-jsonl", default=None, help="O1 baseline JSONL (p1_id, output).")
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    results_path = Path(args.eval_results) if args.eval_results else out / "eval_results.jsonl"
    if not results_path.is_file():
        print(f"ERROR: missing results file: {results_path}", file=sys.stderr)
        sys.exit(1)

    if not args.input_jsonl and not (args.o2_jsonl and args.o1_jsonl):
        print(
            "ERROR: provide either --input-jsonl or both --o2-jsonl and --o1-jsonl "
            "(same as the evaluate job) so flat exports include p1/p2/o1/o2.",
            file=sys.stderr,
        )
        sys.exit(1)

    ns = Namespace(
        input_jsonl=args.input_jsonl,
        o2_jsonl=args.o2_jsonl,
        o1_jsonl=args.o1_jsonl,
    )
    records = load_records(ns)
    results = load_jsonl(str(results_path))

    if not results:
        print("ERROR: eval_results is empty.", file=sys.stderr)
        sys.exit(1)

    summary = compute_summary(results)
    summary_file = out / "eval_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    write_derived_exports(results, records, out)

    print(f"Wrote {summary_file}")
    print(f"Wrote {out / 'eval_metrics_flat.jsonl'}")
    print(f"Wrote {out / 'eval_reasoning_by_prompt.jsonl'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
