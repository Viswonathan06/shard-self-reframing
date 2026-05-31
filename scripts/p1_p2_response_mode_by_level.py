#!/usr/bin/env python3
"""
Pivot P1→P2 JSONL: rows = harm level, columns = count per response_mode.

Usage:
  python scripts/p1_p2_response_mode_by_level.py \\
    --input output/p1_to_p2/.../p1_p2_outputs_openai_one.jsonl \\
    --output output/p1_to_p2/.../p1_p2_response_mode_by_level.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=str,
        required=True,
        help="p1_p2_outputs*.jsonl with level and response_mode",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV (default: <input-dir>/p1_p2_response_mode_by_level.csv)",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else input_path.parent / "p1_p2_response_mode_by_level.csv"

    rows = load_rows(input_path)
    if not rows:
        print("ERROR: no rows loaded.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    if "level" not in df.columns:
        print("ERROR: input has no 'level' column.", file=sys.stderr)
        sys.exit(1)
    if "response_mode" not in df.columns:
        print("ERROR: input has no 'response_mode' column.", file=sys.stderr)
        sys.exit(1)

    df["level"] = pd.to_numeric(df["level"], errors="coerce")
    df["response_mode"] = df["response_mode"].fillna("missing").astype(str)
    df = df.dropna(subset=["level"])
    df["level"] = df["level"].astype(int)

    pivot = pd.crosstab(df["level"], df["response_mode"], margins=False)
    pivot = pivot.sort_index()
    mode_cols = sorted(pivot.columns.astype(str))
    pivot = pivot[mode_cols]
    pivot["row_total"] = pivot.sum(axis=1)
    pivot = pivot.reset_index()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out_path, index=False)
    n_modes = len(pivot.columns) - 2  # level + row_total
    print(f"Wrote {out_path} ({len(pivot)} levels, {n_modes} response_mode columns + row_total)")


if __name__ == "__main__":
    main()
