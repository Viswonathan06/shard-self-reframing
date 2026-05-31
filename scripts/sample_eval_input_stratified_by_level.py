#!/usr/bin/env python3
"""
Build an --input-jsonl subset for evaluate_safety_helpfulness.py by stratifying on
harm ``level`` (from O2 rows, same field as linguasafe / P2 JSONL).

Strategies
----------
  equal        — as close to n_total / n_levels per level as possible (default).
  proportional — sample in proportion to each level's pool size (then fix rounding
                 so the written row count sums to exactly ``--n-total``).

Requires O1 and O2 JSONL with matching ``p1_id``; O2 rows must include ``level``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def merge_pairs(o2_rows: List[Dict], o1_by_id: Dict[str, Dict]) -> Tuple[Dict[int, List[Dict]], int]:
    """Group merged records by integer level. Returns (by_level, skipped_missing_o1)."""
    by_level: Dict[int, List[Dict]] = defaultdict(list)
    skipped = 0
    for r2 in o2_rows:
        pid = str(r2.get("p1_id", r2.get("id", "")))
        if not pid:
            continue
        r1 = o1_by_id.get(pid)
        if not r1:
            skipped += 1
            continue
        if "level" not in r2 and r2.get("level") is None:
            continue
        try:
            lvl = int(r2["level"])
        except (TypeError, ValueError):
            continue
        rec = {
            "p1_id": pid,
            "p1": r2.get("p1", r1.get("p1", "")),
            "p2": r2.get("p2", ""),
            "o1": r1.get("output", ""),
            "o2": r2.get("output", ""),
            "level": lvl,
            "category": r2.get("category", r1.get("category", "")),
            "lang": r2.get("lang", r1.get("lang", "")),
        }
        if not rec["o1"] or not rec["o2"]:
            continue
        by_level[lvl].append(rec)
    return by_level, skipped


def allocate_equal(n_total: int, levels: List[int], pool_sizes: Dict[int, int]) -> Dict[int, int]:
    """Target counts per level; may exceed pool for some levels — caller clips."""
    if not levels:
        return {}
    n_levels = len(levels)
    base = n_total // n_levels
    rem = n_total % n_levels
    targets = {lv: base for lv in levels}
    # Spread remainder across levels with largest pools first (more headroom).
    order = sorted(levels, key=lambda lv: -pool_sizes[lv])
    for i in range(rem):
        targets[order[i % n_levels]] += 1
    return targets


def allocate_proportional(n_total: int, levels: List[int], pool_sizes: Dict[int, int]) -> Dict[int, int]:
    """Hamilton / largest-remainder so targets sum to ``n_total`` (pools assumed large enough)."""
    total_pool = sum(pool_sizes[lv] for lv in levels)
    if total_pool == 0:
        return {lv: 0 for lv in levels}
    targets: Dict[int, int] = {}
    fractions: List[Tuple[float, int]] = []
    floor_sum = 0
    for lv in levels:
        raw = n_total * pool_sizes[lv] / total_pool
        fl = int(raw)
        targets[lv] = fl
        floor_sum += fl
        fractions.append((raw - fl, lv))
    rem = n_total - floor_sum
    fractions.sort(reverse=True)
    for i in range(rem):
        targets[fractions[i][1]] += 1
    return targets


def sample_from_level(pool: List[Dict], k: int, rng: random.Random) -> List[Dict]:
    k = min(k, len(pool))
    if k <= 0:
        return []
    return rng.sample(pool, k)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--o2-jsonl", required=True, help="Controlled / O2 JSONL (must have level, p1_id, output).")
    ap.add_argument("--o1-jsonl", required=True, help="Baseline / O1 JSONL (p1_id, output).")
    ap.add_argument("--n-total", type=int, default=500, help="Total rows to write (default: 500).")
    ap.add_argument(
        "--strategy",
        choices=("equal", "proportional"),
        default="equal",
        help="How to split n_total across levels (default: equal).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, help="Output JSONL path for --input-jsonl.")
    args = ap.parse_args()

    o2_path = Path(args.o2_jsonl)
    o1_path = Path(args.o1_jsonl)
    if not o2_path.is_file() or not o1_path.is_file():
        print("ERROR: o2 or o1 path not found.", file=sys.stderr)
        sys.exit(1)

    o1_by_id = {str(r.get("p1_id", r.get("id", ""))): r for r in load_jsonl(o1_path) if r.get("p1_id") or r.get("id")}
    by_level, skipped_o1 = merge_pairs(load_jsonl(o2_path), o1_by_id)
    levels = sorted(by_level.keys())
    if not levels:
        print("ERROR: no merged rows with level + O1.", file=sys.stderr)
        sys.exit(1)

    pool_sizes = {lv: len(by_level[lv]) for lv in levels}
    rng = random.Random(args.seed)

    if args.strategy == "equal":
        targets = allocate_equal(args.n_total, levels, pool_sizes)
    else:
        targets = allocate_proportional(args.n_total, levels, pool_sizes)

    out_rows: List[Dict] = []
    shortfall: Dict[str, Any] = {}
    for lv in levels:
        want = targets.get(lv, 0)
        got = sample_from_level(by_level[lv], want, rng)
        out_rows.extend(got)
        if len(got) < want:
            shortfall[str(lv)] = {"wanted": want, "got": len(got), "pool": pool_sizes[lv]}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_level_out: Dict[str, int] = {}
    for r in out_rows:
        k = str(r["level"])
        by_level_out[k] = by_level_out.get(k, 0) + 1

    report = {
        "wrote": str(out_path),
        "n_out": len(out_rows),
        "n_requested": args.n_total,
        "strategy": args.strategy,
        "seed": args.seed,
        "levels_in_input": levels,
        "pool_sizes": {str(k): pool_sizes[k] for k in levels},
        "counts_in_output": by_level_out,
        "skipped_no_o1": skipped_o1,
        "shortfall_by_level": shortfall,
    }
    print(json.dumps(report, indent=2))
    if len(out_rows) < args.n_total:
        print(
            f"WARNING: only {len(out_rows)} rows (requested {args.n_total}); "
            "see shortfall_by_level.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
