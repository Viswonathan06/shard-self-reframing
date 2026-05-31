#!/usr/bin/env python3
"""
Sample N examples per (helpfulness_winner, harmlessness_winner) combination from
eval_metrics_flat.jsonl (O1 / O2 / Tie on each axis → 9 buckets).

Each output row is the flat record plus:
  - combo_helpfulness_harmlessness: e.g. "O1|Tie"
  - sample_slot: 0..N-1 within the combo bucket
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

WINNERS = ("O1", "O2", "Tie")


def _combo_key(h: Any, m: Any) -> Tuple[str, str] | None:
    if h not in WINNERS or m not in WINNERS:
        return None
    return (h, m)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--flat-jsonl",
        required=True,
        help="eval_metrics_flat.jsonl from the evaluation run.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Path to write sampled JSONL.",
    )
    ap.add_argument(
        "--n-per-combo",
        type=int,
        default=5,
        help="Max rows to sample per (helpfulness_winner, harmlessness_winner).",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    flat_path = Path(args.flat_jsonl)
    if not flat_path.is_file():
        print(f"ERROR: not found: {flat_path}", file=sys.stderr)
        sys.exit(1)

    buckets: Dict[Tuple[str, str], List[Dict]] = {k: [] for k in [(a, b) for a in WINNERS for b in WINNERS]}
    skipped = 0
    with open(flat_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ck = _combo_key(row.get("helpfulness_winner"), row.get("harmlessness_winner"))
            if ck is None:
                skipped += 1
                continue
            buckets[ck].append(row)

    rng = random.Random(args.seed)
    out_rows: List[Dict] = []
    for h_win, m_win in [(a, b) for a in WINNERS for b in WINNERS]:
        pool = buckets[(h_win, m_win)]
        k = min(args.n_per_combo, len(pool))
        chosen = rng.sample(pool, k) if k else []
        for slot, rec in enumerate(chosen):
            out = dict(rec)
            out["combo_helpfulness_harmlessness"] = f"{h_win}|{m_win}"
            out["sample_slot"] = slot
            out_rows.append(out)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = {f"{a}|{b}": len(buckets[(a, b)]) for a in WINNERS for b in WINNERS}
    sampled = {f"{a}|{b}": min(args.n_per_combo, len(buckets[(a, b)])) for a in WINNERS for b in WINNERS}
    print(json.dumps({"total_flat_rows_used": sum(len(v) for v in buckets.values()), "skipped_bad_winner": skipped, "pool_counts": counts, "sampled_per_combo": sampled, "wrote": str(out_path), "n_out": len(out_rows)}, indent=2))


if __name__ == "__main__":
    main()
