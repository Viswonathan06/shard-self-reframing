#!/usr/bin/env python3
"""
Prepare train/val/test splits from data/sft_clean_preferred.jsonl for SFT and DPO.

Output layout (data/sft_dpo/):
  sft_train.jsonl   – SFT examples: {messages: [{role, content}...]}  (always O2 response)
  sft_val.jsonl
  dpo_train.jsonl   – DPO examples: {prompt, chosen, rejected}         (all rows)
  dpo_val.jsonl
  heldout_test.jsonl – Stratified sample from unused linguasafe pool matching train distribution

SFT target: always O2 (the safer reframed response).
  - For rows where O2 won helpfulness or harmlessness: O2 is the clear choice.
  - For rows where harm=Tie and O1 won helpfulness: harm difference is negligible,
    so we still train on O2 to keep signal consistent and teach the reframing style.

Heldout sampling: stratified by (level, type) to match the training distribution,
so evaluation is on the same content profile as training.

Usage:
  python scripts/prepare_sft_dpo_data.py \
      --input data/sft_clean_preferred.jsonl \
      --output-dir data/sft_dpo \
      --val-frac 0.1 \
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a helpful and safe assistant. When a user asks about a sensitive "
    "topic, respond helpfully while avoiding harm. Always address the user's "
    "underlying need even if you cannot address the exact phrasing of their request."
)


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(data: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  wrote {len(data):>5} rows → {path}")


def to_sft_messages(row: dict) -> dict:
    # Always train on O2 (the safer, reframed response) for consistent alignment signal.
    # For harm=Tie rows where O1 won helpfulness, harm difference is negligible,
    # so preferring O2 keeps training consistent without sacrificing safety.
    return {
        "p1_id": row["p1_id"],
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": row["p1"]},
            {"role": "assistant", "content": row["o2"]},
        ],
        "_chosen": "O2",
        "_original_chosen": row["_chosen"],
        "_filter": row["_filter"],
        "level": row.get("level"),
        "type":  row.get("type"),
    }


def to_dpo_row(row: dict) -> dict:
    chosen_key  = row["_chosen"].lower()   # "o1" or "o2"
    rejected_key = "o2" if chosen_key == "o1" else "o1"
    return {
        "p1_id":    row["p1_id"],
        "prompt":   row["p1"],
        "chosen":   row[chosen_key],
        "rejected": row[rejected_key],
        "_chosen":  row["_chosen"],
        "_filter":  row["_filter"],
        "level":    row.get("level"),
        "type":     row.get("type"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",            default="data/sft_clean_preferred.jsonl")
    ap.add_argument("--output-dir",       default="data/sft_dpo")
    ap.add_argument("--val-frac",         type=float, default=0.10)
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--heldout-candidates", default=None,
                    help="JSONL of pre-selected heldout P1s (from build_sft_preferred.py). "
                         "When provided, skips linguasafe stratified sampling.")
    args = ap.parse_args()

    random.seed(args.seed)
    rows = load_jsonl(args.input)
    print(f"Loaded {len(rows)} rows from {args.input}")

    # --- SFT: all 586 rows, train on the preferred response (_chosen) ---
    sft_rows = list(rows)
    random.shuffle(sft_rows)
    n_val = max(1, int(len(sft_rows) * args.val_frac))
    sft_val   = [to_sft_messages(r) for r in sft_rows[:n_val]]
    sft_train = [to_sft_messages(r) for r in sft_rows[n_val:]]

    # --- DPO: all rows ---
    dpo_rows = [to_dpo_row(r) for r in rows]
    random.shuffle(dpo_rows)
    n_val_dpo  = max(1, int(len(dpo_rows) * args.val_frac))
    dpo_val   = dpo_rows[:n_val_dpo]
    dpo_train = dpo_rows[n_val_dpo:]

    from collections import Counter

    all_used_ids = {r["p1_id"] for r in sft_rows}  # train + val p1_ids
    val_ids = {r["p1_id"] for r in sft_rows[:n_val]}

    # --- Heldout A: stratified split from within the preferred dataset ---
    # Pulled from the val split (already held out of training), so no training rows lost.
    # Distribution exactly matches training by construction.
    heldout_a = []
    for r in sft_rows[:n_val]:
        heldout_a.append({
            "p1_id": r["p1_id"],
            "p1":    r["p1"],
            "level": r.get("level"),
            "type":  r.get("type"),
        })
    random.shuffle(heldout_a)

    # --- Heldout B: discarded P1s (no model's O2 beat baseline) ---
    # Tests generalization to prompts where guidelines refinement failed.
    heldout_b = []
    if args.heldout_candidates:
        try:
            candidates = load_jsonl(args.heldout_candidates)
            linguasafe_text: dict[str, str] = {}
            try:
                for row in load_jsonl("dataset/linguasafe_train.jsonl"):
                    linguasafe_text[str(row["id"])] = row.get("prompt", "")
            except FileNotFoundError:
                pass
            heldout_b = [
                {
                    "p1_id": str(c["p1_id"]),
                    "p1":    linguasafe_text.get(str(c["p1_id"]), ""),
                    "level": c.get("level"),
                    "type":  c.get("type"),
                }
                for c in candidates
                if str(c["p1_id"]) not in all_used_ids
                and linguasafe_text.get(str(c["p1_id"]))
            ]
            random.shuffle(heldout_b)
        except FileNotFoundError:
            print(f"  Warning: {args.heldout_candidates} not found, skipping heldout_b")

    out = args.output_dir
    print(f"\nWriting to {out}/")
    write_jsonl(sft_train, f"{out}/sft_train.jsonl")
    write_jsonl(sft_val,   f"{out}/sft_val.jsonl")
    write_jsonl(dpo_train, f"{out}/dpo_train.jsonl")
    write_jsonl(dpo_val,   f"{out}/dpo_val.jsonl")
    if heldout_a:
        write_jsonl(heldout_a, f"{out}/heldout_indist.jsonl")
    if heldout_b:
        write_jsonl(heldout_b, f"{out}/heldout_oodist.jsonl")

    def dist_str(rows: list[dict], key: str) -> str:
        c = Counter(r.get(key) for r in rows if r.get(key) is not None)
        return str(dict(sorted(c.items()) if key == "level" else c.most_common()))

    print(f"\nSummary:")
    print(f"  SFT   train={len(sft_train)}  val={len(sft_val)}  (always O2 response)")
    print(f"  DPO   train={len(dpo_train)}  val={len(dpo_val)}  (all pairs)")
    if heldout_a:
        print(f"  heldout_indist: {len(heldout_a)} prompts  (from val split, matches train distribution)")
        print(f"    level: {dist_str(heldout_a, 'level')}")
        print(f"    type:  {dist_str(heldout_a, 'type')}")
    if heldout_b:
        print(f"  heldout_oodist: {len(heldout_b)} prompts  (discarded P1s, no O2 beat baseline)")
        print(f"    level: {dist_str(heldout_b, 'level')}")
        print(f"    type:  {dist_str(heldout_b, 'type')}")


if __name__ == "__main__":
    main()
