#!/usr/bin/env python3
"""
Compare P1 vs P2 semantically and with ROUGE; report change stats per category and per harm level.

- Loads p1_p2_outputs.jsonl (or given path).
- SBERT: cosine similarity (0-1); "change" = 1 - similarity.
- ROUGE: ROUGE-1, ROUGE-2, ROUGE-L F1 (reference=p1, candidate=p2).
- Outputs: per-category and per-level stats for all metrics; optional per-row CSV.

Usage:
  python scripts/compare_p1_p2_similarity.py [--input path] [--output-dir dir] [--no-rouge]
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compute_similarities_batched(rows, model, batch_size: int = 64):
    """Compute cosine similarity for each (p1, p2) in rows; return list of floats."""
    try:
        from sentence_transformers import SentenceTransformer
        from torch.nn.functional import cosine_similarity
        import torch
    except ImportError:
        print("sentence-transformers is required: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    if model is None:
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    similarities = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        p1_texts = [r.get("p1") or "" for r in batch]
        p2_texts = [r.get("p2") or "" for r in batch]
        e1 = model.encode(p1_texts, convert_to_tensor=True)
        e2 = model.encode(p2_texts, convert_to_tensor=True)
        for j in range(len(batch)):
            sim = cosine_similarity(e1[j : j + 1], e2[j : j + 1]).item()
            similarities.append(float(sim))
    return similarities


def compute_rouge(reference: str, candidate: str) -> dict:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 (reference=p1, candidate=p2). Returns dict with rouge1, rouge2, rougeL."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        scores = scorer.score(reference, candidate)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def stats_for_group(values, metric_key: str = "similarity"):
    """Stats for a list of similarity values (or list of dicts with metric_key)."""
    if not values:
        return {"count": 0, "mean_similarity": None, "mean_change": None, "std_similarity": None}
    import statistics
    if isinstance(values[0], dict):
        vals = [v[metric_key] for v in values]
    else:
        vals = values
    mean_sim = statistics.mean(vals)
    return {
        "count": len(vals),
        "mean_similarity": round(mean_sim, 4),
        "mean_change": round(1.0 - mean_sim, 4),
        "std_similarity": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
    }


def stats_for_group_multi(rows_in_group: list, keys: list) -> dict:
    """Given list of row dicts, compute mean and std for each key."""
    import statistics
    out = {"count": len(rows_in_group)}
    for k in keys:
        vals = [r[k] for r in rows_in_group if k in r and r[k] is not None]
        if vals:
            out[f"mean_{k}"] = round(statistics.mean(vals), 4)
            out[f"std_{k}"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0
        else:
            out[f"mean_{k}"] = None
            out[f"std_{k}"] = None
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Compare P1 vs P2 semantically; stats per category and per harm level."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "output/p1_to_p2/p1_p2_guidelines/p1_p2_outputs.jsonl"),
        help="Input JSONL path (default: output/p1_to_p2/p1_p2_guidelines/p1_p2_outputs.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output CSVs (default: same dir as input)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding (default: 64)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model name (default: paraphrase-multilingual-MiniLM-L12-v2)",
    )
    parser.add_argument(
        "--per-row-csv",
        type=str,
        default=None,
        help="Path for per-row CSV (default: <output-dir>/p1_p2_similarity_per_row.csv)",
    )
    parser.add_argument(
        "--no-per-row-csv",
        action="store_true",
        help="Do not write the per-row CSV",
    )
    parser.add_argument("--no-rouge", action="store_true", help="Skip ROUGE computation")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    per_row_path = Path(args.per_row_csv) if args.per_row_csv else (out_dir / "p1_p2_similarity_per_row.csv")
    if args.no_per_row_csv:
        per_row_path = None

    print("Loading rows...")
    rows = load_jsonl(input_path)
    # Only rows with both p1 and p2
    rows = [r for r in rows if r.get("p1") and r.get("p2")]
    print(f"Loaded {len(rows)} rows with both p1 and p2.")

    if not rows:
        print("No rows to analyze.")
        sys.exit(0)

    print(f"Loading SBERT model: {args.model}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    print("Computing semantic similarities (batched)...")
    similarities = compute_similarities_batched(rows, model, args.batch_size)
    assert len(similarities) == len(rows)

    # Attach similarity and change to each row
    for i, r in enumerate(rows):
        r["similarity"] = similarities[i]
        r["change"] = 1.0 - similarities[i]

    use_rouge = not args.no_rouge
    extra_metrics = []
    if use_rouge:
        print("Computing ROUGE (p1=ref, p2=candidate)...")
        for r in rows:
            rouge = compute_rouge(r.get("p1") or "", r.get("p2") or "")
            r["rouge1"] = rouge["rouge1"]
            r["rouge2"] = rouge["rouge2"]
            r["rougeL"] = rouge["rougeL"]
        extra_metrics.extend(["rouge1", "rouge2", "rougeL"])

    # Per category (store full row dicts for multi-metric stats)
    by_cat = defaultdict(list)
    for r in rows:
        cat = r.get("category") or "Unknown"
        by_cat[cat].append(r)

    # Per level (harm level)
    by_level = defaultdict(list)
    for r in rows:
        level = r.get("level")
        if level is None:
            key = "unknown"
        else:
            key = int(level) if isinstance(level, (int, float)) and level == int(level) else level
        by_level[str(key)].append(r)

    def build_group_stats(group_rows, id_key=None, id_value=None):
        s = stats_for_group([x["similarity"] for x in group_rows])
        if id_key and id_value is not None:
            s[id_key] = id_value
        if extra_metrics:
            multi = stats_for_group_multi(group_rows, extra_metrics)
            for k, v in multi.items():
                if k != "count":
                    s[k] = v
        return s

    # Build stats tables
    cat_stats = []
    for cat in sorted(by_cat.keys()):
        s = build_group_stats(by_cat[cat], "category", cat)
        cat_stats.append(s)

    level_stats = []
    for level in sorted(by_level.keys(), key=lambda x: (x == "unknown", x)):
        s = build_group_stats(by_level[level], "level", level)
        level_stats.append(s)

    # Overall
    overall = build_group_stats(rows, "category", "OVERALL")
    cat_stats.insert(0, overall)

    # Save CSVs
    cat_path = out_dir / "p1_p2_similarity_by_category.csv"
    level_path = out_dir / "p1_p2_similarity_by_level.csv"
    pd.DataFrame(cat_stats).to_csv(cat_path, index=False)
    pd.DataFrame(level_stats).to_csv(level_path, index=False)
    print(f"Wrote {cat_path}")
    print(f"Wrote {level_path}")

    if per_row_path is not None:
        per_row_path.parent.mkdir(parents=True, exist_ok=True)
        optional = [k for k in ["rouge1", "rouge2", "rougeL"] if k in (rows[0] if rows else {})]
        df = pd.DataFrame([
            {
                "p1_id": r.get("p1_id"),
                "category": r.get("category"),
                "level": r.get("level"),
                "similarity": r["similarity"],
                "change": r["change"],
                "len_p1": len((r.get("p1") or "")),
                "len_p2": len((r.get("p2") or "")),
                **{k: r.get(k) for k in optional},
            }
            for r in rows
        ])
        df.to_csv(per_row_path, index=False)
        print(f"Wrote per-row CSV: {per_row_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"  Count: {overall['count']}")
    print(f"  Mean similarity (p1 vs p2): {overall['mean_similarity']}")
    print(f"  Mean change (1 - similarity): {overall['mean_change']}")
    print(f"  Std similarity: {overall['std_similarity']}")
    if use_rouge:
        for k in ["rouge1", "rouge2", "rougeL"]:
            if f"mean_{k}" in overall:
                label = "L" if k == "rougeL" else k.replace("rouge", "")
                print(f"  Mean ROUGE-{label} F1: {overall[f'mean_{k}']}")

    print("\n" + "=" * 70)
    print("BY CATEGORY (how much prompts changed after conversion)")
    print("=" * 70)
    print(pd.DataFrame(cat_stats).to_string(index=False))

    print("\n" + "=" * 70)
    print("BY HARM LEVEL (how much prompts changed after conversion)")
    print("=" * 70)
    print(pd.DataFrame(level_stats).to_string(index=False))


if __name__ == "__main__":
    main()
