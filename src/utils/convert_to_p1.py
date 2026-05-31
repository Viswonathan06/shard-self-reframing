#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
minimal_convert_to_p1.py
------------------------
Convert a dataset into a minimal JSONL with ONLY one field per line:
{"p1": "..."}  or  {"p1_redacted": "..."}

Supports:
- Input formats: JSONL, CSV, JSON (array of objects or object with "data"/"items"/"records").
- Choose the source text field with --text_field, or auto-detect (p1,text,prompt,content,instruction).
- Optional placeholder redaction with --apply_terms config/terms_{lang}.json
- Optional dedupe and limit.
- Random sampling using reservoir sampling: --sample N [--random_seed S].
- NEW: Language filtering: --filter_lang XX (e.g., ar, zh) with optional --lang_field (default: "lang").

Examples
--------
# 1) JSONL -> p1 (no redaction), only Arabic rows, sample 200
python minimal_convert_to_p1.py \
  --in_file data/mixed.jsonl \
  --filter_lang ar \
  --lang_field lang \
  --text_field prompt \
  --output_path appendix/prompts/p1_ar.jsonl \
  --output_field p1 \
  --sample 200

# 2) CSV -> p1_redacted using Arabic terms, only 'ar' rows
python minimal_convert_to_p1.py \
  --in_file data/arabic.csv \
  --filter_lang ar \
  --apply_terms config/terms_ar.json \
  --text_field text \
  --output_field p1_redacted \
  --output_path appendix/prompts/p1_ar.jsonl

# 3) Convert all English rows without sampling
python minimal_convert_to_p1.py \
  --in_file data/english.jsonl \
  --filter_lang en \
  --output_path appendix/prompts/p1_en.jsonl
"""

import sys, json, re, csv, argparse, hashlib, random
from pathlib import Path

def infer_format(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".jsonl":
        return "jsonl"
    if ext == ".csv":
        return "csv"
    if ext == ".json":
        return "json"
    return "unknown"

def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def stream_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row

def stream_json_array_or_object(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj
    elif isinstance(data, dict):
        for k in ("data", "items", "records"):
            arr = data.get(k)
            if isinstance(arr, list):
                for obj in arr:
                    if isinstance(obj, dict):
                        yield obj
                return
        # fallthrough: treat root object as a single record if it has text-ish keys
        yield data

def load_terms(terms_path: str):
    with open(terms_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    order = cfg.get("order", [])
    cats = cfg.get("categories", {})
    compiled = {}
    for cat in order:
        pats = cats.get(cat, [])
        compiled[cat] = [re.compile(p, re.IGNORECASE) for p in pats]
    return order, compiled

def redact_text(text: str, terms_path: str):
    order, compiled_by_cat = load_terms(terms_path)
    taken = [False] * (len(text) + 1)

    def free(s, e):
        for i in range(s, e):
            if i < len(taken) and taken[i]:
                return False
        return True
    def mark(s, e):
        for i in range(s, e):
            if i < len(taken):
                taken[i] = True

    matches = []
    for cat in order:
        for rx in compiled_by_cat.get(cat, []):
            for m in rx.finditer(text):
                s, e = m.start(), m.end()
                if s < e and free(s, e):
                    matches.append({"start": s, "end": e, "span": text[s:e], "cat": cat})
                    mark(s, e)

    # placeholders
    counters = {}
    mapping = {}
    for m in matches:
        key = (m["span"], m["cat"])
        if key in mapping: continue
        n = counters.get(m["cat"], 0) + 1
        counters[m["cat"]] = n
        mapping[key] = f"[{m['cat'].upper()}_{n:03d}]"

    # replace from right to left
    repls = [(m["start"], m["end"], mapping[(m["span"], m["cat"])]) for m in matches]
    repls.sort(key=lambda x: x[0], reverse=True)
    out = text
    for s, e, ph in repls:
        out = out[:s] + ph + out[e:]

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True, help="Input dataset (CSV/JSONL/JSON).")
    ap.add_argument("--output_path", required=True, help="Output JSONL path (e.g., appendix/prompts/p1_en.jsonl).")
    ap.add_argument("--text_field", default=None, help="Field containing the prompt text (auto-detect if omitted).")
    ap.add_argument("--output_field", default="p1", choices=["p1", "p1_redacted"], help="Output field name.")
    ap.add_argument("--apply_terms", default=None, help="Optional terms_*.json to placeholder-redact.")
    ap.add_argument("--dedupe", action="store_true", help="Drop exact duplicates.")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N records.")
    ap.add_argument("--sample", type=int, default=None, help="Randomly sample N records using reservoir sampling.")
    ap.add_argument("--random_seed", type=int, default=42, help="Random seed for sampling.")

    # Language filtering
    ap.add_argument("--filter_lang", default=None, help="Keep only rows whose language field matches this value (e.g., 'ar').")
    ap.add_argument("--lang_field", default="lang", help="Field name that contains the language code (default: 'lang').")
    ap.add_argument("--case_sensitive", action="store_true", help="Use case-sensitive language match (default: case-insensitive).")
    args = ap.parse_args()

    random.seed(args.random_seed)

    fmt = infer_format(args.in_file)
    if fmt == "unknown":
        print("[minimal_convert] Unknown input format; use CSV/JSONL/JSON.", file=sys.stderr)
        sys.exit(2)

    # choose stream
    if fmt == "jsonl":
        stream = stream_jsonl(args.in_file)
    elif fmt == "csv":
        stream = stream_csv(args.in_file)
    else:
        stream = stream_json_array_or_object(args.in_file)

    # auto-detect field if needed
    candidates = ["p1", "text", "prompt", "content", "instruction"]
    text_field = args.text_field

    # dedupe set
    seen = set() if args.dedupe else None

    total_in = total_out = 0
    skipped_empty = skipped_dupe = 0
    skipped_lang = 0

    # sampling
    k = args.sample if args.sample and args.sample > 0 else None
    reservoir = [] if k else None
    n = 0  # eligible rows seen

    # prepare output dir
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    def matches_lang(obj):
        if not args.filter_lang:
            return True
        if not isinstance(obj, dict):
            return False
        val = obj.get(args.lang_field)
        if val is None:
            return False
        sval = str(val)
        target = args.filter_lang
        if not args.case_sensitive:
            sval = sval.lower().strip()
            target = target.lower().strip()
        return sval == target

    def prepare_out(text_val: str):
        return {args.output_field: text_val}

    # open file if not sampling
    fout = None
    if not k:
        fout = open(args.output_path, "w", encoding="utf-8")

    for obj in stream:
        total_in += 1
        if args.limit and total_in > args.limit:
            break

        if not matches_lang(obj):
            skipped_lang += 1
            continue

        if text_field is None:
            for c in candidates:
                if isinstance(obj, dict) and c in obj and isinstance(obj[c], str):
                    text_field = c
                    break

        text = (obj.get(text_field) if isinstance(obj, dict) and text_field else None)
        if not isinstance(text, str) or not text.strip():
            skipped_empty += 1
            continue
        text = text.strip()

        if args.apply_terms:
            try:
                text = redact_text(text, args.apply_terms)
            except Exception as e:
                print(f"[minimal_convert] redaction failed on record {total_in}: {e}", file=sys.stderr)

        if seen is not None:
            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in seen:
                skipped_dupe += 1
                continue
            seen.add(h)

        # eligible
        if k:
            n += 1
            entry = prepare_out(text)
            if len(reservoir) < k:
                reservoir.append(entry)
            else:
                j = random.randint(1, n)
                if j <= k:
                    reservoir[j - 1] = entry
        else:
            fout.write(json.dumps(prepare_out(text), ensure_ascii=False) + "\n")
            total_out += 1

    # finalize sampling
    if k:
        random.shuffle(reservoir)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for entry in reservoir:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total_out += 1

    if fout:
        fout.close()

    print(f"[minimal_convert] in={total_in} written={total_out} empty={skipped_empty} dupes={skipped_dupe} lang_filtered_out={skipped_lang}")
    print(f"[minimal_convert] -> {args.output_path}")

if __name__ == "__main__":
    main()
