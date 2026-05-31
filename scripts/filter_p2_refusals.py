#!/usr/bin/env python3
"""
Filter P2 outputs to remove refusals and extract clean P2s from meta-commentary.

Removes:
- Direct refusals (I cannot provide, I cannot fulfill, etc.)
- Meta-commentary where the quoted content is a refusal
- Outputs that are explanations rather than rewritten questions

Extracts:
- P2 from meta-wrapper like "Sure, here's a rewritten safe prompt: \"<actual P2>\""
  when the quoted part is a valid question.

Output: JSONL with p1_id, p1, p2, lang, category, level
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# Phrases that indicate a refusal (case-insensitive, checked at start or early in text)
REFUSAL_PHRASES = [
    "i cannot fulfill",
    "i cannot provide",
    "i cannot assist",
    "i cannot offer",
    "i cannot help",
    "i can't provide",
    "i can't assist",
    "i can't fulfill",
    "i can't satisfy",
    "i'm unable to",
    "i'm just an ai",
    "it's not within my programming",
    "it's not within my ethical",
    "please refine your prompt",
    "i won't provide",
    "i will not provide",
]

# Phrase that indicates meta-commentary wrapping
META_PREFIX = re.compile(
    r"^(?:Sure,?\s*!?\s*)?here'?s?\s+(?:a\s+)?rewritten\s+safe\s+prompt",
    re.IGNORECASE,
)


def is_refusal(text: str, max_pos: int = 250) -> bool:
    """Return True if text is a refusal (not a valid P2)."""
    if not text or not isinstance(text, str):
        return True
    t = text.strip().lower()
    first_part = t[:max_pos]
    for phrase in REFUSAL_PHRASES:
        pos = first_part.find(phrase)
        if pos >= 0 and pos < max_pos:
            return True
    return False


def looks_like_refusal_in_disguise(text: str) -> bool:
    """Check if text is refusal-ish even without explicit 'I cannot' (e.g. 'I understand your interest... However')."""
    if not text:
        return True
    t = text.strip().lower()
    # "I understand your interest" followed by refusal-style content
    if t.startswith("i understand your") and ("however" in t[:200] or "it's important to note" in t[:200]):
        return True
    if t.startswith("i'm glad you're interested") and "however" in t[:200]:
        return True
    # Long explanation with no question
    if "?" not in t[:150] and len(t) > 200 and ("instead" in t[:300] or "however" in t[:300]):
        if any(p in t for p in ["i suggest", "i would encourage", "i recommend"]):
            return True
    return False


def extract_quoted_p2(text: str) -> list[str]:
    """Extract quoted strings that might be the actual P2 (double-quoted or single-quoted)."""
    # Prefer double-quoted
    quoted = re.findall(r'"([^"]+)"', text)
    if not quoted:
        quoted = re.findall(r"'([^']+)'", text)
    return quoted


def is_valid_p2(text: str) -> bool:
    """Return True if text looks like a valid rewritten P2 (a question)."""
    if not text or len(text.strip()) < 10:
        return False
    t = text.strip()
    # Should contain a question mark (for question-style P2s)
    if "?" not in t:
        return False
    # Should not be a refusal
    if is_refusal(t, max_pos=150):
        return False
    if looks_like_refusal_in_disguise(t):
        return False
    # Reasonable length for a question (allow up to 400 chars for complex questions)
    if len(t) > 500:
        return False
    return True


def extract_clean_p2(raw_p2: str) -> Optional[str]:
    """
    Extract a clean P2 from raw model output.
    Returns the clean P2 string, or None if it should be filtered out.
    """
    if not raw_p2 or not isinstance(raw_p2, str):
        return None
    raw = raw_p2.strip()

    # Direct refusal
    if is_refusal(raw, max_pos=200):
        return None

    # Check for meta-commentary wrapper
    if META_PREFIX.match(raw) or ("here's a rewritten" in raw.lower()[:80] or "here is a rewritten" in raw.lower()[:80]):
        quoted = extract_quoted_p2(raw)
        for q in quoted:
            q = q.strip()
            if len(q) < 20:
                continue
            if is_refusal(q, max_pos=120):
                continue
            if looks_like_refusal_in_disguise(q):
                continue
            if is_valid_p2(q):
                return q
        # All quoted parts were refusals or invalid
        return None

    # Check for "Explanation:" style - P2 might be in first quote
    if "explanation:" in raw.lower() or "explanation\n" in raw.lower():
        quoted = extract_quoted_p2(raw)
        for q in quoted:
            if is_valid_p2(q):
                return q.strip()
        return None

    # Plain output - use as-is if valid
    if is_valid_p2(raw):
        return raw
    if looks_like_refusal_in_disguise(raw):
        return None
    # Overlong or no question - likely not a clean P2
    if "?" not in raw or len(raw) > 400:
        return None
    # Last resort: if it has a question and isn't a refusal, keep it
    if not is_refusal(raw, max_pos=150):
        return raw
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter P2 outputs to remove refusals and extract clean P2s."
    )
    parser.add_argument(
        "input",
        type=Path,
        default=Path("output/p1_to_p2/p1_p2_guidelines/p1_p2_outputs.jsonl"),
        nargs="?",
        help="Input JSONL file (default: output/p1_to_p2/p1_p2_guidelines/p1_p2_outputs.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file (default: same dir as input, suffix _filtered.jsonl)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print filtering stats to stderr",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        output_path = input_path.parent / (
            input_path.stem + "_filtered" + input_path.suffix
        )

    records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    kept = []
    filtered = 0
    extracted_from_meta = 0

    for rec in records:
        raw_p2 = rec.get("p2") or ""
        clean = extract_clean_p2(raw_p2)
        if clean is None:
            filtered += 1
            continue

        # Track if we extracted from meta
        if META_PREFIX.match(raw_p2.strip()) or "here's a rewritten" in raw_p2.lower()[:100]:
            extracted_from_meta += 1

        kept.append({
            "p1_id": rec.get("p1_id"),
            "p1": rec.get("p1"),
            "p2": clean,
            "lang": rec.get("lang"),
            "category": rec.get("category"),
            "level": rec.get("level"),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.stats:
        print(f"Input:  {len(records)} records", file=sys.stderr)
        print(f"Kept:   {len(kept)} records ({100*len(kept)/len(records):.1f}%)", file=sys.stderr)
        print(f"Filtered: {filtered}", file=sys.stderr)
        print(f"Extracted from meta-commentary: {extracted_from_meta}", file=sys.stderr)
        print(f"Output: {output_path}", file=sys.stderr)

    print(str(output_path))


if __name__ == "__main__":
    main()
