# utils/sanitize.py
import json
import re
from typing import Dict, Tuple, List

PLACEHOLDER_PREFIX = "TAG_"

def load_terms(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # expect {"dangerous_terms": ["regex1", "regex2", ...]}
    return data.get("dangerous_terms", [])

def build_placeholder_map(text: str, term_patterns: List[str]) -> Dict[str, str]:
    """
    Find unique matches and map each unique risky span (by surface form) to [TAG_xxx].
    We DO NOT store the original spans in outputs; we only use them internally here.
    """
    found = []
    for pat in term_patterns:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            span = text[m.start():m.end()]
            if span not in found:
                found.append(span)

    mapping = {}
    for i, span in enumerate(found, start=1):
        mapping[span] = f"[{PLACEHOLDER_PREFIX}{i:03d}]"
    return mapping

def apply_placeholder_map(text: str, mapping: Dict[str, str]) -> str:
    # Replace longer spans first to avoid partial overlaps
    for span in sorted(mapping.keys(), key=len, reverse=True):
        text = re.sub(re.escape(span), mapping[span], text)
    return text

def sanitize_with_placeholders(p1: str, term_patterns: List[str]) -> Tuple[str, Dict[str, str], Dict]:
    """
    Returns (p1_redacted, mapping, stats). mapping: original_span -> [TAG_xxx]
    stats: counts only (no raw spans).
    """
    mapping = build_placeholder_map(p1, term_patterns)
    redacted = apply_placeholder_map(p1, mapping)
    stats = {"num_redactions": len(mapping), "placeholders": list(mapping.values())}
    return redacted, mapping, stats

def fallback_strict_sanitize(p1_redacted: str) -> str:
    """
    If the first try still triggers filter, we can further neutralize verbs like 'design/make/bypass'.
    Keep it conservative.
    """
    patterns = [
        r"\bdesign\b", r"\bmake\b", r"\bbuild\b", r"\bexploit\b",
        r"\bbypass\b", r"\bcircumvent\b", r"\binject\b", r"\bsteal\b",
    ]
    out = p1_redacted
    for pat in patterns:
        out = re.sub(pat, "[ACTION]", out, flags=re.IGNORECASE)
    return out
