#!/usr/bin/env python3
"""
Generate category-specific safety guidelines from universal guidelines + category.
Uses the universal (pillar) guidelines to derive category-specific rules.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Union

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Default model for guideline generation: LLaMA 70B via Hugging Face
DEFAULT_GUIDELINES_MODEL = "meta-llama/Llama-2-70b-chat-hf"

from src.utils.model_client import call_model, AZURE_MODEL


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def categories_from_linguasafe_csv(path: Path, type_column: str = "type") -> List[str]:
    """Load unique category names from linguasafe CSV (column 'type')."""
    seen: Set[str] = set()
    out: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if type_column not in (reader.fieldnames or []):
            raise ValueError(f"linguasafe CSV missing column '{type_column}'")
        for row in reader:
            cat = (row.get(type_column) or "").strip()
            if cat and cat not in seen:
                seen.add(cat)
                out.append(cat)
    return sorted(out)


def categories_from_linguasafe_jsonl(path: Path, type_key: str = "type") -> List[str]:
    """Load unique category names from linguasafe JSONL."""
    seen: Set[str] = set()
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            cat = (obj.get(type_key) or obj.get("category") or "").strip()
            if cat and cat not in seen:
                seen.add(cat)
                out.append(cat)
        except json.JSONDecodeError:
            continue
    return sorted(out)


def sanitize_category_filename(category: str) -> str:
    """Turn category name into a safe filename (no slashes, minimal punctuation)."""
    name = re.sub(r"[^\w\s\-]", "", category)
    name = re.sub(r"\s+", "_", name.strip())
    return name or "unknown"


def load_universal_guidelines(path: Path) -> Union[Dict, str]:
    """Load universal safety guidelines from .txt (plain text) or JSON/JSONL."""
    if path.suffix.lower() == ".txt":
        return load_text(path)
    raw = load_text(path)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # JSONL: one JSON object per line
    for line in raw.splitlines():
        line = line.strip()
        if line and (line.startswith("{") or line.startswith("[")):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    # File has .json/.jsonl extension but content is plain text
    if raw and not (raw.lstrip().startswith("{") or raw.lstrip().startswith("[")):
        return raw
    raise ValueError(
        f"Cannot parse {path}: not valid JSON/JSONL and not plain text (must start with {{ or [ for JSON)."
    )


def format_universal_guidelines(data: Union[Dict, str]) -> str:
    """Format universal guidelines as text for the prompt. If data is already str (from .txt), return as-is."""
    if isinstance(data, str):
        return data
    pillars = data.get("pillars", data.get("principles", []))
    lines: List[str] = []
    for pillar in pillars:
        name = pillar.get("name", "").strip()
        if name:
            lines.append(name)
        for guideline in pillar.get("guidelines", []):
            lines.append(f"- {guideline}")
        lines.append("")
    return "\n".join(lines).strip()


def load_categories(path: Path) -> List[str]:
    categories: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            value = data.get("category") or data.get("harm_category") or data.get("name")
            if value:
                categories.append(str(value).strip())
                continue
        except json.JSONDecodeError:
            pass
        categories.append(line)
    return categories


def build_prompt(template: str, category: str, universal_guidelines: str) -> str:
    return (
        "You are a Safety Rule Generation Agent.\n\n"
        f"{template}\n\n"
        f"CORE SAFETY PRINCIPLES:\n{universal_guidelines}\n\n"
        f"RISK DOMAIN:\n{category}\n"
    )


def call_openai_model(
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call OpenAI directly (non-Azure), using OPENAI_API_KEY from env."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Add it to .env or environment when using --use-openai."
        )

    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def _log_ok(api: str, status_code: object, resp_id: object) -> None:
        if status_code is not None:
            print(f"  ✓ OpenAI {api} OK — HTTP {status_code} — id={resp_id}", flush=True)
        else:
            print(f"  ✓ OpenAI {api} OK — id={resp_id}", flush=True)

    # Prefer Responses API for newer models, fallback to chat.completions for compatibility.
    try:
        rs = client.responses
        if hasattr(rs, "with_raw_response"):
            raw = rs.with_raw_response.create(
                model=model,
                input=messages,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            status = getattr(raw.http_response, "status_code", None)
            resp = raw.parse()
            text = getattr(resp, "output_text", None)
            if text and text.strip():
                _log_ok("responses", status, getattr(resp, "id", None))
                return text.strip()
        else:
            resp = rs.create(
                model=model,
                input=messages,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            text = getattr(resp, "output_text", None)
            if text and text.strip():
                _log_ok("responses", None, getattr(resp, "id", None))
                return text.strip()
    except Exception:
        pass

    cc = client.chat.completions
    if hasattr(cc, "with_raw_response"):
        raw = cc.with_raw_response.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        status = getattr(raw.http_response, "status_code", None)
        completion = raw.parse()
        content = (completion.choices[0].message.content or "").strip()
        _log_ok("chat.completions", status, getattr(completion, "id", None))
        return content

    completion = cc.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = (completion.choices[0].message.content or "").strip()
    _log_ok("chat.completions", None, getattr(completion, "id", None))
    return content


def main():
    parser = argparse.ArgumentParser(description="Generate category-specific guidelines.")
    parser.add_argument("--category", action="append", help="Category name (repeatable).")
    parser.add_argument("--categories-file", type=str, help="Text or JSONL file with categories.")
    parser.add_argument(
        "--linguasafe",
        type=str,
        default=None,
        help="Path to linguasafe CSV or JSONL; use its 'type' column for categories. Default: dataset/linguasafe.csv if present.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="src/prompts/category_guidelines.jsonl",
        help="Prompt template path.",
    )
    parser.add_argument(
        "--universal-guidelines",
        type=str,
        default="src/prompts/guidelines.txt",
        help="Universal safety guidelines path (.txt plain text, or JSON/JSONL).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory; one file per category. Default: output/guidelines/<stem of universal-guidelines> (e.g. output/guidelines/guidelines, output/guidelines/guidelines_v2).",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=".txt",
        choices=[".txt", ".md"],
        help="Extension for per-category files (default: .txt).",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GUIDELINES_MODEL,
        help="Local path to LLaMA model or HF repo id (used when --use-local-model not set).",
    )
    parser.add_argument(
        "--use-local-model",
        action="store_true",
        help="Use local model (same as p1_to_p2 job: vLLM when available).",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Local model path or HF id (use with --use-local-model; overrides --model).",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use Azure OpenAI API instead of local model.",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API directly (OPENAI_API_KEY in .env), instead of local or Azure.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5.4",
        help="OpenAI model id when --use-openai is set (default: gpt-5.4).",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Use Hugging Face pipeline for inference instead of vLLM (slower).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing category files in the output dir. Default: skip existing files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N categories after deduplication (e.g. 1 = one API call / one category smoke test).",
    )
    args = parser.parse_args()
    if args.use_api and args.use_openai:
        raise SystemExit("Use only one remote provider: choose either --use-api (Azure) or --use-openai.")

    # Same as run_p1_to_p2_guidelines.py: set env so model_client uses vLLM when --use-local-model
    if args.use_api:
        os.environ["USE_LOCAL_MODEL"] = "false"
    elif args.use_local_model and args.local_model:
        os.environ["USE_LOCAL_MODEL"] = "true"
        os.environ["LOCAL_MODEL"] = args.local_model
    else:
        os.environ["USE_LOCAL_MODEL"] = "true"
        os.environ["LOCAL_MODEL"] = args.model

    categories: List[str] = []
    if args.category:
        categories.extend([c.strip() for c in args.category if c and c.strip()])
    if args.categories_file:
        categories.extend(load_categories(REPO_ROOT / args.categories_file))
    if args.linguasafe is not None:
        p = REPO_ROOT / args.linguasafe
        if not p.exists():
            raise SystemExit(f"linguasafe path not found: {p}")
        if p.suffix.lower() == ".csv":
            categories.extend(categories_from_linguasafe_csv(p))
        else:
            categories.extend(categories_from_linguasafe_jsonl(p))
    if not categories and (REPO_ROOT / "dataset/linguasafe.csv").exists():
        categories.extend(categories_from_linguasafe_csv(REPO_ROOT / "dataset/linguasafe.csv"))

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for c in categories:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    categories = unique

    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        categories = categories[: args.limit]
        print(f"Using --limit {args.limit}: processing {len(categories)} categor(y/ies).", flush=True)

    if not categories:
        raise SystemExit(
            "No categories provided. Use --category, --categories-file, or --linguasafe (or place dataset/linguasafe.csv)."
        )

    # Default output dir from universal-guidelines filename under output/guidelines/
    # e.g. guidelines.txt -> output/guidelines/guidelines, guidelines_v2.jsonl -> output/guidelines/guidelines_v2
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"output/guidelines/{Path(args.universal_guidelines).stem}"

    template = load_text(REPO_ROOT / args.template)
    universal = load_universal_guidelines(REPO_ROOT / args.universal_guidelines)
    universal_text = format_universal_guidelines(universal)

    out_dir = REPO_ROOT / output_dir
    # Create output directory only if it does not exist (never wipe or overwrite existing)
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output dir: {out_dir}")
    else:
        print(f"Output dir already exists (will skip existing category files): {out_dir}", flush=True)

    to_generate = sum(
        1 for c in categories
        if not (out_dir / (sanitize_category_filename(c) + args.ext)).exists() or args.overwrite
    )
    if to_generate > 0:
        if args.use_openai:
            gen_note = f"OpenAI model `{args.openai_model}` (API latency varies)."
        elif args.use_api:
            gen_note = "Azure OpenAI (API latency varies)."
        else:
            gen_note = f"local model `{args.local_model or args.model}` (large models often ~1–5 min each on GPU)."
        print(f"Generating {to_generate} category guideline(s). {gen_note}", flush=True)
    written = 0
    for i, category in enumerate(categories):
        fname = sanitize_category_filename(category) + args.ext
        out_path = out_dir / fname
        if out_path.exists() and not args.overwrite:
            print(f"[{i + 1}/{len(categories)}] {category} -> skip (already exists: {out_path})", flush=True)
            continue
        wait_hint = (
            ""
            if (args.use_openai or args.use_api)
            else " (may take 1–5 min on large local models)"
        )
        print(f"[{i + 1}/{len(categories)}] {category} -> generating...{wait_hint}", flush=True)
        prompt = build_prompt(template, category, universal_text)
        if args.use_openai:
            guidelines_text = call_openai_model(
                system_prompt="You are a Safety Guideline Generation Agent.",
                user_prompt=prompt,
                model=args.openai_model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        else:
            model_name = (
                (args.local_model if (args.use_local_model and args.local_model) else args.model)
                if not args.use_api
                else None
            )
            guidelines_text = call_model(
                system_prompt="You are a Safety Guideline Generation Agent.",
                user_prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                use_hf=args.use_hf,
                model_name=model_name,
            )
        out_path.write_text(guidelines_text.strip(), encoding="utf-8")
        written += 1
        print(f"[{i + 1}/{len(categories)}] {category} -> {out_path}", flush=True)

    print(f"Wrote {written} category guidelines to {out_dir} (skipped existing files)", flush=True)


if __name__ == "__main__":
    main()
