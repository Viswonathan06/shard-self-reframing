#!/usr/bin/env python3
"""
P1 → P2 conversion using the safety rewriter prompt.

Supports three guidelines modes (--guidelines-mode):
- full: universal guidelines + category-specific guidelines + harm category (default).
- universal_only: only universal guidelines; no category or category-specific guidelines.
- instructions_only: no guidelines; only instructions to reframe the prompt to a safe alternative.
"""

import argparse
import json
import os
import re
import fcntl
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.utils.model_client import call_model, ContentFilteredError
from src.utils.response_controls import apply_mention_level_to_control_signals
from src.utils.wandb_utils import init_wandb, log_metrics, finish_run

REPO_ROOT = Path(__file__).parents[2]

# System instruction for rewriter: output only the safe prompt
REWRITE_SYSTEM = (
    "Output ONLY the rewritten safe prompt text, with no preface, no apologies, "
    "no explanations, no labels, no extra quotes or markdown."
)


def sanitize_category_filename(category: str) -> str:
    """Turn category name into a safe filename (matches generate_category_guidelines)."""
    name = re.sub(r"[^\w\s\-]", "", category)
    name = re.sub(r"\s+", "_", name.strip())
    return name or "unknown"


def load_universal_guidelines(path: Path) -> str:
    """Load universal guidelines as text. Supports .txt (plain text) or JSON/JSONL (pillars/principles)."""
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8").strip()
    raw = path.read_text(encoding="utf-8").strip()
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = line.strip()
            if line and (line.startswith("{") or line.startswith("[")):
                try:
                    data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    if data is None:
        # .json/.jsonl file with plain-text content
        if raw and not (raw.lstrip().startswith("{") or raw.lstrip().startswith("[")):
            return raw
        raise ValueError(
            f"Cannot parse {path}: not valid JSON/JSONL and not plain text (must start with {{ or [ for JSON)."
        )
    # Format pillars or principles (v2/v3 use "principles")
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


def load_category_guidelines(guidelines_dir: Path, category: str) -> str:
    """Load category-specific guidelines from guidelines_dir/<sanitized_category>.txt."""
    fname = sanitize_category_filename(category) + ".txt"
    path = guidelines_dir / fname
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_rewriter_user_prompt(
    template: str,
    universal_guidelines: str,
    category_guidelines: str,
    category: str,
    p1: str,
) -> str:
    """Fill the safety rewriter template (full mode: universal + category guidelines)."""
    return (
        template.replace("{INSERT_UNIVERSAL_GUIDELINES_HERE}", universal_guidelines)
        .replace("{INSERT_CATEGORY_SPECIFIC_GUIDELINES_HERE}", category_guidelines or "(No category-specific guidelines.)")
        .replace("{INSERT_CATEGORY_NAME}", category)
        .replace("{INSERT_UNSAFE_PROMPT_HERE}", p1)
        .replace("{INSERT_PROMPT_HERE}", p1)
    )


def build_rewriter_user_prompt_universal_only(template: str, universal_guidelines: str, p1: str) -> str:
    """Fill the rewriter template when using only universal guidelines (no category)."""
    return (
        template.replace("{INSERT_UNIVERSAL_GUIDELINES_HERE}", universal_guidelines)
        .replace("{INSERT_UNSAFE_PROMPT_HERE}", p1)
        .replace("{INSERT_PROMPT_HERE}", p1)
    )


def build_rewriter_user_prompt_instructions_only(template: str, p1: str) -> str:
    """Fill the rewriter template when using only reframing instructions (no guidelines)."""
    return template.replace("{INSERT_UNSAFE_PROMPT_HERE}", p1).replace("{INSERT_PROMPT_HERE}", p1)


# Control signal keys from rewriter JSON (mention_level is derived from response_mode in code).
CONTROL_SIGNAL_KEYS = ("response_mode", "reformulated_component", "preserved_benign_need")


def parse_p2(raw: str) -> tuple:
    """
    Extract P2 and control signals from model output.
    Returns (p2_str, control_signals_dict).
    Supports: (1) JSON with 'p2' and optional control fields, (2) plain quoted string.
    """
    s = raw.strip()
    # Strip markdown code fence if present
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    s = s.strip()
    control: Dict = {}
    # Try to parse as JSON and get reformulated prompt + control signal fields
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            raw_p2 = obj.get("p2")
            if raw_p2 is None and "the reformulated prompt" in obj:
                raw_p2 = obj["the reformulated prompt"]
            if raw_p2 is not None:
                p2_str = str(raw_p2).strip() if raw_p2 else ""
                for key in CONTROL_SIGNAL_KEYS:
                    if key in obj and obj[key] is not None:
                        control[key] = obj[key]
                control = apply_mention_level_to_control_signals(control)
                return (p2_str, control)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: strip surrounding quotes; no control signals
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    if s.startswith("'") and s.endswith("'"):
        s = s[1:-1].strip()
    return (s, {})


def load_p1_from_csv(csv_path: Path, language: str, max_p1: Optional[int] = None) -> List[Dict]:
    """Load P1 prompts from linguasafe CSV filtered by language."""
    df = pd.read_csv(csv_path)
    df_filtered = df[df["lang"] == language].copy()
    if len(df_filtered) == 0:
        return []
    if max_p1 is not None and max_p1 > 0:
        df_filtered = df_filtered.head(max_p1)
    records = []
    for _, row in df_filtered.iterrows():
        rec = {
            "id": str(row.get("id", "")),
            "p1": str(row.get("prompt", "")),
            "lang": str(row.get("lang", language)),
            "level": row.get("level", None),
            "subtype": str(row.get("subtype", "")) if pd.notna(row.get("subtype")) else "",
            "type": str(row.get("type", "")) if pd.notna(row.get("type")) else "",
        }
        records.append(rec)
    return records


def load_p1_from_jsonl(jsonl_path: Path, max_p1: Optional[int] = None) -> List[Dict]:
    """Load P1 prompts from a JSONL file (e.g. donotanswer_no_outputs.jsonl)."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rec = {
                "id": str(obj.get("p1_id", obj.get("id", ""))),
                "p1": str(obj.get("p1", obj.get("prompt", ""))),
                "lang": str(obj.get("lang", "en")),
                "level": obj.get("level"),
                "subtype": str(obj.get("subtype", "")),
                "type": str(obj.get("type", "")),
            }
            records.append(rec)
            if max_p1 is not None and max_p1 > 0 and len(records) >= max_p1:
                break
    return records


def load_checkpoint(checkpoint_file: Path) -> Set[str]:
    """Load P1 IDs from checkpoint that have a valid p2 (considered done). Ids missing or with null/empty p2 are reprocessed."""
    processed = set()
    if not checkpoint_file.exists():
        return processed
    for line in checkpoint_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            pid = obj.get("p1_id") or obj.get("id")
            p2 = obj.get("p2")
            # Only consider done if id exists and p2 exists and is non-null, non-empty
            if pid is not None and str(pid).strip() and str(pid).lower() != "null":
                if p2 is not None and str(p2).strip() and str(p2).lower() != "null":
                    processed.add(str(pid))
        except json.JSONDecodeError:
            continue
    return processed


def append_jsonl(path: Path, records: List[Dict], lock_file: Path):
    """Append records to JSONL with file lock; flush so checkpoints are visible on disk immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(path, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(
        description="P1 → P2 conversion using safety rewriter prompt + universal + category guidelines"
    )
    parser.add_argument("--csv-path", type=str, default=None, help="Path to linguasafe.csv (mutually exclusive with --jsonl-path)")
    parser.add_argument("--jsonl-path", type=str, default=None, help="Path to a JSONL dataset file (e.g. donotanswer_no_outputs.jsonl); mutually exclusive with --csv-path")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g. en, zh, ar)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoint and p1_p2_outputs (default: from guidelines-dir, e.g. guidelines_v4 -> output/p1_to_p2/p1_p2_guidelines_v4)",
    )
    parser.add_argument(
        "--guidelines-dir",
        type=str,
        default=None,
        help="Directory with category guideline .txt files (default: output/guidelines)",
    )
    parser.add_argument(
        "--guidelines-mode",
        type=str,
        choices=["full", "universal_only", "instructions_only"],
        default="full",
        help="full: universal + category guidelines + category; universal_only: only universal guidelines (no category); instructions_only: no guidelines, only reframing instructions",
    )
    parser.add_argument(
        "--universal-guidelines",
        type=str,
        default="src/prompts/guidelines.txt",
        help="Path to universal guidelines (.txt plain text or JSON); used in full and universal_only modes",
    )
    parser.add_argument(
        "--rewriter-prompt",
        type=str,
        default=None,
        help="Path to safety rewriter prompt template (default: chosen by --guidelines-mode)",
    )
    parser.add_argument("--max-p1", type=int, default=None, help="Max P1 prompts to process (default: all)")
    parser.add_argument("--use-local-model", action="store_true", help="Use local model (vLLM when available)")
    parser.add_argument("--local-model", type=str, default=None, help="Local model path or HF id")
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Use HuggingFace backend only (skip vLLM). Use if vLLM fails due to numba/NumPy; fix with: pip install 'numba>=0.61'",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Write to output and checkpoint files every N prompts (default: 50)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging (set WANDB_API_KEY or run wandb login).",
    )
    args = parser.parse_args()

    root = REPO_ROOT
    guidelines_mode = args.guidelines_mode
    # Default rewriter prompt path from guidelines mode when not overridden
    if args.rewriter_prompt is not None:
        rewriter_path = root / args.rewriter_prompt
    elif guidelines_mode == "full":
        rewriter_path = root / "src/prompts/safety_rewriter_prompt.txt"
    elif guidelines_mode == "universal_only":
        rewriter_path = root / "src/prompts/safety_rewriter_prompt_universal_only.txt"
    else:
        rewriter_path = root / "src/prompts/safety_rewriter_prompt_instructions_only.txt"

    guidelines_dir = root / (args.guidelines_dir or "output/guidelines")
    guidelines_dir_str = str(guidelines_dir).replace("\\", "/")
    # Default output dir: separate dir per mode so runs never overwrite each other
    if args.output_dir is not None:
        output_dir = root / args.output_dir
    elif guidelines_mode == "universal_only":
        output_dir = root / "output/p1_to_p2/p1_p2_guidelines_universal_only"
    elif guidelines_mode == "instructions_only":
        output_dir = root / "output/p1_to_p2/p1_p2_guidelines_instructions_only"
    elif "guidelines_v4" in guidelines_dir_str:
        output_dir = root / "output/p1_to_p2/p1_p2_guidelines_v4"
    elif "guidelines_v3" in guidelines_dir_str:
        output_dir = root / "output/p1_to_p2/p1_p2_guidelines_v3"
    elif "guidelines_v2" in guidelines_dir_str:
        output_dir = root / "output/p1_to_p2/p1_p2_guidelines_v2"
    elif "guidelines_openai_one" in guidelines_dir_str:
        output_dir = root / "output/p1_to_p2/p1_p2_guidelines_openai_one"
    else:
        output_dir = root / "output/p1_to_p2/p1_p2_guidelines"
    # Create output directory only if it does not exist (never wipe or overwrite existing)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output dir: {output_dir}")
    else:
        print(f"Output dir already exists: {output_dir}")
    universal_path = root / args.universal_guidelines

    # Version- and mode-specific checkpoint/output so runs do not overwrite each other
    if "guidelines_v4" in guidelines_dir_str:
        base_suffix = "_v4"
    elif "guidelines_v3" in guidelines_dir_str:
        base_suffix = "_v3"
    elif "guidelines_v2" in guidelines_dir_str:
        base_suffix = "_v2"
    elif "guidelines_openai_one" in guidelines_dir_str:
        base_suffix = "_openai_one"
    else:
        base_suffix = ""
    if guidelines_mode == "full":
        file_suffix = base_suffix
    elif guidelines_mode == "universal_only":
        file_suffix = base_suffix + "_universal_only"
    else:
        file_suffix = base_suffix + "_instructions_only"

    if args.use_local_model and args.local_model:
        os.environ["USE_LOCAL_MODEL"] = "true"
        os.environ["LOCAL_MODEL"] = args.local_model

    # Load universal guidelines only when used (full or universal_only)
    if guidelines_mode == "instructions_only":
        universal_text = ""
    else:
        universal_text = load_universal_guidelines(universal_path)
    template = rewriter_path.read_text(encoding="utf-8").strip()
    if args.jsonl_path:
        p1_records = load_p1_from_jsonl(Path(args.jsonl_path), args.max_p1)
    elif args.csv_path:
        p1_records = load_p1_from_csv(root / args.csv_path, args.language, args.max_p1)
    else:
        parser.error("Either --csv-path or --jsonl-path must be provided")
    checkpoint_file = output_dir / f"checkpoint{file_suffix}.jsonl"
    outputs_file = output_dir / f"p1_p2_outputs{file_suffix}.jsonl"
    lock_file = output_dir / f"write{file_suffix}.lock"

    # Optional wandb run (config only; metrics logged during loop)
    init_wandb(
        job_name=f"p1_to_p2_{args.language}",
        config={
            "language": args.language,
            "guidelines_mode": guidelines_mode,
            "guidelines_dir": str(guidelines_dir),
            "universal_guidelines": args.universal_guidelines,
            "output_dir": str(output_dir),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "checkpoint_every": args.checkpoint_every,
            "total_p1": len(p1_records),
        },
        enabled=args.wandb,
    )

    processed_ids = load_checkpoint(checkpoint_file)
    if processed_ids:
        print(f"Resuming: {len(processed_ids)} P1s already in checkpoint")

    print(f"Loaded {len(p1_records)} P1 prompts for language '{args.language}'")
    print(f"Guidelines mode: {guidelines_mode}")
    print(f"Rewriter prompt: {rewriter_path}")
    print(f"Guidelines dir: {guidelines_dir}")
    print(f"Output dir (checkpoint and outputs): {output_dir}")
    print(f"Checkpoint: {checkpoint_file}")
    print(f"Outputs: {outputs_file}")
    print(f"Checkpoint every: {args.checkpoint_every} prompts")

    success = 0
    failed = 0
    write_buffer: List[Dict] = []
    checkpoint_every = max(1, args.checkpoint_every)

    for idx, rec in enumerate(p1_records, 1):
        p1_id = rec.get("id", f"p1_{idx}")
        if str(p1_id) in processed_ids:
            print(f"[{idx}/{len(p1_records)}] {p1_id} already done, skipping.")
            continue

        p1 = rec["p1"]
        category = rec.get("type", "Unknown")
        if guidelines_mode == "full":
            category_guidelines = load_category_guidelines(guidelines_dir, category)
            user_prompt = build_rewriter_user_prompt(
                template, universal_text, category_guidelines, category, p1
            )
        elif guidelines_mode == "universal_only":
            user_prompt = build_rewriter_user_prompt_universal_only(template, universal_text, p1)
        else:
            user_prompt = build_rewriter_user_prompt_instructions_only(template, p1)

        try:
            raw_p2 = call_model(
                system_prompt=REWRITE_SYSTEM,
                user_prompt=user_prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                use_hf=args.use_hf,
                model_name=args.local_model if args.use_local_model else None,
            )
            p2, control_signals = parse_p2(raw_p2) if raw_p2 else ("", {})
        except ContentFilteredError:
            p2, control_signals = "", {}
            print(f"    Content filtered")
        except Exception as e:
            p2, control_signals = "", {}
            print(f"    Error: {e}")

        if not p2:
            failed += 1
        else:
            success += 1

        result = {
            "p1_id": p1_id,
            "p1": p1,
            "p2": p2 or None,
            "category": category,
            "lang": rec.get("lang"),
            "level": rec.get("level"),
        }
        # Persist control signals (mention_level derived from response_mode)
        for k, v in control_signals.items():
            result[k] = v
        write_buffer.append(result)
        processed_ids.add(str(p1_id))

        if len(write_buffer) >= checkpoint_every:
            append_jsonl(outputs_file, write_buffer, lock_file)
            append_jsonl(checkpoint_file, write_buffer, lock_file)
            print(f"Checkpoint: wrote {len(write_buffer)} results to file (total processed: {len(processed_ids)})", flush=True)
            log_metrics({"success": success, "failed": failed, "processed": len(processed_ids)}, step=idx)
            write_buffer = []

        print(f"[{idx}/{len(p1_records)}] {p1_id} -> {'OK' if p2 else 'FAIL'} (success={success}, failed={failed})", flush=True)

    if write_buffer:
        append_jsonl(outputs_file, write_buffer, lock_file)
        append_jsonl(checkpoint_file, write_buffer, lock_file)
        print(f"Checkpoint: wrote {len(write_buffer)} remaining results to file", flush=True)

    log_metrics({"success": success, "failed": failed, "total_processed": success + failed})
    finish_run()
    print(f"\nDone. Success: {success}, Failed: {failed}, Output: {outputs_file}")


if __name__ == "__main__":
    main()
