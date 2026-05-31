"""
Shared utilities for baseline pipelines (P1 and P2).
Used by generate_baseline_outputs.py and generate_p2_baseline_outputs.py.
"""

import json
import os
import fcntl
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

load_dotenv()

from src.utils.model_client import call_model, ContentFilteredError

# Model config (read from env; callers set USE_LOCAL_MODEL and LOCAL_MODEL)
USE_LOCAL_MODEL_ENV = os.getenv("USE_LOCAL_MODEL", "false").strip().lower()
USE_LOCAL_MODEL = USE_LOCAL_MODEL_ENV == "true" or USE_LOCAL_MODEL_ENV == "1"
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct")

BASELINE_SYSTEM = "You are a helpful assistant. Provide a clear, helpful response to the user's question."


def load_jsonl(path: str) -> List[Dict]:
    """
    Load records from a JSONL-like file.

    Supports:
    - strict JSONL (one JSON object per line)
    - multiline pretty-printed JSON objects concatenated in a file
    """
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        buffer: List[str] = []
        depth = 0
        in_string = False
        escape = False

        for raw_line in f:
            if not raw_line.strip() and not buffer:
                continue

            line = raw_line.rstrip("\n")
            buffer.append(line)

            for ch in line:
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                else:
                    if ch == '"':
                        in_string = True
                    elif ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1

            # Completed one top-level object.
            if buffer and depth == 0 and not in_string:
                chunk = "\n".join(buffer).strip()
                if chunk:
                    records.append(json.loads(chunk))
                buffer = []

        if buffer:
            chunk = "\n".join(buffer).strip()
            if chunk:
                records.append(json.loads(chunk))

    return records


def save_jsonl(path: str, records: List[Dict]) -> None:
    """Save records to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, records: List[Dict], lock_file: Path) -> None:
    """Append records to JSONL file with locking."""
    if not records:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(path, "a", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def claim_next_index(cursor_file: Path, queue_size: int) -> Optional[int]:
    """Atomically claim next index from cursor file. Returns None if queue exhausted."""
    cursor_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = cursor_file.with_suffix(".lock")
    with open(lock_file, "a+") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            if cursor_file.exists():
                with open(cursor_file, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    current = int(first_line) if first_line else 0
            else:
                current = 0

            if current >= queue_size:
                return None

            next_idx = current
            with open(cursor_file, "w", encoding="utf-8") as f:
                f.write(f"{next_idx + 1}\n")

            return next_idx
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def load_checkpoint(checkpoint_file: Path) -> Set[str]:
    """Load processed P1 IDs from checkpoint file."""
    processed_ids: Set[str] = set()
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            p1_id = data.get("p1_id")
                            if p1_id:
                                processed_ids.add(str(p1_id))
                        except json.JSONDecodeError:
                            continue
            print(f"Loaded checkpoint: {len(processed_ids)} P1s already processed")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return processed_ids


def generate_model_response(prompt: str) -> Optional[str]:
    """
    Generate model response for a given prompt.
    Uses vLLM when USE_LOCAL_MODEL and LOCAL_MODEL are set via environment.

    Args:
        prompt: User prompt (P1 or P2)

    Returns:
        Model response or None if generation failed
    """
    try:
        kwargs: Dict = dict(
            system_prompt=BASELINE_SYSTEM,
            user_prompt=prompt,
            temperature=0.7,
            max_tokens=500,
            use_hf=False,
        )
        use_local = os.getenv("USE_LOCAL_MODEL", "false").strip().lower() in ("true", "1")
        local_model = os.getenv("LOCAL_MODEL")
        if use_local and local_model:
            kwargs["model_name"] = local_model
        response = call_model(**kwargs)

        if response and response.strip():
            return response.strip()
    except ContentFilteredError:
        print("    ⚠️  Content filtered by safety system")
        return None
    except Exception as e:
        print(f"    Error generating response: {e}")
        return None

    return None
