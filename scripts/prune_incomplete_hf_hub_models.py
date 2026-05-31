#!/usr/bin/env python3
"""
Remove incomplete Hugging Face hub model trees (models--org--name) that have no
snapshot containing both config.json and model weights.

Default: dry-run (print only). Pass --delete to remove directories.

Examples:
  HF_HOME=/playpen/$USER/huggingface python scripts/prune_incomplete_hf_hub_models.py
  python scripts/prune_incomplete_hf_hub_models.py --hub-root $HF_HOME --delete
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def snapshot_has_weights(snap: Path) -> bool:
    if not (snap / "config.json").is_file():
        return False
    if (snap / "model.safetensors").is_file() or (snap / "pytorch_model.bin").is_file():
        return True
    if (snap / "model.safetensors.index.json").is_file():
        return True
    return any(snap.glob("model-*.safetensors"))


def model_tree_complete(model_dir: Path) -> bool:
    snap_root = model_dir / "snapshots"
    if not snap_root.is_dir():
        return False
    for snap in sorted(snap_root.iterdir(), reverse=True):
        if snap.is_dir() and snapshot_has_weights(snap):
            return True
    return False


def iter_model_dirs(hub_root: Path) -> list[Path]:
    if not hub_root.is_dir():
        return []
    out: list[Path] = []
    for p in hub_root.iterdir():
        if p.is_dir() and p.name.startswith("models--"):
            out.append(p)
    return sorted(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--hub-root",
        action="append",
        dest="hub_roots",
        default=None,
        help="Hub directory (…/hub or …/hf_cache containing models--*). Repeatable.",
    )
    ap.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete incomplete model directories (default: list only).",
    )
    args = ap.parse_args()

    roots: list[Path] = []
    if args.hub_roots:
        roots = [Path(r).expanduser().resolve() for r in args.hub_roots]
    else:
        hf_home = os.environ.get("HF_HOME", "").strip()
        if hf_home:
            roots.append(Path(hf_home).expanduser().resolve() / "hub")
        roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        roots.append(Path("$HF_HOME"))
        roots.append(Path("$HF_HOME"))

    seen: set[Path] = set()
    uniq_roots: list[Path] = []
    for r in roots:
        if r in seen:
            continue
        seen.add(r)
        uniq_roots.append(r)

    incomplete: list[tuple[Path, Path]] = []
    for hub in uniq_roots:
        for md in iter_model_dirs(hub):
            if model_tree_complete(md):
                continue
            incomplete.append((hub, md))

    if not incomplete:
        print("No incomplete models--* trees found under the given hub roots.", file=sys.stderr)
        return 0

    for hub, md in incomplete:
        print(f"{'DELETE' if args.delete else 'WOULD_DELETE'}\t{md}")

    if not args.delete:
        print("\nDry-run only. Re-run with --delete to remove these directories.", file=sys.stderr)
        return 0

    for hub, md in incomplete:
        shutil.rmtree(md, ignore_errors=False)
        print(f"Removed\t{md}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
