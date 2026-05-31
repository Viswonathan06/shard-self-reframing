"""
Weights & Biases (wandb) helpers for this project.
Use --wandb in scripts to enable logging; otherwise set WANDB_DISABLED=1 or WANDB_MODE=disabled.

Login at runtime (key is never committed): put your API key in project root file .wandb_key
(one line, no quotes). Or set env WANDB_KEY_FILE=/path/to/file. init_wandb() loads it and
calls wandb.login() before the first run.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


# Default project name for this repo
WANDB_PROJECT = "multi-lingual-alignment-llms"

# Default key file (project root); override with env WANDB_KEY_FILE
DEFAULT_WANDB_KEY_FILE = ".wandb_key"


def _ensure_wandb_login() -> None:
    """
    Ensure wandb can authenticate at runtime. Uses (in order):
    - WANDB_API_KEY already in environment
    - Key read from WANDB_KEY_FILE or .wandb_key in repo root
    Then calls wandb.login(key=...) if a key was found and not already set.
    """
    try:
        import wandb
    except ImportError:
        return
    if os.getenv("WANDB_API_KEY"):
        return
    key_file = os.getenv("WANDB_KEY_FILE", "").strip()
    if not key_file:
        # Repo root: assume we're in src/utils or similar
        for parent in Path(__file__).resolve().parents:
            if (parent / DEFAULT_WANDB_KEY_FILE).exists():
                key_file = str(parent / DEFAULT_WANDB_KEY_FILE)
                break
    if not key_file or not Path(key_file).exists():
        return
    try:
        key = Path(key_file).read_text(encoding="utf-8").strip()
        if key:
            os.environ["WANDB_API_KEY"] = key
            wandb.login(key=key)
    except Exception:
        pass


def is_wandb_enabled(flag: bool = False) -> bool:
    """True if wandb should run (not disabled by env or explicit off)."""
    if os.getenv("WANDB_DISABLED", "").strip().lower() in ("1", "true", "yes"):
        return False
    if os.getenv("WANDB_MODE", "").strip().lower() == "disabled":
        return False
    return flag


def init_wandb(
    job_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    enabled: bool = True,
) -> Optional[Any]:
    """
    Initialize a wandb run. Returns the run object or None if disabled.
    When enabled=False or WANDB_DISABLED/WANDB_MODE=disabled, skips init (no login needed).
    """
    try:
        import wandb
    except ImportError:
        return None

    if not enabled or not is_wandb_enabled(True):
        return None

    _ensure_wandb_login()

    try:
        run = wandb.init(
            project=project or WANDB_PROJECT,
            name=job_name,
            config=config or {},
        )
    except AttributeError as e:
        import logging
        logging.warning("wandb.init unavailable (%s) — continuing without WandB.", e)
        return None
    return run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to the current wandb run (no-op if no active run)."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except (ImportError, AttributeError):
        pass


def finish_run() -> None:
    """Finish the current wandb run."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except (ImportError, AttributeError):
        pass
