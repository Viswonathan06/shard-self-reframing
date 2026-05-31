"""
Upload all LinguaSafe adapter collections to Hugging Face Hub.

Collections:
  - teacher_nonthinking : qwen35_122b_teacher / combined_dna_linguasafe
  - teacher_thinking    : qwen35_122b_teacher / combined_dna_linguasafe_reasoning
  - rational            : rational_sft
  - shard               : self_model  (thinking variants for qwen)
"""

import json
import re
import shutil
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_USERNAME = "Vish0606"
BASE = Path(__file__).resolve().parents[1] / "output" / "SFT"

UPLOADS = [
    # (local_folder, repo_id)

    # Teacher (non-thinking)
    (BASE / "qwen35_122b_teacher/combined_dna_linguasafe/models/llama8b",
     f"{HF_USERNAME}/teacher-llama-8b"),
    (BASE / "qwen35_122b_teacher/combined_dna_linguasafe/models/llama70b",
     f"{HF_USERNAME}/teacher-llama-70b"),
    (BASE / "qwen35_122b_teacher/combined_dna_linguasafe/models/mistral7b",
     f"{HF_USERNAME}/teacher-mistral-7b"),
    (BASE / "qwen35_122b_teacher/combined_dna_linguasafe/models/mistral24b_rerun",
     f"{HF_USERNAME}/teacher-mistral-24b"),
    (BASE / "qwen35_122b_teacher/combined_dna_linguasafe/models/phi4",
     f"{HF_USERNAME}/teacher-phi-4"),

    # Teacher (thinking / reasoning)
    (BASE / "qwen35_122b_teacher/combined_dna_linguasafe_reasoning/models/qwen35_9b",
     f"{HF_USERNAME}/teacher-qwen-9b-thinking"),
    (BASE / "qwen35_122b_teacher/combined_dna_linguasafe_reasoning/models/qwen35_27b_thinking",
     f"{HF_USERNAME}/teacher-qwen-27b-thinking"),

    # RATIONAL baseline
    (BASE / "rational_sft/llama8b/models/llama8b",
     f"{HF_USERNAME}/rational-llama-8b"),
    (BASE / "rational_sft/llama70b/models/llama70b",
     f"{HF_USERNAME}/rational-llama-70b"),
    (BASE / "rational_sft/mistral7b/models/mistral7b",
     f"{HF_USERNAME}/rational-mistral-7b"),
    (BASE / "rational_sft/mistral24b/models/mistral24b",
     f"{HF_USERNAME}/rational-mistral-24b"),
    (BASE / "rational_sft/phi4/models/phi4",
     f"{HF_USERNAME}/rational-phi-4"),
    (BASE / "rational_sft/qwen35_9b/models/qwen35_9b",
     f"{HF_USERNAME}/rational-qwen-9b"),
    (BASE / "rational_sft/qwen35_27b/models/qwen35_27b",
     f"{HF_USERNAME}/rational-qwen-27b"),

    # SHARD (thinking variants for qwen)
    (BASE / "self_model/llama8b/models/llama8b",
     f"{HF_USERNAME}/shard-llama-8b"),
    (BASE / "self_model/llama70b/models/llama70b",
     f"{HF_USERNAME}/shard-llama-70b"),
    (BASE / "self_model/mistral7b/models/mistral7b",
     f"{HF_USERNAME}/shard-mistral-7b"),
    (BASE / "self_model/mistral24b/models/mistral24b",
     f"{HF_USERNAME}/shard-mistral-24b"),
    (BASE / "self_model/phi4/models/phi4",
     f"{HF_USERNAME}/shard-phi-4"),
    (BASE / "self_model/qwen35_9b_thinking/models/qwen35_9b_thinking",
     f"{HF_USERNAME}/shard-qwen-9b"),
    (BASE / "self_model/qwen35_27b_thinking/models/qwen35_27b_thinking",
     f"{HF_USERNAME}/shard-qwen-27b"),
]

IGNORE_PATTERNS = ["checkpoint-*", "*.bin", "final_eval.json"]


def local_path_to_hf_id(local_path: str) -> str:
    """Convert a local HF cache snapshot path to a hub model ID.

    e.g. /playpen/.../models--meta-llama--Llama-3.1-8B-Instruct/snapshots/abc
         -> meta-llama/Llama-3.1-8B-Instruct
    """
    m = re.search(r"models--([^/]+)--([^/]+)", local_path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return local_path  # fallback: leave unchanged


def fix_readme(folder: Path, tmpdir: Path) -> Path:
    """Copy folder to tmpdir with README.md base_model corrected."""
    src_readme = folder / "README.md"
    if not src_readme.exists():
        return folder

    cfg = json.loads((folder / "adapter_config.json").read_text())
    raw_path = cfg.get("base_model_name_or_path", "")
    hf_id = local_path_to_hf_id(raw_path)

    content = src_readme.read_text()
    # Replace every occurrence of the local path (and $HF_HOME variants)
    content = re.sub(
        r"\$HF_HOME/models--[^\s\"']+",
        hf_id,
        content,
    )
    content = re.sub(
        r"/(?:playpen|home|common)/[^\s\"']*models--[^\s\"']+",
        hf_id,
        content,
    )

    out = tmpdir / "README.md"
    out.write_text(content)
    return tmpdir


def main():
    api = HfApi()

    missing = [(p, r) for p, r in UPLOADS if not p.exists()]
    if missing:
        print("WARNING — these paths don't exist and will be skipped:")
        for p, r in missing:
            print(f"  {p}")

    uploads = [(p, r) for p, r in UPLOADS if p.exists()]
    print(f"\nUploading {len(uploads)} adapter collections...\n")

    for folder, repo_id in uploads:
        print(f"[{repo_id}]  {folder}")
        create_repo(repo_id, repo_type="model", exist_ok=True, private=False)

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            # Copy the whole folder, then overwrite README with fixed version
            shutil.copytree(folder, tmpdir / "adapter", dirs_exist_ok=True)
            fix_readme(folder, tmpdir / "adapter")

            api.upload_folder(
                folder_path=str(tmpdir / "adapter"),
                repo_id=repo_id,
                repo_type="model",
                ignore_patterns=IGNORE_PATTERNS,
            )

        print(f"  -> https://huggingface.co/{repo_id}\n")

    print("Done.")


if __name__ == "__main__":
    main()
