#!/bin/bash
# Download / verify Qwen3.5-122B-A10B on gpu5's /playpen.
# Uses resume_download=True so it only fetches missing shards.
# After verifying all 32+ shards are present, reports disk usage
# and exits cleanly so the pipeline job can be safely resubmitted.
#
# Usage:
#   sbatch src/slurm/p2/job_qwen35_122b_download_verify.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:0
#SBATCH --nodelist=gpu5
#SBATCH --time=4:00:00
#SBATCH --job-name=qwen35_122b_dl
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
export REPO_ID="Qwen/Qwen3.5-122B-A10B"

export HF_HOME="/playpen/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}/hub"

CACHE_DIR="${HF_HOME}/hub"

echo "======================================================"
echo "Qwen3.5-122B-A10B  —  Download / Verify"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "Cache: ${CACHE_DIR}"
echo "======================================================"

"${PYTHON}" --version

SNAP_BASE="${CACHE_DIR}/models--${REPO_ID//\//--}"

# ── Show what's already there ──────────────────────────────────────────────────
if [[ -d "${SNAP_BASE}/snapshots" ]]; then
    echo ""
    echo "Existing snapshots:"
    find "${SNAP_BASE}/snapshots" -maxdepth 1 -mindepth 1 -type d | while read snap; do
        n_safe=$(find "${snap}" -name "*.safetensors" 2>/dev/null | wc -l)
        n_idx=$(find "${snap}" -name "model.safetensors.index.json" 2>/dev/null | wc -l)
        echo "  ${snap}  (safetensors=${n_safe}, index=${n_idx})"
    done
fi

# ── Resume / complete the download ─────────────────────────────────────────────
echo ""
echo "Resuming download of ${REPO_ID} ..."
# Only print the resolved path to stdout; everything else goes to stderr.
MODEL_PATH=$("${PYTHON}" - <<'PYEOF'
import os, sys
from huggingface_hub import snapshot_download

repo_id   = os.environ["REPO_ID"]
cache_dir = os.environ["HF_HUB_CACHE"]
token     = os.getenv("HF_TOKEN")

print(f"Downloading {repo_id} → {cache_dir} ...", file=sys.stderr, flush=True)
path = snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    token=token,
)
print(f"snapshot_download returned: {path}", file=sys.stderr, flush=True)
print(path)   # only this line goes to stdout → MODEL_PATH
PYEOF
)

echo ""
echo "Model path: ${MODEL_PATH}"

# ── Verify all safetensors shards listed in the index ─────────────────────────
echo ""
echo "Verifying shards ..."
export MODEL_PATH
"${PYTHON}" - <<'PYEOF'
import json, os, sys

model_path = os.environ["MODEL_PATH"]
index_file = os.path.join(model_path, "model.safetensors.index.json")

if not os.path.exists(index_file):
    print("No model.safetensors.index.json found — checking for single-file model:")
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        size = os.path.getsize(single) / (1024**3)
        print(f"  model.safetensors  {size:.2f} GB  OK")
    else:
        print("  ERROR: no safetensors found!", file=sys.stderr)
        sys.exit(1)
    sys.exit(0)

with open(index_file) as f:
    index = json.load(f)

shards = sorted(set(index["weight_map"].values()))
print(f"Index lists {len(shards)} shard files:")
missing = []
for shard in shards:
    fpath = os.path.join(model_path, shard)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath) / (1024**3)
        print(f"  {shard}  {size:.2f} GB  OK")
    else:
        print(f"  {shard}  MISSING", file=sys.stderr)
        missing.append(shard)

if missing:
    print(f"\nERROR: {len(missing)} shard(s) missing:", file=sys.stderr)
    for m in missing:
        print(f"  {m}", file=sys.stderr)
    sys.exit(1)
else:
    print(f"\nAll {len(shards)} shards present.")
PYEOF

echo ""
du -sh "${MODEL_PATH}" 2>/dev/null || true

echo ""
echo "======================================================"
echo "Download and verification complete: $(date)"
echo "  → ${MODEL_PATH}"
echo ""
echo "You can now resubmit the pipeline:"
echo "  sbatch src/slurm/p2/job_qwen35_122b_linguasafe_pipeline.slurm"
echo "======================================================"
