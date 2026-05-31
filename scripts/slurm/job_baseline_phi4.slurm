#!/usr/bin/bash
# Baseline inference for Phi-4 on DNA + LinguaSafe datasets.
# Downloads the model if not present on this node.
#
# Usage:
#   sbatch --nodelist=gpu1 scripts/slurm/job_baseline_phi4.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --job-name=baseline_phi4
#SBATCH --output=$PROJECT_ROOT/logs/baseline_phi4_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/baseline_phi4_%j.err

set -euo pipefail

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_PATH="$HF_HOME/models--microsoft--phi-4/snapshots/main"
HF_ID="microsoft/phi-4"

# ── Download model if not on this node ────────────────────────────────────────
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Phi-4 not found at $MODEL_PATH — downloading from HuggingFace..."
    "$PYTHON" -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='${HF_ID}',
    cache_dir='$HF_HOME',
)
print('Downloaded to:', path)
import os, pathlib
snap = pathlib.Path(path)
target = pathlib.Path('${MODEL_PATH}')
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    target.symlink_to(snap)
    print('Symlinked:', target, '->', snap)
"
    # Re-resolve: snapshot_download returns the actual cache path
    MODEL_PATH=$("$PYTHON" -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='${HF_ID}', cache_dir='$HF_HOME'))
")
    echo "Model path resolved to: $MODEL_PATH"
fi

echo "========================================================"
echo "Phi-4 baseline inference"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Model:      $MODEL_PATH"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

TP=1  # Phi-4 14B ~28GB bf16 fits on a single 49GB GPU

DNA_INPUT="output/DNA_Per_Model_Experiments/refinement_from_baseline_multimodel/llama8b/refinement_input.jsonl"
LS_INPUT="output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_multimodel/llama8b/refinement_input.jsonl"
DNA_OUTPUT="output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/phi4/p1_baseline_outputs.jsonl"
LS_OUTPUT="output/Linguasafe_Experiments/helpful_assistant_multimodel/phi4/p1_baseline_outputs.jsonl"

mkdir -p "$(dirname "$DNA_OUTPUT")" "$(dirname "$LS_OUTPUT")"

# ── DNA ───────────────────────────────────────────────────────────────────────
echo ">>> DNA baseline inference (939 rows)"
"$PYTHON" scripts/run_baseline_inference_vllm.py \
    --model                  "$MODEL_PATH" \
    --input                  "$DNA_INPUT" \
    --output                 "$DNA_OUTPUT" \
    --tensor-parallel-size   "$TP" \
    --gpu-memory-utilization 0.85 \
    --max-model-len          4096 \
    --max-new-tokens         2048 \
    --temperature            0.3 \
    --batch-size             256
echo ">>> DNA done — $DNA_OUTPUT"
echo ""

# ── LinguaSafe ────────────────────────────────────────────────────────────────
echo ">>> LinguaSafe baseline inference (3267 rows)"
"$PYTHON" scripts/run_baseline_inference_vllm.py \
    --model                  "$MODEL_PATH" \
    --input                  "$LS_INPUT" \
    --output                 "$LS_OUTPUT" \
    --tensor-parallel-size   "$TP" \
    --gpu-memory-utilization 0.85 \
    --max-model-len          4096 \
    --max-new-tokens         2048 \
    --temperature            0.3 \
    --batch-size             256
echo ">>> LinguaSafe done — $LS_OUTPUT"
echo ""

echo "========================================================"
echo "Phi-4 baseline inference complete"
echo "========================================================"
