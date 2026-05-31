#!/usr/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=compare_p1_p2_similarity_gpt5-4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || {
    echo "ERROR: Cannot cd to $ROOT_DIR"
    exit 1
}

source ~/.bashrc 2>/dev/null || true

LOGS_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOGS_DIR"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Prefer project venv (CUDA-enabled) for torch/sentence-transformers
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    source "${ROOT_DIR}/.venv/bin/activate"
    PYTHON_CMD="python"
    echo "Using project .venv"
elif command -v python &> /dev/null && [[ $(which python) == *"anaconda3"* ]]; then
    PYTHON_CMD="python"
elif [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "Python command: ${PYTHON_CMD}"
echo "PYTHONPATH: ${PYTHONPATH}"

echo "=== GPU diagnostics (current user: $USER, job id: ${SLURM_JOB_ID:-N/A}) ==="
nvidia-smi 2>/dev/null || true
echo "=== end GPU diagnostics ==="

INPUT_JSONL="${ROOT_DIR}/output/p1_p2_outputs_gpt5-4.jsonl"
OUTPUT_DIR="${ROOT_DIR}/output/p1_p2_similarity_gpt5-4"

mkdir -p "${OUTPUT_DIR}"

echo "Running compare_p1_p2_similarity.py on GPU node..."
${PYTHON_CMD} scripts/compare_p1_p2_similarity.py \
  --input "${INPUT_JSONL}" \
  --output-dir "${OUTPUT_DIR}"

EXIT_CODE=$?
exit $EXIT_CODE
