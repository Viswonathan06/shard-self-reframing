#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=gen_guidelines_v2_v3_70b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodelist=gpu4

# Generate category-specific guidelines for v2 and v3 using LLaMA 70B.
# Same setup as job_p1_to_p2_guidelines_gpu.slurm: 4 GPUs, vLLM 4-way TP (or HuggingFace device_map=auto via accelerate).
# Output: output/guidelines/guidelines_v2/ and output/guidelines/guidelines_v3/ (auto from --universal-guidelines).
# If you see "CUDA error: device-side assert triggered", re-run with: export CUDA_LAUNCH_BLOCKING=1 (before sbatch) to get the real failing line.

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }

source ~/.bashrc 2>/dev/null || true

# Same as p1_to_p2: prefer project venv (numpy<2.4 for vLLM/numba)
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    source "${ROOT_DIR}/.venv/bin/activate"
    PYTHON_CMD="python"
    echo "Using project .venv (numpy<2.4 for vLLM)"
elif command -v python &> /dev/null && [[ $(which python) == *"anaconda3"* ]]; then
    PYTHON_CMD="python"
elif [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    PYTHON_CMD="python3"
fi

DATE_TAG=$(date +%Y%m%d)
RUN_TAG="${DATE_TAG}_gen_guidelines_v2_v3_llama70b_${SLURM_JOB_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
mkdir -p "$LOG_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Starting GPU job on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Run tag: ${RUN_TAG}"
echo "Logs: ${LOG_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [ -d "/playpen" ]; then
    echo ""
    echo "========== /playpen space =========="
    df -h /playpen 2>/dev/null || true
    echo ""
    echo "========== /playpen contents =========="
    ls -la /playpen 2>/dev/null || true
    if [ -d "/playpen/huggingface" ]; then
        echo ""
        echo "========== /playpen/huggingface usage =========="
        du -sh /playpen/huggingface 2>/dev/null || true
        ls -la /playpen/huggingface 2>/dev/null || true
    fi
    echo "========================================"
    echo ""
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
if [ -d "/playpen" ]; then
    export HF_HOME="${HF_HOME:-/playpen/huggingface}"
else
    export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
fi

# Same as p1_to_p2: vLLM 4-way TP, or HuggingFace device_map=auto (accelerate)
export USE_LOCAL_MODEL=true
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85

# Use existing LLaMA 3.3 70B Instruct (override with LOCAL_MODEL_SNAPSHOT if needed)
LLAMA_MODEL="${LOCAL_MODEL_SNAPSHOT:-$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}"
if [ ! -d "${LLAMA_MODEL}" ]; then
    echo "ERROR: Model snapshot not found: ${LLAMA_MODEL}"
    echo "Set LOCAL_MODEL_SNAPSHOT to a valid path, or ensure the model exists on this node."
    exit 1
fi
echo "Using model: ${LLAMA_MODEL}"
export LOCAL_MODEL="${LLAMA_MODEL}"

LINGUASAFE_CSV=${LINGUASAFE_CSV:-${ROOT_DIR}/dataset/linguasafe.csv}
GUIDELINES_V1="${ROOT_DIR}/src/prompts/guidelines.txt"
GUIDELINES_V2="${ROOT_DIR}/src/prompts/guidelines_v2.jsonl"
GUIDELINES_V3="${ROOT_DIR}/src/prompts/guidelines_v3.jsonl"

echo "Configuration:"
echo "  Model: ${LLAMA_MODEL}"
echo "  Linguasafe CSV: ${LINGUASAFE_CSV}"
echo "  v1 guidelines: ${GUIDELINES_V1}"
echo "  v2 guidelines: ${GUIDELINES_V2}"
echo "  v3 guidelines: ${GUIDELINES_V3}"
echo "  (output dirs auto: output/guidelines/guidelines, output/guidelines/guidelines_v2, output/guidelines/guidelines_v3)"
echo ""

if [[ ! -f "${LINGUASAFE_CSV}" ]]; then
  echo "ERROR: Linguasafe CSV not found: ${LINGUASAFE_CSV}"
  exit 1
fi
if [[ ! -f "${GUIDELINES_V1}" ]]; then
  echo "ERROR: Guidelines file not found: ${GUIDELINES_V1}"
  exit 1
fi

# --- Run v1 (guidelines.txt) ---
# Same invocation as p1_to_p2: --use-local-model --local-model so vLLM is used (same backend as working job)
echo "========== Running v1 (guidelines.txt) =========="
$PYTHON_CMD src/prompt_generation/generate_category_guidelines.py \
  --linguasafe "${LINGUASAFE_CSV}" \
  --universal-guidelines "${GUIDELINES_V1}" \
  --use-local-model \
  --local-model "${LLAMA_MODEL}" \
  --temperature 0.3 \
  --max-tokens 800

EXIT_V1=$?
if [[ $EXIT_V1 -ne 0 ]]; then
  echo "✗ v1 failed with exit code: ${EXIT_V1}"
  exit $EXIT_V1
fi
echo "✓ v1 done -> output/guidelines/"
echo ""


echo "Job finished at: $(date)"
echo "✓ v1 category guidelines generated (LLaMA 70B)"
echo "  output/guidelines/"
echo "Logs in: ${LOG_DIR}"

exit 0
