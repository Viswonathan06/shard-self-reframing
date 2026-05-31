#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=gen_cat_guidelines_llama70b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodelist=gpu1
# Generate category-specific safety guidelines (LLaMA 70B)

ROOT_DIR="$HOME"
cd "$ROOT_DIR"

DATE_TAG=$(date +%Y%m%d)
RUN_TAG="${DATE_TAG}_gen_cat_guidelines_llama70b_${SLURM_JOB_ID}"

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

# Use existing LLaMA 3.3 70B Instruct (override with LOCAL_MODEL_SNAPSHOT if needed)
LLAMA_MODEL="${LOCAL_MODEL_SNAPSHOT:-$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}"
if [ ! -d "${LLAMA_MODEL}" ]; then
    echo "ERROR: Model snapshot not found: ${LLAMA_MODEL}"
    echo "Set LOCAL_MODEL_SNAPSHOT to a valid path, or ensure the model exists on this node."
    exit 1
fi
echo "Using model: ${LLAMA_MODEL}"

LINGUASAFE_CSV=${LINGUASAFE_CSV:-${ROOT_DIR}/dataset/linguasafe.csv}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/output/guidelines}
UNIVERSAL_GUIDELINES=${UNIVERSAL_GUIDELINES:-${ROOT_DIR}/src/prompts/guidelines.txt}

export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${LLAMA_MODEL}"
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85

echo "Configuration:"
echo "  Model (vLLM): ${LLAMA_MODEL}"
echo "  Linguasafe CSV: ${LINGUASAFE_CSV}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  Universal guidelines: ${UNIVERSAL_GUIDELINES}"

if [[ ! -f "${LINGUASAFE_CSV}" ]]; then
  echo "ERROR: Linguasafe CSV not found: ${LINGUASAFE_CSV}"
  exit 1
fi

echo "Running generate_category_guidelines.py (LLaMA 3.3 70B, vLLM 4-way TP)..."
echo ""

python src/prompt_generation/generate_category_guidelines.py \
  --linguasafe "${LINGUASAFE_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --universal-guidelines "${UNIVERSAL_GUIDELINES}" \
  --use-local-model \
  --local-model "${LLAMA_MODEL}" \
  --temperature 0.2 \
  --max-tokens 800

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "✓ Category guidelines generated successfully"
  echo "Results in: ${OUTPUT_DIR}"
else
  echo "✗ Job failed with exit code: ${EXIT_CODE}"
fi
echo "Logs in: ${LOG_DIR}"

exit $EXIT_CODE
