#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=rank_outputs_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@institution.edu

# ===============================
# 1. Environment setup
# ===============================
ROOT_DIR="/nas/longleaf$HOME"
cd "$ROOT_DIR"

DATE_TAG=$(date +%Y%m%d)
RUN_TAG="${DATE_TAG}_rank_outputs_gpu_${SLURM_JOB_ID}"

LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_TAG}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Starting ranking GPU job on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Run tag: ${RUN_TAG}"
echo "Logs: ${LOG_DIR}"
echo "Outputs: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/users/v/m/vmanoran/.cache/huggingface}"

# ===============================
# 2. Configuration
# ===============================
# Input file: p1_p2_outputs.jsonl from the main experiment (1152892)
INPUT_FILE=${INPUT_FILE:-${ROOT_DIR}/outputs/20251126_p1_p2_outputs_gpu_1152892/p1_p2_outputs.jsonl}
LOCAL_MODEL=${LOCAL_MODEL:-Qwen/Qwen2.5-32B-Instruct}
MAX_P1=${MAX_P1:-}  # Optional: limit number of P1 prompts for testing

echo "Configuration:"
echo "  Input file: ${INPUT_FILE}"
echo "  Local model: ${LOCAL_MODEL}"
echo "  Output directory: ${OUTPUT_DIR}"
if [[ -n "${MAX_P1}" ]]; then
  echo "  Max P1 prompts: ${MAX_P1}"
fi

# ===============================
# 3. Activate environment and run
# ===============================
# Set environment variables for local model (GPU)
export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${LOCAL_MODEL}"
export CUDA_VISIBLE_DEVICES=0

# ===============================
# 4. Run ranking script
# ===============================
echo "Running rank_outputs_per_p1.py..."
echo ""

OUTPUT_FILE="${OUTPUT_DIR}/ranked_outputs.json"

# Build command arguments
CMD_ARGS=(
  --input "${INPUT_FILE}"
  --output "${OUTPUT_FILE}"
)

# Only add --max-p1 if it's set
if [[ -n "${MAX_P1}" ]]; then
  CMD_ARGS+=(--max-p1 "${MAX_P1}")
fi

python rank_outputs_per_p1.py "${CMD_ARGS[@]}"

echo ""
echo "Job finished at: $(date)"
echo "Results saved to: ${OUTPUT_FILE}"
echo "Logs in: ${LOG_DIR}"

