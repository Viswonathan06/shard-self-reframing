#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=rank_outputs_ar_gpu
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
RUN_TAG="${DATE_TAG}_rank_outputs_arabic_gpu_${SLURM_JOB_ID}"

LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_TAG}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Starting Arabic ranking GPU job on: $(hostname)"
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
# Input file: p1_p2_outputs.jsonl with Arabic data
# If you have a specific Arabic output file, set INPUT_FILE environment variable
# Otherwise, we'll try to find or filter from available files
INPUT_FILE=${INPUT_FILE:-${ROOT_DIR}/outputs/20251207_baseline_remaining_gpu_1204242/queue_all.jsonl}
LOCAL_MODEL=${LOCAL_MODEL:-Qwen/Qwen2.5-32B-Instruct}
MAX_P1=${MAX_P1:-}  # Optional: limit number of P1 prompts for testing
LANGUAGE=${LANGUAGE:-ar}  # Arabic language code

echo "Configuration:"
echo "  Input file: ${INPUT_FILE}"
echo "  Language: ${LANGUAGE}"
echo "  Local model: ${LOCAL_MODEL}"
echo "  Output directory: ${OUTPUT_DIR}"
if [[ -n "${MAX_P1}" ]]; then
  echo "  Max P1 prompts: ${MAX_P1}"
fi

# ===============================
# 3. Filter Arabic data if needed
# ===============================
# Check if input file contains Arabic data or needs filtering
FILTERED_INPUT="${OUTPUT_DIR}/p1_p2_outputs_arabic.jsonl"

echo ""
echo "Checking input file for Arabic data..."
ARABIC_COUNT=$(grep -c '"lang":\s*"ar"' "${INPUT_FILE}" 2>/dev/null || echo "0")
echo "  Found ${ARABIC_COUNT} Arabic entries in input file"

if [[ "${ARABIC_COUNT}" -eq "0" ]]; then
    echo "  WARNING: No Arabic entries found in input file!"
    echo "  Please ensure INPUT_FILE points to a file with Arabic p1_p2_outputs"
    echo "  Or run generate_p1_p2_outputs.py first for Arabic language"
    exit 1
fi

# If input is not already filtered, create filtered version
if [[ "${INPUT_FILE}" == *"queue_all"* ]] || [[ "${INPUT_FILE}" == *"all"* ]]; then
    echo "  Filtering Arabic entries from input file..."
    grep '"lang":\s*"ar"' "${INPUT_FILE}" > "${FILTERED_INPUT}" || {
        echo "  ERROR: Failed to filter Arabic entries"
        exit 1
    }
    FILTERED_COUNT=$(wc -l < "${FILTERED_INPUT}")
    echo "  Created filtered file with ${FILTERED_COUNT} Arabic entries: ${FILTERED_INPUT}"
    ACTUAL_INPUT="${FILTERED_INPUT}"
else
    echo "  Using input file directly (assuming it contains Arabic data)"
    ACTUAL_INPUT="${INPUT_FILE}"
fi

# ===============================
# 4. Activate environment and run
# ===============================
# Set environment variables for local model (GPU)
export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${LOCAL_MODEL}"
export CUDA_VISIBLE_DEVICES=0

# ===============================
# 5. Run ranking script
# ===============================
echo ""
echo "Running rank_outputs_per_p1.py for Arabic..."
echo ""

OUTPUT_FILE="${OUTPUT_DIR}/ranked_outputs_arabic.json"

# Build command arguments
CMD_ARGS=(
  --input "${ACTUAL_INPUT}"
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




