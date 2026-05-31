#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=judge_prefs_llama70b
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
RUN_TAG="${DATE_TAG}_judge_prefs_llama70b_${SLURM_JOB_ID}"

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

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/users/v/m/vmanoran/.cache/huggingface}"

# ===============================
# 2. Configuration
# ===============================
# Input file with preference pairs
INPUT_FILE=${INPUT_FILE:-${ROOT_DIR}/outputs/20251126_p1_p2_outputs_gpu_1152892/preference_pairs_tournament.json}

# Output file for judged pairs (can be same as input to update in place)
OUTPUT_FILE=${OUTPUT_FILE:-${ROOT_DIR}/outputs/20251126_p1_p2_outputs_gpu_1152892/preference_pairs_judged.json}

# Llama 70B model name (HuggingFace)
LLAMA_MODEL=${LLAMA_MODEL:-meta-llama/Llama-3-70B-Instruct}

# Processing range (for parallel processing or resuming)
START_IDX=${START_IDX:-0}
END_IDX=${END_IDX:-None}

# Batch size (1 is safest for large models, increase if you have enough GPU memory)
BATCH_SIZE=${BATCH_SIZE:-1}

# Memory optimization options
LOAD_IN_8BIT=${LOAD_IN_8BIT:-false}  # Set to "true" to use 8-bit quantization
LOAD_IN_4BIT=${LOAD_IN_4BIT:-false}   # Set to "true" to use 4-bit quantization

# Skip pairs that already have judge_scores
SKIP_JUDGED=${SKIP_JUDGED:-true}

echo "Configuration:"
echo "  Input file: ${INPUT_FILE}"
echo "  Output file: ${OUTPUT_FILE}"
echo "  Model: ${LLAMA_MODEL}"
echo "  Start index: ${START_IDX}"
echo "  End index: ${END_IDX}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Load in 8-bit: ${LOAD_IN_8BIT}"
echo "  Load in 4-bit: ${LOAD_IN_4BIT}"
echo "  Skip judged: ${SKIP_JUDGED}"

# Validate input file exists
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "ERROR: Input file not found: ${INPUT_FILE}"
  exit 1
fi

# ===============================
# 3. Activate environment and run
# ===============================
# Load modules if needed (adjust based on your cluster)
# module load python/3.9
# source venv/bin/activate  # if using virtual environment

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0

# ===============================
# 4. Run judging script
# ===============================
echo "Running judge_preference_pairs_llama70b.py..."
echo ""

# Build command arguments
CMD_ARGS=(
  --input "${INPUT_FILE}"
  --output "${OUTPUT_FILE}"
  --model "${LLAMA_MODEL}"
  --start-idx "${START_IDX}"
  --batch-size "${BATCH_SIZE}"
)

# Add end-idx if specified
if [[ "${END_IDX}" != "None" && -n "${END_IDX}" ]]; then
  CMD_ARGS+=(--end-idx "${END_IDX}")
fi

# Add quantization flags if enabled
if [[ "${LOAD_IN_8BIT}" == "true" ]]; then
  CMD_ARGS+=(--load-in-8bit)
fi

if [[ "${LOAD_IN_4BIT}" == "true" ]]; then
  CMD_ARGS+=(--load-in-4bit)
fi

# Add skip-judged flag if enabled
if [[ "${SKIP_JUDGED}" == "true" ]]; then
  CMD_ARGS+=(--skip-judged)
fi

echo "Command: python judge_preference_pairs_llama70b.py ${CMD_ARGS[*]}"
echo ""

python judge_preference_pairs_llama70b.py "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "Job finished at: $(date)"
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "✓ Judging completed successfully"
  echo "Results saved to: ${OUTPUT_FILE}"
else
  echo "✗ Judging failed with exit code: ${EXIT_CODE}"
fi
echo "Logs in: ${LOG_DIR}"

exit $EXIT_CODE

