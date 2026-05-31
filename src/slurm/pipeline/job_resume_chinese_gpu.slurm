#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=resume_zh_gpu
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
RUN_TAG="${DATE_TAG}_resume_chinese_gpu_${SLURM_JOB_ID}"

LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
OUTPUT_DIR="${ROOT_DIR}/outputs/20251214_p1_p2_outputs_chinese_gpu_1234626"  # Use existing output dir
mkdir -p "$LOG_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Resuming Chinese P1->P2->Outputs GPU job on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Run tag: ${RUN_TAG}"
echo "Logs: ${LOG_DIR}"
echo "Output directory: ${OUTPUT_DIR} (resuming from previous run)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/users/v/m/vmanoran/.cache/huggingface}"

# ===============================
# 2. Configuration
# ===============================
LANGUAGE=${LANGUAGE:-zh}  # Chinese
CSV_PATH=${CSV_PATH:-/nas/longleaf$HOME
PPRIME_FILE=${PPRIME_FILE:-${ROOT_DIR}/appendix/prompts/pprime_by_category.jsonl}
SANITIZER_TERMS=${SANITIZER_TERMS:-${ROOT_DIR}/config/terms_zh.json}
LOCAL_MODEL=${LOCAL_MODEL:-Qwen/Qwen2.5-32B-Instruct}

# Use existing checkpoint and queue files
QUEUE_FILE="${OUTPUT_DIR}/queue_${LANGUAGE}.jsonl"
CHECKPOINT_FILE="${OUTPUT_DIR}/checkpoint.jsonl"
CURSOR_FILE="${OUTPUT_DIR}/../logs/20251214_p1_p2_outputs_chinese_gpu_1234626/cursor_${LANGUAGE}.txt"
LOCK_FILE="${LOG_DIR}/write_lock_${LANGUAGE}.lock"

echo "Configuration:"
echo "  Language: ${LANGUAGE} (Chinese)"
echo "  CSV path: ${CSV_PATH}"
echo "  P' file: ${PPRIME_FILE}"
echo "  Sanitizer terms: ${SANITIZER_TERMS}"
echo "  Local model: ${LOCAL_MODEL}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Checkpoint file: ${CHECKPOINT_FILE}"
echo "  Queue file: ${QUEUE_FILE}"

# Check if checkpoint exists
if [[ ! -f "${CHECKPOINT_FILE}" ]]; then
  echo "ERROR: Checkpoint file not found: ${CHECKPOINT_FILE}"
  exit 1
fi

# Count processed prompts
PROCESSED_COUNT=$(wc -l < "${CHECKPOINT_FILE}" 2>/dev/null || echo "0")
echo "  Already processed: ${PROCESSED_COUNT} prompts"

# ===============================
# 3. Activate environment and run
# ===============================
export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${LOCAL_MODEL}"
export CUDA_VISIBLE_DEVICES=0

# Initialize lock file if needed
: > "${LOCK_FILE}"

# ===============================
# 4. Run with queue-based processing
# ===============================
echo ""
echo "Resuming generate_p1_p2_outputs.py for Chinese with queue-based processing on GPU..."
echo ""

CMD_ARGS=(
  --csv-path "${CSV_PATH}"
  --language "${LANGUAGE}"
  --pprime-file "${PPRIME_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --sanitizer-terms "${SANITIZER_TERMS}"
  --use-local-model
  --local-model "${LOCAL_MODEL}"
  --queue-file "${QUEUE_FILE}"
  --cursor-file "${CURSOR_FILE}"
  --write-lock-file "${LOCK_FILE}"
  --checkpoint-file "${CHECKPOINT_FILE}"
)

python src/prompt_generation/generate_p1_p2_outputs.py "${CMD_ARGS[@]}"

echo ""
echo "Job finished at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs in: ${LOG_DIR}"


