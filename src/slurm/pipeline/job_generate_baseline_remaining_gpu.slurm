#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=baseline_remaining_gpu
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
RUN_TAG="${DATE_TAG}_baseline_remaining_gpu_${SLURM_JOB_ID}"

LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_TAG}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Starting baseline GPU job for remaining prompts on: $(hostname)"
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
CSV_PATH=${CSV_PATH:-/nas/longleaf$HOME
LOCAL_MODEL=${LOCAL_MODEL:-Qwen/Qwen2.5-32B-Instruct}
# Queue file with remaining prompts (English first, then other languages)
QUEUE_FILE_SOURCE=${QUEUE_FILE_SOURCE:-${ROOT_DIR}/remaining_prompts_queue.jsonl}

echo "Configuration:"
echo "  CSV path: ${CSV_PATH}"
echo "  Local model: ${LOCAL_MODEL}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Queue file source: ${QUEUE_FILE_SOURCE}"

# ===============================
# 3. Activate environment and run
# ===============================
# Set environment variables for local model (GPU)
export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${LOCAL_MODEL}"
export CUDA_VISIBLE_DEVICES=0

# ===============================
# 4. Setup queue file (copy from source)
# ===============================
QUEUE_FILE="${OUTPUT_DIR}/queue_all.jsonl"
CURSOR_FILE="${LOG_DIR}/cursor_all.txt"
LOCK_FILE="${LOG_DIR}/write_lock_all.lock"
CHECKPOINT_FILE="${OUTPUT_DIR}/checkpoint.jsonl"

echo "Setting up queue file from remaining prompts..."
if [[ ! -f "${QUEUE_FILE_SOURCE}" ]]; then
    echo "ERROR: Queue file source not found: ${QUEUE_FILE_SOURCE}"
    exit 1
fi

# Copy queue file to output directory
cp "${QUEUE_FILE_SOURCE}" "${QUEUE_FILE}"
QUEUE_COUNT=$(wc -l < "${QUEUE_FILE}")
echo "Queue file copied: ${QUEUE_COUNT} prompts"
echo "  English prompts will be processed first, followed by other languages"

# Count by language
python3 - <<'PY'
import json
import sys
from collections import Counter

queue_file = sys.argv[1]
langs = []
with open(queue_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            langs.append(rec.get('lang', 'unknown'))

lang_counts = Counter(langs)
print("\nLanguage breakdown:")
for lang, count in sorted(lang_counts.items()):
    print(f"  {lang}: {count}")
PY "${QUEUE_FILE}"

# Save reference file (copy of queue)
reference_file="${OUTPUT_DIR}/selected_prompts_reference.jsonl"
cp "${QUEUE_FILE}" "${reference_file}"
echo "Saved reference file: ${reference_file}"

# Initialize cursor and lock files
echo "0" > "${CURSOR_FILE}"
: > "${LOCK_FILE}"

echo ""
echo "Queue setup complete:"
echo "  Queue file: ${QUEUE_FILE}"
echo "  Cursor file: ${CURSOR_FILE}"
echo "  Lock file: ${LOCK_FILE}"
echo "  Checkpoint file: ${CHECKPOINT_FILE}"
echo ""

# ===============================
# 5. Run baseline script with queue-based processing
# ===============================
echo "Running generate_baseline_outputs.py with queue-based processing on GPU..."
echo "Processing all remaining prompts (English prioritized, then other languages)..."
echo ""

# Use 'all' as language code since we're processing multiple languages
# The script will use the queue file which contains all languages
CMD_ARGS=(
  --csv-path "${CSV_PATH}"
  --language "all"  # Placeholder - actual language comes from queue file
  --output-dir "${OUTPUT_DIR}"
  --use-local-model
  --local-model "${LOCAL_MODEL}"
  --queue-file "${QUEUE_FILE}"
  --cursor-file "${CURSOR_FILE}"
  --write-lock-file "${LOCK_FILE}"
  --checkpoint-file "${CHECKPOINT_FILE}"
)

python src/baselines/generate_baseline_outputs.py "${CMD_ARGS[@]}"

echo ""
echo "Job finished at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs in: ${LOG_DIR}"

