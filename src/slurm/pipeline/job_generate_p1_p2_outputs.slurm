#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=4
#SBATCH --mem=256G
#SBATCH --job-name=p1_p2_outputs
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
RUN_TAG="${DATE_TAG}_p1_p2_outputs_${SLURM_JOB_ID}"

LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_TAG}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Starting job on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Run tag: ${RUN_TAG}"
echo "Logs: ${LOG_DIR}"
echo "Outputs: ${OUTPUT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/users/v/m/vmanoran/.cache/huggingface}"

# ===============================
# 2. Configuration
# ===============================
LANGUAGE=${LANGUAGE:-en}
MAX_P1=${MAX_P1:-20}  # Start with 20 prompts for testing
CSV_PATH=${CSV_PATH:-/nas/longleaf$HOME
PPRIME_FILE=${PPRIME_FILE:-${ROOT_DIR}/appendix/prompts/pprime_by_category.jsonl}
SANITIZER_TERMS=${SANITIZER_TERMS:-${ROOT_DIR}/config/terms_${LANGUAGE}.json}
LOCAL_MODEL=${LOCAL_MODEL:-Qwen/Qwen2.5-32B-Instruct}

echo "Configuration:"
echo "  Language: ${LANGUAGE}"
echo "  Max P1 prompts: ${MAX_P1}"
echo "  CSV path: ${CSV_PATH}"
echo "  P' file: ${PPRIME_FILE}"
echo "  Sanitizer terms: ${SANITIZER_TERMS}"
echo "  Local model: ${LOCAL_MODEL}"
echo "  Output directory: ${OUTPUT_DIR}"

# ===============================
# 3. Activate environment and run
# ===============================
# Load modules if needed (adjust based on your cluster)
# module load python/3.9
# source venv/bin/activate  # if using virtual environment

# Set environment variables for local model
export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${LOCAL_MODEL}"

# ===============================
# 4. Create queue file
# ===============================
QUEUE_FILE="${OUTPUT_DIR}/queue_${LANGUAGE}.jsonl"
CURSOR_FILE="${LOG_DIR}/cursor_${LANGUAGE}.txt"
LOCK_FILE="${LOG_DIR}/write_lock_${LANGUAGE}.lock"
CHECKPOINT_FILE="${OUTPUT_DIR}/checkpoint.jsonl"

echo "Creating queue file for language ${LANGUAGE}..."
python - <<'PY' "${CSV_PATH}" "${QUEUE_FILE}" "${LANGUAGE}" "${MAX_P1}"
import json, sys, pandas as pd
csv_path, queue_file, language, max_p1 = sys.argv[1:]
max_p1 = int(max_p1) if max_p1 and max_p1 != "None" else None

df = pd.read_csv(csv_path)
df_filtered = df[df['lang'] == language].copy()

if max_p1 and max_p1 > 0:
    df_filtered = df_filtered.head(max_p1)

with open(queue_file, "w", encoding="utf-8") as f:
    for _, row in df_filtered.iterrows():
        rec = {
            "id": str(row.get("id", "")),
            "p1": str(row.get("prompt", "")),
            "lang": str(row.get("lang", language)),
            "level": row.get("level", None),
            "subtype": str(row.get("subtype", "")) if pd.notna(row.get("subtype")) else "",
            "type": str(row.get("type", "")) if pd.notna(row.get("type")) else "",
            "source": str(row.get("source", "")) if pd.notna(row.get("source")) else "",
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Created queue file with {len(df_filtered)} records")
PY

if [[ ! -s "${QUEUE_FILE}" ]]; then
  echo "ERROR: Queue file is empty. Exiting."
  exit 1
fi

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
# 5. Run with queue-based processing
# ===============================
echo "Running generate_p1_p2_outputs.py with queue-based processing..."
echo ""

srun python src/prompt_generation/generate_p1_p2_outputs.py \
  --csv-path "${CSV_PATH}" \
  --language "${LANGUAGE}" \
  --pprime-file "${PPRIME_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-p1 "${MAX_P1}" \
  --sanitizer-terms "${SANITIZER_TERMS}" \
  --use-local-model \
  --local-model "${LOCAL_MODEL}" \
  --queue-file "${QUEUE_FILE}" \
  --cursor-file "${CURSOR_FILE}" \
  --write-lock-file "${LOCK_FILE}" \
  --checkpoint-file "${CHECKPOINT_FILE}"

echo ""
echo "Job finished at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs in: ${LOG_DIR}"

