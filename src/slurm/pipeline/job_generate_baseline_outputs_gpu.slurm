#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=baseline_outputs_gpu
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
RUN_TAG="${DATE_TAG}_baseline_outputs_gpu_${SLURM_JOB_ID}"

LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_TAG}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Starting baseline GPU job on: $(hostname)"
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
LANGUAGE=${LANGUAGE:-en}
MAX_P1=${MAX_P1:-None}  # Not used when using selected_prompts_reference - all prompts from that file will be processed
CSV_PATH=${CSV_PATH:-/nas/longleaf$HOME
LOCAL_MODEL=${LOCAL_MODEL:-Qwen/Qwen2.5-32B-Instruct}
# Path to selected_prompts_reference.jsonl from 1152892 to use the exact same prompts
# This ensures we process all 2388 prompts from that experiment
SELECTED_PROMPTS_REF=${SELECTED_PROMPTS_REF:-${ROOT_DIR}/outputs/20251126_p1_p2_outputs_gpu_1152892/selected_prompts_reference.jsonl}

echo "Configuration:"
echo "  Language: ${LANGUAGE}"
echo "  Max P1 prompts: ${MAX_P1}"
echo "  CSV path: ${CSV_PATH}"
echo "  Local model: ${LOCAL_MODEL}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Selected prompts reference: ${SELECTED_PROMPTS_REF}"

# ===============================
# 3. Activate environment and run
# ===============================
# Load modules if needed (adjust based on your cluster)
# module load python/3.9
# source venv/bin/activate  # if using virtual environment

# Set environment variables for local model (GPU)
export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${LOCAL_MODEL}"
export CUDA_VISIBLE_DEVICES=0

# ===============================
# 4. Create queue file (using same P1 prompts as main experiment)
# ===============================
QUEUE_FILE="${OUTPUT_DIR}/queue_${LANGUAGE}.jsonl"
CURSOR_FILE="${LOG_DIR}/cursor_${LANGUAGE}.txt"
LOCK_FILE="${LOG_DIR}/write_lock_${LANGUAGE}.lock"
CHECKPOINT_FILE="${OUTPUT_DIR}/checkpoint.jsonl"

echo "Creating queue file from selected prompts reference (1152892)..."
python - <<'PY' "${SELECTED_PROMPTS_REF}" "${QUEUE_FILE}" "${LANGUAGE}"
import json, sys, os
selected_ref_file, queue_file, language = sys.argv[1:]

# Load selected prompts from 1152892 reference file
selected_prompts = []
if os.path.exists(selected_ref_file):
    print(f"Loading selected prompts from: {selected_ref_file}")
    with open(selected_ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    # Only include prompts matching the specified language
                    if rec.get('lang') == language:
                        selected_prompts.append(rec)
                except:
                    continue
    print(f"Loaded {len(selected_prompts)} prompts for language '{language}'")
else:
    print(f"ERROR: Selected prompts reference file not found: {selected_ref_file}")
    sys.exit(1)

# Create queue file directly from selected prompts
with open(queue_file, "w", encoding="utf-8") as f:
    for rec in selected_prompts:
        queue_rec = {
            "id": str(rec.get("id", "")),
            "p1": str(rec.get("p1", "")),
            "lang": str(rec.get("lang", language)),
            "level": rec.get("level", None),
            "subtype": str(rec.get("subtype", "")),
            "type": str(rec.get("type", "")),
            "source": str(rec.get("source", "")),
        }
        f.write(json.dumps(queue_rec, ensure_ascii=False) + "\n")

print(f"Created queue file with {len(selected_prompts)} records")
print(f"All prompts from 1152892 experiment will be processed")

# Save reference file (copy of selected prompts)
reference_file = os.path.join(os.path.dirname(queue_file), "selected_prompts_reference.jsonl")
with open(reference_file, "w", encoding="utf-8") as f:
    for rec in selected_prompts:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Saved reference file: {reference_file}")
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
# 5. Run baseline script with queue-based processing
# ===============================
echo "Running generate_baseline_outputs.py with queue-based processing on GPU..."
echo ""

# Build command with conditional --max-p1 argument
CMD_ARGS=(
  --csv-path "${CSV_PATH}"
  --language "${LANGUAGE}"
  --output-dir "${OUTPUT_DIR}"
  --use-local-model
  --local-model "${LOCAL_MODEL}"
  --queue-file "${QUEUE_FILE}"
  --cursor-file "${CURSOR_FILE}"
  --write-lock-file "${LOCK_FILE}"
  --checkpoint-file "${CHECKPOINT_FILE}"
)

# Only add --max-p1 if it's not "None" or empty
if [[ "${MAX_P1}" != "None" && -n "${MAX_P1}" ]]; then
  CMD_ARGS+=(--max-p1 "${MAX_P1}")
fi

python src/baselines/generate_baseline_outputs.py "${CMD_ARGS[@]}"

echo ""
echo "Job finished at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs in: ${LOG_DIR}"

