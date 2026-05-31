#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --job-name=train_classifier
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
RUN_TAG="${DATE_TAG}_train_classifier_${SLURM_JOB_ID}"

LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
MODEL_DIR="${ROOT_DIR}/models/multihead_classifier"
mkdir -p "$LOG_DIR" "$MODEL_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Starting training job on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Model output: ${MODEL_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ===============================
# 2. Configuration
# ===============================
CSV_PATH=${CSV_PATH:-/nas/longleaf$HOME
OUTPUT_JSONL=${OUTPUT_JSONL:-${ROOT_DIR}/outputs/20251126_p1_p2_outputs_gpu_1152892/p1_p2_outputs.jsonl}
MODEL_NAME=${MODEL_NAME:-distilbert-base-uncased}  # Lightweight model
BATCH_SIZE=${BATCH_SIZE:-32}
EPOCHS=${EPOCHS:-5}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
TEST_SIZE=${TEST_SIZE:-0.2}

echo "Configuration:"
echo "  CSV path: ${CSV_PATH}"
echo "  Output JSONL: ${OUTPUT_JSONL}"
echo "  Model: ${MODEL_NAME}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  Learning rate: ${LEARNING_RATE}"
echo "  Test size: ${TEST_SIZE}"

# ===============================
# 3. Train classifier
# ===============================
echo "Starting training..."
srun python train_multihead_classifier.py \
  --csv-path "${CSV_PATH}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${MODEL_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --test-size "${TEST_SIZE}"

EXIT_CODE=$?

echo ""
echo "Training finished at: $(date)"
if [[ $EXIT_CODE -eq 0 ]]; then
  echo "✓ Training completed successfully"
  echo "Model saved to: ${MODEL_DIR}"
  
  # Run evaluation
  echo ""
  echo "Running evaluation..."
  srun python evaluate_classifier.py \
    --csv-path "${CSV_PATH}" \
    --model-path "${MODEL_DIR}" \
    --test-size 1000
else
  echo "✗ Training failed with exit code: ${EXIT_CODE}"
fi

echo "Logs in: ${LOG_DIR}"

exit $EXIT_CODE

