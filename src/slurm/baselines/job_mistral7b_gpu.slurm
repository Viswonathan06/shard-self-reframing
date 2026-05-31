#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name=baseline_en_mistral7b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@institution.edu

# Baseline English: Mistral-7B-Instruct (P1 prompts)

ROOT_DIR="/nas/longleaf$HOME"
cd "$ROOT_DIR"

DATE_TAG=$(date +%Y%m%d)
RUN_TAG="${DATE_TAG}_baseline_en_mistral7b_${SLURM_JOB_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
OUT_DIR="${ROOT_DIR}/output/baseline_en_mistral7b"
mkdir -p "$LOG_DIR" "$OUT_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Baseline English (Mistral-7B) on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "Output: ${OUT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/users/v/m/vmanoran/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES=0
export USE_LOCAL_MODEL=true

MODEL_PATH="${HF_HOME}/hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: Model path not found: $MODEL_PATH"
  exit 1
fi

python src/baselines/generate_baseline_outputs.py \
  --csv-path "${ROOT_DIR}/dataset/linguasafe.csv" \
  --language en \
  --output-dir "${OUT_DIR}" \
  --use-local-model \
  --local-model "${MODEL_PATH}"

EXIT_CODE=$?
echo "Finished at: $(date) | Exit: $EXIT_CODE"
exit $EXIT_CODE
