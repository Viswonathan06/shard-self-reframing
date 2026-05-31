#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name=p2_qwen7b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@institution.edu

# P2: Qwen2.5-7B-Instruct on filtered P2 (safe rewrite) prompts

ROOT_DIR="/nas/longleaf$HOME"
cd "$ROOT_DIR"

DATE_TAG=$(date +%Y%m%d)
RUN_TAG="${DATE_TAG}_p2_qwen7b_${SLURM_JOB_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
OUT_DIR="${ROOT_DIR}/output/p2_baselines/p2_qwen7b"
P2_INPUT="${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines/p1_p2_outputs_filtered.jsonl"
mkdir -p "$LOG_DIR" "$OUT_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "P2 (Qwen2.5-7B) on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID | Start: $(date)"
echo "P2 input: ${P2_INPUT}"
echo "Output: ${OUT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/users/v/m/vmanoran/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES=0
export USE_LOCAL_MODEL=true

MODEL_PATH="${HF_HOME}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: Model path not found: $MODEL_PATH"
  exit 1
fi

python src/baselines/generate_p2_baseline_outputs.py \
  --input-jsonl "${P2_INPUT}" \
  --output-dir "${OUT_DIR}" \
  --use-local-model \
  --local-model "${MODEL_PATH}"

EXIT_CODE=$?
echo "Finished at: $(date) | Exit: $EXIT_CODE"
exit $EXIT_CODE
