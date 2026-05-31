#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --job-name=baseline_en_multi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@institution.edu

# Baseline: run P1 prompts through all three models (Llama8B, Mistral7B, Qwen7B)

ROOT_DIR="/nas/longleaf$HOME"
cd "$ROOT_DIR"

DATE_TAG=$(date +%Y%m%d)
RUN_TAG="${DATE_TAG}_baseline_en_multi_${SLURM_JOB_ID}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
mkdir -p "$LOG_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Starting baseline English multi-model job on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Logs: ${LOG_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/users/v/m/vmanoran/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES=0
export USE_LOCAL_MODEL=true

CSV_PATH="${ROOT_DIR}/dataset/linguasafe.csv"
LANGUAGE=en

HF_HUB="${HF_HOME}/hub"
LLAMA_8B="${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
MISTRAL_7B="${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
QWEN_7B="${HF_HUB}/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"

for model_path in "$LLAMA_8B" "$MISTRAL_7B" "$QWEN_7B"; do
  if [[ ! -d "$model_path" ]]; then
    echo "ERROR: Model path not found: $model_path"
    exit 1
  fi
done

run_baseline() {
  local model_name=$1
  local model_path=$2
  local out_subdir=$3
  local out_dir="${ROOT_DIR}/output/baseline_en_${out_subdir}"
  mkdir -p "$out_dir"
  echo ""
  echo "=============================================="
  echo "Running baseline English: ${model_name}"
  echo "  Model: ${model_path}"
  echo "  Output: ${out_dir}"
  echo "=============================================="
  python src/baselines/generate_baseline_outputs.py \
    --csv-path "${CSV_PATH}" \
    --language "${LANGUAGE}" \
    --output-dir "${out_dir}" \
    --use-local-model \
    --local-model "${model_path}"
  local code=$?
  if [[ $code -ne 0 ]]; then
    echo "FAILED: ${model_name} exited with $code"
    return $code
  fi
  echo "OK: ${model_name} completed."
  return 0
}

run_baseline "Meta-Llama-3.1-8B" "$LLAMA_8B" "llama8b" || exit $?
run_baseline "Mistral-7B-Instruct" "$MISTRAL_7B" "mistral7b" || exit $?
run_baseline "Qwen2.5-7B-Instruct" "$QWEN_7B" "qwen7b" || exit $?

echo ""
echo "All three baseline runs finished at: $(date)"
echo "Outputs:"
echo "  output/baseline_en_llama8b/"
echo "  output/baseline_en_mistral7b/"
echo "  output/baseline_en_qwen7b/"
