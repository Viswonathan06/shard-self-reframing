#!/usr/bin/bash
# Re-run SFT inference on failed records (lines 59-86)
#
# Usage: sbatch src/slurm/inference/job_sft_rerun_failed.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=sft_rerun_failed
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Repo root detection
ROOT_DIR="${ROOT_DIR:-}"
if [[ -z "$ROOT_DIR" ]] && [[ -d "${SLURM_SUBMIT_DIR:-}/scripts" ]]; then
  ROOT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
fi
if [[ -z "$ROOT_DIR" ]]; then
  _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  _ROOT_CAND="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
  if [[ -d "${_ROOT_CAND}/scripts" ]]; then
    ROOT_DIR="${_ROOT_CAND}"
  fi
fi
if [[ -z "$ROOT_DIR" ]] || [[ ! -d "$ROOT_DIR/scripts" ]]; then
  echo "ERROR: Could not resolve repo root. Submit from repo or set ROOT_DIR."
  exit 1
fi

cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

echo "=== Re-running SFT Inference on Failed Records ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Root: ${ROOT_DIR}"

# Input and output files (override via env on sbatch if needed)
INPUT_FILE="${INPUT_FILE:-${ROOT_DIR}/output/sft_rerun_input.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-${ROOT_DIR}/output/sft_rerun_results.jsonl}"
SFT_MODEL_PATH="${ROOT_DIR}/models/llama31_full_sft"

# Validate input
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "ERROR: Input file not found: ${INPUT_FILE}"
  exit 1
fi

if [[ ! -d "${SFT_MODEL_PATH}" ]]; then
  echo "ERROR: SFT model not found: ${SFT_MODEL_PATH}"
  exit 1
fi

RECORD_COUNT=$(wc -l < "${INPUT_FILE}")
echo "Re-running inference on ${RECORD_COUNT} failed records"

# Python setup
if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
  echo "✓ Using project .venv"
else
  PYTHON_CMD="python3"
  echo "✓ Using system python"
fi

# Set up HuggingFace cache and offline mode
export HF_HOME="${HF_HOME:-$HF_HOME}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
echo "✓ Using offline mode with HF_HOME=${HF_HOME}"

# GPU diagnostics
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

# Clean up any existing output
rm -f "${OUTPUT_FILE}"

# Run SFT inference
echo "Running SFT inference..."
"${PYTHON_CMD}" scripts/inference_sft_model.py \
  --base-model "$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
  --input-data "${INPUT_FILE}" \
  --output "${OUTPUT_FILE}" \
  --sft-adapter "${SFT_MODEL_PATH}" \
  --temperature 0.8 \
  --max-new-tokens 2048 \
  --max-records 0

if [[ -f "${OUTPUT_FILE}" ]]; then
  RESULT_COUNT=$(wc -l < "${OUTPUT_FILE}")
  echo "✓ Inference complete: ${RESULT_COUNT} records generated"
  echo "Results saved to: ${OUTPUT_FILE}"
else
  echo "ERROR: Output file not created"
  exit 1
fi

echo ""
echo "=== Re-inference Complete ==="
echo "Next: Merge these results back into the main SFT inference results file"

exit 0