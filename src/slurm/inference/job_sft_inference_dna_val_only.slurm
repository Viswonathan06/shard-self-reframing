#!/usr/bin/bash
# SFT Llama 3.1 8B inference on DNA val split only (141 prompts).
#
# Usage:
#   sbatch src/slurm/inference/job_sft_inference_dna_val_only.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=sft_infer_dna_val
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-}"
if [[ -z "$ROOT_DIR" ]] && [[ -d "${SLURM_SUBMIT_DIR:-}/scripts" ]]; then
  ROOT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
fi
if [[ -z "$ROOT_DIR" ]]; then
  _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  _ROOT_CAND="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
  [[ -d "${_ROOT_CAND}/scripts" ]] && ROOT_DIR="${_ROOT_CAND}"
fi
if [[ -z "$ROOT_DIR" ]] || [[ ! -d "$ROOT_DIR/scripts" ]]; then
  echo "ERROR: Could not resolve repo root." && exit 1
fi
cd "$ROOT_DIR" || exit 1

source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TMPDIR="/tmp"

BASE_MODEL="${BASE_MODEL:-$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
SFT_ADAPTER="${SFT_ADAPTER:-${ROOT_DIR}/models/llama31_full_sft}"
INPUT_DATA="${INPUT_DATA:-${ROOT_DIR}/data/sft_val_only_prompts.jsonl}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/output/sft_evaluation/sft_val_only_inference.jsonl}"

if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"; PYTHON_CMD="python"
elif command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate llm_safety 2>/dev/null || true
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

if [[ ! -f "${INPUT_DATA}" ]]; then
  echo "ERROR: Missing ${INPUT_DATA}"
  echo "Run: python scripts/prepare_sft_val_eval_set.py"
  exit 1
fi

echo "=== SFT Inference (DNA val only) ==="
echo "Context: max_model_len=4096 max_new_tokens=2048 (scratchpad + answer need headroom)"
echo "ROOT:        ${ROOT_DIR}"
echo "SFT_ADAPTER: ${SFT_ADAPTER}"
echo "INPUT:       ${INPUT_DATA}"
echo "OUTPUT:      ${OUTPUT}"
echo ""
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

mkdir -p "$(dirname "${OUTPUT}")"

"${PYTHON_CMD}" scripts/inference_sft_vllm.py \
  --base-model              "${BASE_MODEL}" \
  --sft-adapter             "${SFT_ADAPTER}" \
  --input-data              "${INPUT_DATA}" \
  --output                  "${OUTPUT}" \
  --max-records             0 \
  --max-new-tokens          2048 \
  --temperature             0.0 \
  --repetition-penalty      1.15 \
  --tensor-parallel-size    4 \
  --gpu-memory-utilization  0.85 \
  --max-model-len           4096 \
  --lora-rank               16

echo ""
echo "=== Done ==="
wc -l "${OUTPUT}" || true
