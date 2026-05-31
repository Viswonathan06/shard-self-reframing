#!/usr/bin/bash
# Inference with the DPO-trained Llama 3.1 8B adapter on all P1 prompts.
# Single GPU is enough (no DDP needed for inference).
#
# Usage:
#   sbatch src/slurm/inference/job_dpo_inference.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=dpo_inference
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
DPO_ADAPTER="${DPO_ADAPTER:-${ROOT_DIR}/models/llama31_dpo_v2_beta005_3ep}"
INPUT_DATA="${INPUT_DATA:-${ROOT_DIR}/output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/llama8b/baseline_outputs.jsonl}"
OUTPUT="${OUTPUT:-${ROOT_DIR}/output/dpo_evaluation/dpo_llama8b_inference.jsonl}"

# Python
if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"; PYTHON_CMD="python"
elif command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate llm_safety 2>/dev/null || true
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

echo "=== DPO Inference ==="
echo "ROOT:        ${ROOT_DIR}"
echo "DPO_ADAPTER: ${DPO_ADAPTER}"
echo "INPUT:       ${INPUT_DATA}"
echo "OUTPUT:      ${OUTPUT}"
echo ""
echo "=== GPU status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

mkdir -p "$(dirname "${OUTPUT}")"

"${PYTHON_CMD}" scripts/inference_dpo_vllm.py \
  --base-model              "${BASE_MODEL}" \
  --dpo-adapter             "${DPO_ADAPTER}" \
  --input-data              "${INPUT_DATA}" \
  --output                  "${OUTPUT}" \
  --max-records             0 \
  --max-new-tokens          512 \
  --temperature             0.0 \
  --repetition-penalty      1.2 \
  --tensor-parallel-size    4 \
  --gpu-memory-utilization  0.85 \
  --max-model-len           2048 \
  --lora-rank               16

echo ""
echo "=== Done ==="
echo "Output: ${OUTPUT}"
wc -l "${OUTPUT}" || true
