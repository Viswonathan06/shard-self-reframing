#!/usr/bin/bash
# DPO: Llama 3.1 8B — chosen = Qwen 480B generated_o2, rejected = Llama 8B P1 baseline output.
# Uses 4 GPUs via torchrun (DDP). Install deps once: pip install trl datasets peft
#
# Usage (from repo root):
#   sbatch src/slurm/dpo/job_dpo_llama31_8b_qwen_vs_baseline.slurm
#
# Overrides:
#   BASE_MODEL=/path/to/snapshot sbatch ...
#   DPO_DATA=data/dpo/qwen_chosen_vs_llama8b_rejected.jsonl sbatch ...
#   OUT_DIR=models/my_dpo_run sbatch ...

#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=dpo_llama31_qwen_vs_baseline
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
  if [[ -d "${_ROOT_CAND}/scripts" ]]; then
    ROOT_DIR="${_ROOT_CAND}"
  fi
fi
if [[ -z "$ROOT_DIR" ]] || [[ ! -d "$ROOT_DIR/scripts" ]]; then
  echo "ERROR: Could not resolve repo root."
  exit 1
fi

cd "$ROOT_DIR" || exit 1
source ~/.bashrc 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

BASE_MODEL="${BASE_MODEL:-$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
QWEN_JSONL="${QWEN_JSONL:-${ROOT_DIR}/output/qwen480b_scratchpad_llama8b_full.jsonl}"
BASELINE_JSONL="${BASELINE_JSONL:-${ROOT_DIR}/output/DNA_New_Experiments/dna_baseline_multimodel/llama8b/p1_baseline_outputs.jsonl}"
DPO_DATA="${DPO_DATA:-${ROOT_DIR}/data/dpo/qwen_chosen_vs_llama8b_rejected.jsonl}"
# v2: beta=0.05 (more deviation allowed), 3 epochs, lr=1e-6 (TRL default)
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/models/llama31_dpo_v2_beta005_3ep}"
DPO_BETA="${DPO_BETA:-0.05}"
DPO_LR="${DPO_LR:-1e-6}"
DPO_EPOCHS="${DPO_EPOCHS:-3}"

echo "=== DPO Llama 3.1 8B (4 GPUs) ==="
echo "ROOT: ${ROOT_DIR}"
echo "BASE_MODEL: ${BASE_MODEL}"
echo "OUT_DIR:    ${OUT_DIR}"

# Python
if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
elif command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate llm_safety 2>/dev/null || true
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

echo "Installing TRL stack if missing..."
"${PYTHON_CMD}" -m pip install -q "trl>=0.9.0" "datasets>=2.16.0" "peft>=0.10.0" || true

echo ""
echo "=== GPU status ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

echo "=== Step 1: build DPO JSONL ==="
mkdir -p "$(dirname "${DPO_DATA}")"
"${PYTHON_CMD}" scripts/prepare_dpo_qwen_vs_baseline.py \
  --qwen-jsonl "${QWEN_JSONL}" \
  --baseline-jsonl "${BASELINE_JSONL}" \
  --output-jsonl "${DPO_DATA}"

echo ""
echo "=== Step 2: torchrun DPO (4 processes) ==="
# DeepSpeed probes op compatibility at import time via temp files.
# On NFS (/common/guam/tmp), shutil.rmtree races and fails with ENOTEMPTY.
# Redirect TMPDIR to node-local /tmp and disable DS op building to avoid this.
export TMPDIR="/tmp"
export TEMP="/tmp"
export TMP="/tmp"
export DS_BUILD_OPS=0
export DS_BUILD_ASYNC_IO=0
# Clearer failures than bare elastic "exitcode 1":
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

RANK_LOG_DIR="${ROOT_DIR}/logs/dpo_rank_logs_${SLURM_JOB_ID}"
mkdir -p "${RANK_LOG_DIR}"
echo "DPO hyperparams: beta=${DPO_BETA}  lr=${DPO_LR}  epochs=${DPO_EPOCHS}"

torchrun --standalone --nproc_per_node=4 \
  --log-dir "${RANK_LOG_DIR}" --redirects 3 \
  scripts/train_dpo_llama31_8b.py \
  --dataset-jsonl "${DPO_DATA}" \
  --base-model "${BASE_MODEL}" \
  --output-dir "${OUT_DIR}" \
  --beta "${DPO_BETA}" \
  --learning-rate "${DPO_LR}" \
  --num-epochs "${DPO_EPOCHS}" \
  --bf16

echo ""
echo "=== Done ==="
echo "Adapter + tokenizer: ${OUT_DIR}"
exit 0
