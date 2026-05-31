#!/bin/bash
# DPO on Llama-3.1-8B-Instruct starting from the SFT checkpoint.
# Runs DPO training → held-out eval in one job.
#
# Run AFTER job_sft_linguasafe.slurm has completed.
#
# Usage:
#   SFT_RUN=sft_linguasafe_12345 sbatch src/slurm/dpo/job_dpo_linguasafe.slurm
#
# Overrides:
#   BASE_MODEL=/path/snapshot  SFT_MODEL_DIR=output/SFT/.../model  RUN_NAME=dpo_v2 sbatch ...

#SBATCH --job-name=dpo_linguasafe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --chdir=$PROJECT_ROOT

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || exit 1

echo "======================================================"
echo "DPO LinguaSafe"
echo "Job:  ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "Root: ${ROOT_DIR}"
echo "======================================================"

PYTHON="$CONDA_PREFIX/envs/shortcut-repr/bin/python"
echo "Using shortcut-repr: ${PYTHON}"

"${PYTHON}" --version
"${PYTHON}" -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TMPDIR="/tmp"

# --- Parameters ---
BASE_MODEL="${BASE_MODEL:-/playpen/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
MODEL_NAME="$(basename "$(dirname "$(dirname "${BASE_MODEL}")")" | sed 's/.*--//')"
# SFT_RUN should be the run name from the SFT job (e.g. sft_Llama-3.1-8B-Instruct_12345)
SFT_RUN="${SFT_RUN:-}"
SFT_MODEL_DIR="${SFT_MODEL_DIR:-}"

# Resolve SFT model dir
if [[ -n "${SFT_MODEL_DIR}" ]]; then
    : # already set
elif [[ -n "${SFT_RUN}" ]]; then
    SFT_MODEL_DIR="output/SFT/${SFT_RUN}/model"
else
    # Auto-detect latest SFT run
    SFT_MODEL_DIR=$(ls -td output/SFT/*/model 2>/dev/null | head -1)
fi

[[ -n "${SFT_MODEL_DIR}" ]] || { echo "ERROR: set SFT_RUN or SFT_MODEL_DIR"; exit 1; }
[[ -d "${SFT_MODEL_DIR}"  ]] || { echo "ERROR: SFT model dir not found: ${SFT_MODEL_DIR}"; exit 1; }

RUN_NAME="${RUN_NAME:-dpo_${MODEL_NAME}_${SLURM_JOB_ID:-local}}"
BETA="${BETA:-0.1}"
EPOCHS="${EPOCHS:-1}"
BATCH="${BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-7}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
WANDB_PROJECT="${WANDB_PROJECT:-linguasafe-sft-dpo}"

SPLIT_DIR="data/sft_dpo"
MODEL_OUT="output/DPO/${RUN_NAME}/model"
EVAL_OUT="output/DPO/${RUN_NAME}/eval"

echo ""
echo "BASE_MODEL:    ${BASE_MODEL}"
echo "SFT_MODEL_DIR: ${SFT_MODEL_DIR}"
echo "RUN_NAME:      ${RUN_NAME}"
echo "MODEL_OUT:     ${MODEL_OUT}"
echo "EVAL_OUT:      ${EVAL_OUT}"
echo ""

[[ -d "${BASE_MODEL}"   ]] || { echo "ERROR: base model not found: ${BASE_MODEL}"; exit 1; }
[[ -f "${SPLIT_DIR}/dpo_train.jsonl" ]] || { echo "ERROR: DPO data missing. Run SFT job first."; exit 1; }

mkdir -p logs "${MODEL_OUT}" "${EVAL_OUT}"

# =============================================
# Step 1: DPO training
# =============================================
echo "=== Step 1: DPO training ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

"${PYTHON}" scripts/train_dpo_linguasafe.py \
    --base-model    "${SFT_MODEL_DIR}" \
    --train-jsonl   "${SPLIT_DIR}/dpo_train.jsonl" \
    --val-jsonl     "${SPLIT_DIR}/dpo_val.jsonl" \
    --output-dir    "${MODEL_OUT}" \
    --beta          "${BETA}" \
    --epochs        "${EPOCHS}" \
    --batch-size    "${BATCH}" \
    --grad-accum    "${GRAD_ACCUM}" \
    --lr            "${LR}" \
    --lora-r        "${LORA_R}" \
    --lora-alpha    "${LORA_ALPHA}" \
    --wandb-project "${WANDB_PROJECT}" \
    --run-name      "${RUN_NAME}"

echo "Model saved to ${MODEL_OUT}"

# =============================================
# Step 2: Held-out evaluation
# =============================================
echo ""
echo "=== Step 2: Held-out eval ==="
if [[ -f "${SPLIT_DIR}/heldout_test.jsonl" ]]; then
    "${PYTHON}" scripts/eval_sft_dpo_heldout.py \
        --base-model     "${BASE_MODEL}" \
        --adapter-dir    "${MODEL_OUT}" \
        --heldout        "${SPLIT_DIR}/heldout_test.jsonl" \
        --output         "${EVAL_OUT}/heldout_responses.jsonl" \
        --model-tag      "${RUN_NAME}" \
        --max-new-tokens "${MAX_NEW_TOKENS}"

    echo "Eval responses: $(wc -l < "${EVAL_OUT}/heldout_responses.jsonl") rows"
    echo "Sample:"
    head -1 "${EVAL_OUT}/heldout_responses.jsonl" | "${PYTHON}" -m json.tool | head -10
else
    echo "No heldout_test.jsonl found, skipping eval."
fi

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Outputs:"
echo "  model → ${MODEL_OUT}"
echo "  eval  → ${EVAL_OUT}"
echo "======================================================"
