#!/bin/bash
# SFT on Llama-3.1-8B-Instruct using the LinguaSafe clean-preferred dataset.
# Runs data prep → SFT training → held-out eval in one job.
#
# Usage:
#   sbatch src/slurm/sft/job_sft_linguasafe.slurm
#
# Overrides:
#   BASE_MODEL=/path/to/snapshot sbatch ...
#   RAW_DATA=/path/to/sft_clean_preferred.jsonl sbatch ...
#   RUN_NAME=sft_v2 sbatch ...
#   EPOCHS=2 sbatch ...

#SBATCH --job-name=sft_linguasafe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=$PROJECT_ROOT

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || exit 1

echo "======================================================"
echo "SFT LinguaSafe"
echo "Job:  ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "Root: ${ROOT_DIR}"
echo "======================================================"

PYTHON="$CONDA_PREFIX/envs/shortcut-repr/bin/python"
echo "Using shortcut-repr: ${PYTHON}"

"${PYTHON}" --version
"${PYTHON}" -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"

export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_OFFLINE=0
export HF_DATASETS_OFFLINE=0

# W&B auth
WANDB_KEY_FILE="${HOME}/llm_safety/Multi-Lingual-Alignment-in-LLMs/.wandb_key"
if [[ -f "${WANDB_KEY_FILE}" ]]; then
    export WANDB_API_KEY="$(cat "${WANDB_KEY_FILE}")"
    echo "W&B: API key loaded from ${WANDB_KEY_FILE}"
else
    echo "W&B: key file not found at ${WANDB_KEY_FILE}, logging disabled"
    export WANDB_DISABLED=true
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TMPDIR="/tmp"

# --- Parameters (all overridable) ---
HF_MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME="Llama-3.1-8B-Instruct"

# Resolve cached model path; download if missing
resolve_model_path() {
    local cache_base="${HOME}/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct"
    if [[ -d "${cache_base}/snapshots" ]]; then
        snap=$(find "${cache_base}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
        if [[ -n "${snap}" && -f "${snap}/config.json" ]]; then
            echo "${snap}"; return 0
        fi
    fi
    echo ""; return 1
}

BASE_MODEL="${BASE_MODEL:-$(resolve_model_path || true)}"

if [[ -z "${BASE_MODEL}" ]]; then
    echo "Llama-3.1-8B-Instruct not cached — downloading from HuggingFace Hub..."
    "${PYTHON}" -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('${HF_MODEL_ID}')
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('${HF_MODEL_ID}')
print('Download complete.')
"
    BASE_MODEL=$(resolve_model_path || true)
    [[ -n "${BASE_MODEL}" ]] || BASE_MODEL="${HF_MODEL_ID}"
fi
RUN_NAME="${RUN_NAME:-sft_${MODEL_NAME}_${SLURM_JOB_ID:-local}}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-5}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
WANDB_PROJECT="${WANDB_PROJECT:-linguasafe-sft-dpo}"

RAW_DATA="${RAW_DATA:-data/sft_clean_preferred_expanded.jsonl}"
SPLIT_DIR="${SPLIT_DIR:-data/sft_dpo_expanded}"
BASELINE_RESPONSES="${BASELINE_RESPONSES:-output/Linguasafe_Experiments/helpful_assistant_multimodel/llama8b/p1_baseline_outputs.jsonl}"
MODEL_OUT="output/SFT/${RUN_NAME}/model"
EVAL_OUT="output/SFT/${RUN_NAME}/eval"

echo ""
echo "BASE_MODEL:  ${BASE_MODEL}"
echo "RAW_DATA:    ${RAW_DATA}"
echo "RUN_NAME:    ${RUN_NAME}"
echo "MODEL_OUT:   ${MODEL_OUT}"
echo "EVAL_OUT:    ${EVAL_OUT}"
echo ""

[[ -f "${RAW_DATA}" ]] || { echo "ERROR: raw data not found: ${RAW_DATA}"; exit 1; }

mkdir -p logs "${MODEL_OUT}" "${EVAL_OUT}"

# =============================================
# Step 1: Data preparation
# =============================================
echo ""
echo "=== Step 1: Build expanded preferred dataset + prepare splits ==="

HELDOUT_CANDIDATES="${RAW_DATA%.jsonl}_heldout_candidates.jsonl"
"${PYTHON}" scripts/build_sft_preferred.py \
    --output "${RAW_DATA}"

"${PYTHON}" scripts/prepare_sft_dpo_data.py \
    --input               "${RAW_DATA}" \
    --output-dir          "${SPLIT_DIR}" \
    --heldout-candidates  "${HELDOUT_CANDIDATES}" \
    --val-frac            0.10 \
    --seed                42

echo "Data split sizes:"
for f in "${SPLIT_DIR}"/*.jsonl; do
    echo "  $(wc -l < "$f") rows — $f"
done

# =============================================
# Step 2: SFT training
# =============================================
echo ""
echo "=== Step 2: SFT training ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

"${PYTHON}" scripts/train_sft_linguasafe.py \
    --base-model    "${BASE_MODEL}" \
    --train-jsonl   "${SPLIT_DIR}/sft_train.jsonl" \
    --val-jsonl     "${SPLIT_DIR}/sft_val.jsonl" \
    --output-dir    "${MODEL_OUT}" \
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
# Step 3: Held-out evaluation (both sets)
# =============================================
echo ""
echo "=== Step 3: Held-out eval ==="

run_eval() {
    local tag="$1"
    local heldout_file="${SPLIT_DIR}/${tag}.jsonl"
    local responses_out="${EVAL_OUT}/${tag}_responses.jsonl"

    if [[ ! -f "${heldout_file}" ]]; then
        echo "  Skipping ${tag}: ${heldout_file} not found"
        return
    fi

    echo ""
    echo "--- Generating responses: ${tag} ($(wc -l < "${heldout_file}") prompts) ---"
    "${PYTHON}" scripts/eval_sft_dpo_heldout.py \
        --base-model     "${BASE_MODEL}" \
        --adapter-dir    "${MODEL_OUT}" \
        --heldout        "${heldout_file}" \
        --output         "${responses_out}" \
        --model-tag      "${RUN_NAME}_${tag}" \
        --max-new-tokens "${MAX_NEW_TOKENS}"
    echo "  Responses written: $(wc -l < "${responses_out}") rows → ${responses_out}"
}

run_eval "heldout_indist"
run_eval "heldout_oodist"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Outputs:"
echo "  model → ${MODEL_OUT}"
echo "  eval  → ${EVAL_OUT}"
echo ""
echo "Next: submit Gemma4 judge jobs for each heldout set:"
echo ""
echo "  # In-distribution heldout:"
echo "  RESPONSES_A=${EVAL_OUT}/heldout_indist_responses.jsonl \\"
echo "  RESPONSES_B=${BASELINE_RESPONSES} \\"
echo "  MODEL_TAG=llama8b_sft_vs_baseline_indist \\"
echo "  OUTPUT_DIR=${EVAL_OUT}/judge_indist \\"
echo "  sbatch src/slurm/sft/job_judge_sft_vs_baseline_gemma4.slurm"
echo ""
echo "  # Out-of-distribution heldout:"
echo "  RESPONSES_A=${EVAL_OUT}/heldout_oodist_responses.jsonl \\"
echo "  RESPONSES_B=${BASELINE_RESPONSES} \\"
echo "  MODEL_TAG=llama8b_sft_vs_baseline_oodist \\"
echo "  OUTPUT_DIR=${EVAL_OUT}/judge_oodist \\"
echo "  sbatch src/slurm/sft/job_judge_sft_vs_baseline_gemma4.slurm"
echo "======================================================"
