#!/bin/bash
# SFT Mistral-7B-Instruct-v0.3 on expanded preferred dataset.
# Runs in-distribution held-out inference only (no oodist).
# Submits comparative judge (vs baseline) after inference completes.
#
# Usage:
#   sbatch src/slurm/sft/job_sft_mistral7b.slurm
#
# Optional overrides:
#   RUN_NAME=sft_Mistral-7B_v2  sbatch ...

#SBATCH --job-name=sft_mistral7b
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

ROOT_DIR="$PROJECT_ROOT"
PYTHON="$CONDA_PREFIX/envs/shortcut-repr/bin/python"

RUN_NAME="${RUN_NAME:-sft_Mistral-7B-Instruct-v0.3_v1}"
MODEL_OUT="${MODEL_OUT:-output/SFT/${RUN_NAME}/model}"
EVAL_OUT="${EVAL_OUT:-output/SFT/${RUN_NAME}/eval}"

echo "======================================================"
echo "SFT Mistral-7B-Instruct-v0.3"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"

echo "Using: ${PYTHON}"
"${PYTHON}" --version

# Resolve HF hub cache
if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
else
    PLAYPEN_USER_HF="/playpen/${USER}/huggingface"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" 2>/dev/null && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
        export HF_HOME="${PLAYPEN_USER_HF}"
    else
        export HF_HOME="${HOME}/.cache/huggingface"
    fi
fi
HF_HUB="${HF_HOME}/hub"
mkdir -p "${HF_HUB}"

MISTRAL_REPO="mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_SNAPSHOT="0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
BASE_MODEL=""
for cand in \
  "${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/${MISTRAL_SNAPSHOT}" \
  "${HF_HUB}"/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/*/
do
  cand="${cand%/}"
  [[ -d "$cand" ]] && [[ -f "$cand/config.json" ]] && BASE_MODEL="$cand" && break
done

if [[ -z "$BASE_MODEL" ]]; then
    echo "Model not found locally; downloading ${MISTRAL_REPO} → ${HF_HUB}"
    BASE_MODEL=$("${PYTHON}" -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id='${MISTRAL_REPO}', cache_dir='${HF_HUB}',
                         resume_download=True, token=os.getenv('HF_TOKEN'))
print(path)
")
fi
[[ -n "$BASE_MODEL" ]] || { echo "ERROR: could not resolve ${MISTRAL_REPO}"; exit 1; }

echo ""
echo "BASE_MODEL:  ${BASE_MODEL}"
echo "RUN_NAME:    ${RUN_NAME}"
echo "MODEL_OUT:   ${MODEL_OUT}"
echo "EVAL_OUT:    ${EVAL_OUT}"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TMPDIR="/tmp"
[[ -f "${ROOT_DIR}/.wandb_key" ]] && export WANDB_API_KEY="$(cat "${ROOT_DIR}/.wandb_key")"

mkdir -p "${MODEL_OUT}" "${EVAL_OUT}"

[[ -f "data/sft_dpo_expanded/sft_train.jsonl" ]]      || { echo "ERROR: missing sft_train.jsonl"; exit 1; }
[[ -f "data/sft_dpo_expanded/sft_val.jsonl" ]]        || { echo "ERROR: missing sft_val.jsonl"; exit 1; }
[[ -f "data/sft_dpo_expanded/heldout_indist.jsonl" ]] || { echo "ERROR: missing heldout_indist.jsonl"; exit 1; }

echo "=== Step 1: SFT training ==="
echo "train: data/sft_dpo_expanded/sft_train.jsonl  ($(wc -l < data/sft_dpo_expanded/sft_train.jsonl) rows)"
echo "val:   data/sft_dpo_expanded/sft_val.jsonl    ($(wc -l < data/sft_dpo_expanded/sft_val.jsonl) rows)"
echo ""

"${PYTHON}" scripts/train_sft_linguasafe.py \
  --base-model    "${BASE_MODEL}" \
  --train-jsonl   data/sft_dpo_expanded/sft_train.jsonl \
  --val-jsonl     data/sft_dpo_expanded/sft_val.jsonl \
  --output-dir    "${MODEL_OUT}" \
  --epochs        3 \
  --batch-size    2 \
  --grad-accum    8 \
  --lr            2e-5 \
  --run-name      "${RUN_NAME}" \
  --wandb-project linguasafe-sft-dpo

echo "Model saved to ${MODEL_OUT}"

echo ""
echo "=== Step 2: Held-out inference (in-distribution, $(wc -l < data/sft_dpo_expanded/heldout_indist.jsonl) prompts) ==="
"${PYTHON}" scripts/eval_sft_dpo_heldout.py \
  --base-model  "${BASE_MODEL}" \
  --adapter-dir "${MODEL_OUT}" \
  --heldout     data/sft_dpo_expanded/heldout_indist.jsonl \
  --output      "${EVAL_OUT}/heldout_indist_responses.jsonl" \
  --model-tag   "${RUN_NAME}" \
  --max-new-tokens 512

echo ""
echo "=== Step 3: Submit judge (SFT vs baseline) ==="
BASELINE="output/Linguasafe_Experiments/helpful_assistant_multimodel/mistral_7b/p1_baseline_outputs.jsonl"
JUDGE_JID=$(RESPONSES_A="${EVAL_OUT}/heldout_indist_responses.jsonl" \
            RESPONSES_B="${BASELINE}" \
            MODEL_TAG="mistral7b_sft_v1_vs_baseline_indist" \
            OUTPUT_DIR="${EVAL_OUT}/judge_vs_baseline_indist" \
            sbatch src/slurm/sft/job_judge_sft_vs_baseline_gemma4.slurm | awk '{print $4}')
echo "Judge job submitted: ${JUDGE_JID}"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "  model       → ${MODEL_OUT}"
echo "  indist eval → ${EVAL_OUT}/heldout_indist_responses.jsonl"
echo "  judge       → ${EVAL_OUT}/judge_vs_baseline_indist  (job ${JUDGE_JID})"
echo "======================================================"
