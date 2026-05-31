#!/bin/bash
# Generic Gemma4 judge: compare any two response files (SFT vs baseline, or any A vs B).
#
# Required:
#   RESPONSES_A  — path to JSONL with fields: p1_id, p1, response  (e.g. SFT heldout)
#   RESPONSES_B  — path to JSONL with fields: p1_id, p1, output    (e.g. baseline)
#   MODEL_TAG    — label for output files (e.g. llama8b_sft_vs_baseline)
#   OUTPUT_DIR   — where to write results
#
# Usage:
#   RESPONSES_A=output/SFT/sft_Llama.../eval/heldout_responses.jsonl \
#   RESPONSES_B=output/Linguasafe_Experiments/helpful_assistant_multimodel/llama8b/p1_baseline_outputs.jsonl \
#   MODEL_TAG=llama8b_sft_vs_baseline \
#   OUTPUT_DIR=output/SFT/sft_Llama.../eval/judge_vs_baseline \
#   sbatch src/slurm/sft/job_judge_sft_vs_baseline_gemma4.slurm
#
# Optional:
#   MAX_RECORDS=50    — limit records (smoke-test)
#   BATCH_SIZE=8      — vLLM batch size

#SBATCH --job-name=judge_gemma4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --chdir=$PROJECT_ROOT

set -euo pipefail

ROOT_DIR="$PROJECT_ROOT"

echo "======================================================"
echo "Gemma4 Judge — Generic A vs B"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"

# --- Validate required env vars ---
: "${RESPONSES_A:?Set RESPONSES_A to the first response file (e.g. SFT heldout JSONL)}"
: "${RESPONSES_B:?Set RESPONSES_B to the second response file (e.g. baseline JSONL)}"
: "${MODEL_TAG:?Set MODEL_TAG to a label for output files (e.g. llama8b_sft_vs_baseline)}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to where results should be written}"

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
echo "Using: ${PYTHON}"
"${PYTHON}" --version

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TMPDIR="/tmp"
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=4096

MAX_RECORDS="${MAX_RECORDS:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"

# --- Gemma4: resolve local snapshot path directly to avoid HF cache writes ---
GEMMA_SNAPSHOT="ba74f5b6c647c0911554e50278d6f6f4477f9010"
GEMMA_LOCAL="$HF_HOME/models--google--gemma-4-31B-it/snapshots/${GEMMA_SNAPSHOT}"
if [[ -d "${GEMMA_LOCAL}" ]]; then
    JUDGE_MODEL="${GEMMA_LOCAL}"
    export HF_HOME=/playpen/huggingface
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
else
    JUDGE_MODEL="google/gemma-4-31B-it"
    if [[ -d $HF_HOME ]]; then
        export HF_HOME=/playpen/huggingface
    else
        export HF_HOME="${HOME}/.cache/huggingface"
    fi
fi

echo ""
echo "RESPONSES_A: ${RESPONSES_A}"
echo "RESPONSES_B: ${RESPONSES_B}"
echo "OUTPUT_DIR:  ${OUTPUT_DIR}"
echo "MODEL_TAG:   ${MODEL_TAG}"
echo "JUDGE:       ${JUDGE_MODEL}"
echo ""

[[ -f "${RESPONSES_A}" ]] || { echo "ERROR: RESPONSES_A not found: ${RESPONSES_A}"; exit 1; }
[[ -f "${RESPONSES_B}" ]] || { echo "ERROR: RESPONSES_B not found: ${RESPONSES_B}"; exit 1; }

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

CMD_ARGS=(
    "--sft-responses" "${RESPONSES_A}"
    "--baseline"      "${RESPONSES_B}"
    "--output-dir"    "${OUTPUT_DIR}"
    "--judge-model"   "${JUDGE_MODEL}"
    "--model-tag"     "${MODEL_TAG}"
    "--batch-size"    "${BATCH_SIZE}"
)
[[ -n "${MAX_RECORDS}" ]] && CMD_ARGS+=("--max-records" "${MAX_RECORDS}")

"${PYTHON}" scripts/judge_sft_vs_baseline_gemma4.py "${CMD_ARGS[@]}"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Results → ${OUTPUT_DIR}"
echo "======================================================"
