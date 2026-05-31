#!/bin/bash
# Score SFT heldout responses with Gemma4-31B as judge.
# Run after job_sft_linguasafe.slurm (or job_sft_qwen25_7b.slurm) completes.
#
# Usage:
#   SFT_RUN=sft_Llama-3.1-8B-Instruct_13197 sbatch src/slurm/sft/job_judge_sft_gemma4.slurm
#
# Overrides:
#   RESPONSES=/path/to/heldout_responses.jsonl  sbatch ...
#   MAX_RECORDS=50 sbatch ...   # quick smoke-test

#SBATCH --job-name=judge_sft_gemma4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --chdir=$PROJECT_ROOT

set -euo pipefail

ROOT_DIR="$PROJECT_ROOT"

echo "======================================================"
echo "Gemma4 Judge — SFT Heldout"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
echo "Using: ${PYTHON}"
"${PYTHON}" --version

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export TMPDIR="/tmp"
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=4096

# --- Resolve SFT run ---
SFT_RUN="${SFT_RUN:-}"
RESPONSES="${RESPONSES:-}"

if [[ -z "${RESPONSES}" ]]; then
    if [[ -n "${SFT_RUN}" ]]; then
        RESPONSES="output/SFT/${SFT_RUN}/eval/heldout_responses.jsonl"
    else
        RESPONSES=$(ls -t output/SFT/*/eval/heldout_responses.jsonl 2>/dev/null | head -1)
    fi
fi
[[ -n "${RESPONSES}" ]] || { echo "ERROR: no heldout_responses.jsonl found. Set SFT_RUN or RESPONSES."; exit 1; }
[[ -f "${RESPONSES}" ]] || { echo "ERROR: responses file not found: ${RESPONSES}"; exit 1; }

OUTPUT="${RESPONSES/heldout_responses.jsonl/heldout_judged_gemma4.jsonl}"
MAX_RECORDS="${MAX_RECORDS:-}"

# --- Gemma4: use HF model ID + HF_HOME so vLLM resolves the cache ---
if [[ -d $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
else
    export HF_HOME="${HOME}/.cache/huggingface"
fi
JUDGE_MODEL="google/gemma-4-31B-it"
echo ""
echo "RESPONSES:   ${RESPONSES}"
echo "OUTPUT:      ${OUTPUT}"
echo "JUDGE_MODEL: ${JUDGE_MODEL}"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

CMD_ARGS=(
    "--responses"   "${RESPONSES}"
    "--output"      "${OUTPUT}"
    "--judge-model" "${JUDGE_MODEL}"
    "--batch-size"  "8"
)
[[ -n "${MAX_RECORDS}" ]] && CMD_ARGS+=("--max-records" "${MAX_RECORDS}")

"${PYTHON}" scripts/judge_sft_heldout_gemma4.py "${CMD_ARGS[@]}"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Judged responses → ${OUTPUT}"
echo "======================================================"
