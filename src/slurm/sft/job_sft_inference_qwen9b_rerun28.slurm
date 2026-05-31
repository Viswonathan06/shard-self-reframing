#!/usr/bin/bash
# Re-run inference on the 28 Qwen3.5-9B rows that hit max_tokens mid-think.
# Uses max-new-tokens=8192 so thinking can complete.
# Input:  output/SFT/qwen35_122b_teacher/qwen35_9b/test_no_think_subset.jsonl
# Output: output/SFT/qwen35_122b_teacher/qwen35_9b/inference_rerun28.jsonl

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=sft_inf_qwen9b_28
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true

export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"
export PYTHONUNBUFFERED=1

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"

echo "======================================================"
echo "Inference (rerun 28): Qwen3.5-9B SFT — max-new-tokens=8192"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"

BASE_MODEL=""
for snap in $HF_HOME/models--Qwen--Qwen3.5-9B/snapshots/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && BASE_MODEL="${snap%/}" && break
done
[[ -z "$BASE_MODEL" ]] && { echo "ERROR: Qwen3.5-9B base model not found"; exit 1; }
echo "Base model: ${BASE_MODEL}"

SFT_ADAPTER="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/qwen35_9b/model"
INPUT_DATA="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/qwen35_9b/test_no_think_subset.jsonl"
OUTPUT="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/qwen35_9b/inference_rerun28.jsonl"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

"${PYTHON}" scripts/inference_sft_qwen_vllm.py \
    --base-model              "${BASE_MODEL}" \
    --sft-adapter             "${SFT_ADAPTER}" \
    --input-data              "${INPUT_DATA}" \
    --output                  "${OUTPUT}" \
    --max-new-tokens          8192 \
    --temperature             0.0 \
    --tensor-parallel-size    4 \
    --gpu-memory-utilization  0.85 \
    --max-model-len           16384 \
    --lora-rank               16

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Output: ${OUTPUT}"
wc -l "${OUTPUT}" || true
echo "======================================================"
