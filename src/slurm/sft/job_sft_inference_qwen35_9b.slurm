#!/usr/bin/bash
# Inference: Qwen3.5-9B SFT adapter on the SFT reasoning held-out test set (326 rows).
# Input:  output/SFT/qwen35_122b_teacher/sft_reasoning/test.jsonl
# Output: output/SFT/qwen35_122b_teacher/qwen35_9b/inference.jsonl

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=sft_inf_qwen9b
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
echo "Inference: Qwen3.5-9B SFT"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version

# Resolve base model path
BASE_MODEL=""
for snap in $HF_HOME/models--Qwen--Qwen3.5-9B/snapshots/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && BASE_MODEL="${snap%/}" && break
done
if [[ -z "$BASE_MODEL" ]]; then
    for snap in "${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/"/*/; do
        [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && BASE_MODEL="${snap%/}" && break
    done
fi
[[ -z "$BASE_MODEL" ]] && { echo "ERROR: Qwen3.5-9B base model not found"; exit 1; }
echo "Base model: ${BASE_MODEL}"

SFT_ADAPTER="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/qwen35_9b_ep1/model"
INPUT_DATA="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/sft_reasoning/test.jsonl"
OUTPUT="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/qwen35_9b_ep1/inference.jsonl"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

"${PYTHON}" scripts/inference_sft_qwen_vllm.py \
    --base-model              "${BASE_MODEL}" \
    --sft-adapter             "${SFT_ADAPTER}" \
    --input-data              "${INPUT_DATA}" \
    --output                  "${OUTPUT}" \
    --max-new-tokens          8192 \
    --temperature             0.3 \
    --tensor-parallel-size    4 \
    --gpu-memory-utilization  0.85 \
    --max-model-len           8192 \
    --lora-rank               16

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Output: ${OUTPUT}"
wc -l "${OUTPUT}" || true
echo "======================================================"
