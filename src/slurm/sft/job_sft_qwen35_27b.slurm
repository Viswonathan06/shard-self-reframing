#!/usr/bin/bash
# SFT: Qwen3.5-27B on qwen35_122b teacher data (reasoning split).
# Input:  output/SFT/qwen35_122b_teacher/sft_reasoning/{train,val}.jsonl
# Output: output/SFT/qwen35_122b_teacher/qwen35_27b/model/
# Uses FSDP full_shard (bf16) — 54 GB / 4 GPUs = 13.5 GB/GPU, fits 4x48GB.

#SBATCH -n 1
#SBATCH --nodelist=gpu1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --job-name=sft_qwen27b
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
echo "SFT: Qwen3.5-27B (reasoning teacher)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version

if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
else
    export HF_HOME="${HOME}/.cache/huggingface"
fi
HF_HUB="${HF_HOME}/hub"

MODEL_PATH=""
for search_root in \
    "${HF_HUB}" \
    "$HF_HOME" \
    "${HOME}/.cache/huggingface/hub"; do
    for snap in "${search_root}/models--Qwen--Qwen3.5-27B/snapshots/"/*/; do
        snap="${snap%/}"
        [[ -d "$snap" ]] && \
        ([[ -f "${snap}/model.safetensors.index.json" ]] || [[ -f "${snap}/model.safetensors" ]]) && \
        MODEL_PATH="${snap}" && break 2
    done
done
if [[ -z "$MODEL_PATH" ]]; then
    echo "Model not found locally — downloading Qwen/Qwen3.5-27B to ${HF_HUB}..."
    MODEL_PATH=$("${PYTHON}" -c "
from huggingface_hub import snapshot_download
import os
p = snapshot_download(repo_id='Qwen/Qwen3.5-27B', cache_dir='${HF_HUB}', resume_download=True)
print(p)
") || { echo "ERROR: Download failed"; exit 1; }
fi
echo "Model: ${MODEL_PATH}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

DATA_DIR="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/sft_reasoning"
OUT_DIR="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/qwen35_27b/model"

export WANDB_PROJECT="linguasafe-sft"
[[ -f "${ROOT_DIR}/.wandb_key" ]] && export WANDB_API_KEY=$(cat "${ROOT_DIR}/.wandb_key")

# Effective batch = 4 GPUs x bs=1 x grad_accum=4 = 16 (matches 9B's 4x1x4=16)
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
"${PYTHON}" -m torch.distributed.run --nproc_per_node=4 --master_port="${MASTER_PORT}" scripts/train_sft_linguasafe.py \
    --base-model             "${MODEL_PATH}" \
    --train-jsonl            "${DATA_DIR}/train.jsonl" \
    --val-jsonl              "${DATA_DIR}/val.jsonl" \
    --output-dir             "${OUT_DIR}" \
    --max-length             4096 \
    --epochs                 3 \
    --batch-size             1 \
    --grad-accum             4 \
    --lr                     2e-5 \
    --warmup-ratio           0.05 \
    --lora-r                 16 \
    --lora-alpha             32 \
    --lora-dropout           0.05 \
    --logging-steps          10 \
    --save-steps             100 \
    --fsdp                   "full_shard auto_wrap" \
    --fsdp-transformer-cls   "Qwen3_5DecoderLayer" \
    --fsdp-cpu-efficient \
    --run-name               "sft_qwen35_27b_122b_teacher" \
    --wandb-project          "linguasafe-sft"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Model saved to: ${OUT_DIR}"
echo "======================================================"
