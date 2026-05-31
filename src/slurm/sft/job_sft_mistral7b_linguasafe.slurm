#!/usr/bin/bash
# SFT: Mistral-7B-Instruct-v0.3 on qwen35_122b teacher data (non-reasoning split).
# Input:  output/SFT/qwen35_122b_teacher/sft_nonreasoning/{train,val}.jsonl
# Output: output/SFT/qwen35_122b_teacher/mistral7b/model/

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=sft_mistral7b
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
echo "SFT: Mistral-7B-Instruct-v0.3 (non-reasoning teacher)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version

if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
else
    export HF_HOME="${HOME}/.cache/huggingface"
fi
HF_HUB="${HF_HOME}/hub"

MISTRAL_SNAPSHOT="0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a"
MODEL_PATH=""
for search_root in \
    "${HF_HUB}" \
    "$HF_HOME" \
    "${HOME}/.cache/huggingface/hub"; do
    for snap in \
        "${search_root}/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/${MISTRAL_SNAPSHOT}" \
        "${search_root}/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/"/*/; do
        snap="${snap%/}"
        [[ -d "$snap" ]] && [[ -f "${snap}/config.json" ]] && MODEL_PATH="${snap}" && break 2
    done
done
if [[ -z "$MODEL_PATH" ]]; then
    echo "Model not found locally — downloading mistralai/Mistral-7B-Instruct-v0.3 to ${HF_HUB}..."
    MODEL_PATH=$("${PYTHON}" -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='mistralai/Mistral-7B-Instruct-v0.3', cache_dir='${HF_HUB}', resume_download=True)
print(p)
") || { echo "ERROR: Download failed"; exit 1; }
fi
echo "Model: ${MODEL_PATH}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

DATA_DIR="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/sft_nonreasoning"
OUT_DIR="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/mistral7b/model"

export WANDB_PROJECT="linguasafe-sft"
[[ -f "${ROOT_DIR}/.wandb_key" ]] && export WANDB_API_KEY=$(cat "${ROOT_DIR}/.wandb_key")

MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
"${PYTHON}" -m torch.distributed.run --nproc_per_node=4 --master_port="${MASTER_PORT}" scripts/train_sft_linguasafe.py \
    --base-model    "${MODEL_PATH}" \
    --train-jsonl   "${DATA_DIR}/train.jsonl" \
    --val-jsonl     "${DATA_DIR}/val.jsonl" \
    --output-dir    "${OUT_DIR}" \
    --max-length    4096 \
    --epochs        3 \
    --batch-size    2 \
    --grad-accum    8 \
    --lr            2e-5 \
    --warmup-ratio  0.05 \
    --lora-r        16 \
    --lora-alpha    32 \
    --lora-dropout  0.05 \
    --logging-steps 10 \
    --save-steps    100 \
    --run-name      "sft_mistral7b_122b_teacher" \
    --wandb-project "linguasafe-sft"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Model saved to: ${OUT_DIR}"
echo "======================================================"
