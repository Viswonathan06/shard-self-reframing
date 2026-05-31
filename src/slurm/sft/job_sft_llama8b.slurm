#!/usr/bin/bash
# SFT: Llama-3.1-8B-Instruct on qwen35_122b teacher data (non-reasoning split).
# Input:  output/SFT/qwen35_122b_teacher/sft_nonreasoning/{train,val}.jsonl
# Output: output/SFT/qwen35_122b_teacher/llama8b/model/

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=sft_llama8b
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
echo "SFT: Llama-3.1-8B-Instruct (non-reasoning teacher)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version

if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
else
    export HF_HOME="${HOME}/.cache/huggingface"
fi
HF_HUB="${HF_HOME}/hub"

# Resolve llama8b model path
MODEL_PATH=""
for snap in "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && MODEL_PATH="${snap%/}" && break
done
if [[ -z "$MODEL_PATH" ]]; then
    for snap in "$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"/*/; do
        [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && MODEL_PATH="${snap%/}" && break
    done
fi
[[ -z "$MODEL_PATH" ]] && { echo "ERROR: Llama-3.1-8B-Instruct not found in hub"; exit 1; }
echo "Model: ${MODEL_PATH}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

DATA_DIR="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/sft_nonreasoning"
OUT_DIR="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/llama8b/model"

export WANDB_PROJECT="linguasafe-sft"
[[ -f "${ROOT_DIR}/.wandb_key" ]] && export WANDB_API_KEY=$(cat "${ROOT_DIR}/.wandb_key")

"${PYTHON}" -m torch.distributed.run --nproc_per_node=4 scripts/train_sft_linguasafe.py \
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
    --run-name      "sft_llama8b_122b_teacher" \
    --wandb-project "linguasafe-sft"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Model saved to: ${OUT_DIR}"
echo "======================================================"
