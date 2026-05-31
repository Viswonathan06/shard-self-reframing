#!/usr/bin/bash
# Generic SFT launcher.
# Override any variable via sbatch --export or by editing the defaults below.
#
# Minimal example:
#   sbatch --export=ALL,HF_MODEL_ID="Qwen/Qwen3.5-27B",\
#     DATA_DIR="output/SFT/qwen35_122b_teacher/sft_reasoning",\
#     OUT_DIR="output/SFT/qwen35_122b_teacher/qwen35_27b/model",\
#     FSDP_CLS="Qwen3_5DecoderLayer",USE_FSDP=1,FSDP_CPU_EFFICIENT=1 \
#     src/slurm/sft/job_sft.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --job-name=sft_generic
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

# ─── Model ────────────────────────────────────────────────────────────────────
# Set HF_MODEL_ID to a HuggingFace repo (e.g. "Qwen/Qwen3.5-27B") or
# LOCAL_MODEL_PATH to an absolute path to skip the hub search entirely.
HF_MODEL_ID="${HF_MODEL_ID:-Qwen/Qwen3.5-27B}"
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-}"

# HF cache
if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
else
    export HF_HOME="${HOME}/.cache/huggingface"
fi
HF_HUB="${HF_HOME}/hub"

# Derive the models-- directory name from the HF repo ID (org--name format)
HF_DIR_NAME="models--$(echo "${HF_MODEL_ID}" | tr '/' '--')"

# ─── Data / Output ────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/output/SFT/qwen35_122b_teacher/sft_reasoning}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/output/SFT/qwen35_122b_teacher/qwen35_27b/model}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_DIR}/val.jsonl}"

# ─── Training hyperparams ─────────────────────────────────────────────────────
MAX_LENGTH="${MAX_LENGTH:-4096}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SEED="${SEED:-42}"

# ─── FSDP options ─────────────────────────────────────────────────────────────
# USE_FSDP=1         → full_shard auto_wrap
# FSDP_CLS           → decoder layer class for auto_wrap (e.g. Qwen3_5DecoderLayer)
# FSDP_CPU_EFFICIENT=1 → rank-0 loads on CPU, others use meta device
# USE_QLORA=1        → 4-bit NF4 QLoRA (no FSDP; use for very large models on single node)
USE_FSDP="${USE_FSDP:-1}"
FSDP_CLS="${FSDP_CLS:-Qwen3_5DecoderLayer}"
FSDP_CPU_EFFICIENT="${FSDP_CPU_EFFICIENT:-1}"
USE_QLORA="${USE_QLORA:-0}"

# ─── W&B ──────────────────────────────────────────────────────────────────────
WANDB_PROJECT="${WANDB_PROJECT:-linguasafe-sft}"
_out_base=$(basename "${OUT_DIR%/model}")
RUN_NAME="${RUN_NAME:-sft_${_out_base}}"
export WANDB_PROJECT
[[ -f "${ROOT_DIR}/.wandb_key" ]] && export WANDB_API_KEY=$(cat "${ROOT_DIR}/.wandb_key")

# ─── Resolve model path ───────────────────────────────────────────────────────
if [[ -n "${LOCAL_MODEL_PATH}" ]]; then
    MODEL_PATH="${LOCAL_MODEL_PATH}"
else
    MODEL_PATH=""
    for search_root in "${HF_HUB}" "${HOME}/.cache/huggingface/hub"; do
        for snap in "${search_root}/${HF_DIR_NAME}/snapshots/"/*/; do
            snap="${snap%/}"
            [[ -d "$snap" ]] && \
            ([[ -f "${snap}/model.safetensors.index.json" ]] || [[ -f "${snap}/model.safetensors" ]] || [[ -f "${snap}/config.json" ]]) && \
            MODEL_PATH="${snap}" && break 2
        done
    done
    if [[ -z "$MODEL_PATH" ]]; then
        echo "Model not found locally — downloading ${HF_MODEL_ID} to ${HF_HUB}..."
        MODEL_PATH=$("${PYTHON}" -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='${HF_MODEL_ID}', cache_dir='${HF_HUB}', resume_download=True)
print(p)
") || { echo "ERROR: Download failed"; exit 1; }
    fi
fi

echo "======================================================"
echo "SFT: ${HF_MODEL_ID}"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version
echo "Model:  ${MODEL_PATH}"
echo "Train:  ${TRAIN_JSONL}"
echo "Val:    ${VAL_JSONL}"
echo "Output: ${OUT_DIR}"
echo "Epochs=${EPOCHS}  BS=${BATCH_SIZE}×${GRAD_ACCUM}  LR=${LR}  MaxLen=${MAX_LENGTH}"
[[ "${USE_FSDP}" == "1" ]] && echo "FSDP: full_shard  cls=${FSDP_CLS}  cpu_efficient=${FSDP_CPU_EFFICIENT}"
[[ "${USE_QLORA}" == "1" ]] && echo "Mode: QLoRA 4-bit"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ─── Build training command ───────────────────────────────────────────────────
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
CMD=(
    "${PYTHON}" -m torch.distributed.run
    --nproc_per_node=4
    --master_port="${MASTER_PORT}"
    scripts/train_sft_linguasafe.py
    --base-model       "${MODEL_PATH}"
    --train-jsonl      "${TRAIN_JSONL}"
    --val-jsonl        "${VAL_JSONL}"
    --output-dir       "${OUT_DIR}"
    --max-length       "${MAX_LENGTH}"
    --epochs           "${EPOCHS}"
    --batch-size       "${BATCH_SIZE}"
    --grad-accum       "${GRAD_ACCUM}"
    --lr               "${LR}"
    --warmup-ratio     "${WARMUP_RATIO}"
    --lora-r           "${LORA_R}"
    --lora-alpha       "${LORA_ALPHA}"
    --lora-dropout     "${LORA_DROPOUT}"
    --lora-targets     "${LORA_TARGETS}"
    --logging-steps    "${LOGGING_STEPS}"
    --seed             "${SEED}"
    --run-name         "${RUN_NAME}"
    --wandb-project    "${WANDB_PROJECT}"
)

if [[ "${USE_QLORA}" == "1" ]]; then
    CMD+=(--load-in-4bit)
elif [[ "${USE_FSDP}" == "1" ]]; then
    CMD+=(--fsdp "full_shard auto_wrap")
    [[ -n "${FSDP_CLS}" ]] && CMD+=(--fsdp-transformer-cls "${FSDP_CLS}")
    [[ "${FSDP_CPU_EFFICIENT}" == "1" ]] && CMD+=(--fsdp-cpu-efficient)
fi

echo "Running: ${CMD[*]}"
echo ""
"${CMD[@]}"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Model saved to: ${OUT_DIR}"
echo "======================================================"
