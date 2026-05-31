#!/usr/bin/bash
# SFT: Llama-3.3-70B-Instruct on rational_sft data (from qwen35_27b teacher).
# Data already has messages format — no conversion needed.
#
#   sbatch src/slurm/sft/job_sft_llama70b_rational.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --nodelist=gpu5
#SBATCH --time=24:00:00
#SBATCH --job-name=sft_llama70b_rational
#SBATCH --output=/common/guam/Safeshift/logs/%x_%j.out
#SBATCH --error=/common/guam/Safeshift/logs/%x_%j.err

set -euo pipefail
ROOT="/common/guam/Safeshift"; cd "$ROOT"; mkdir -p logs
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=$HF_HOME
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TMPDIR="/tmp/${USER}_sft70b_rational_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID:-$$}"; mkdir -p "$TRITON_CACHE_DIR"

PYTHON="/common/guam/anaconda3/bin/python3"
NGPU="${SLURM_GPUS_ON_NODE:-4}"

echo "======================================================"
echo "SFT: Llama-3.3-70B-Instruct on rational_sft data"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  GPUs: ${NGPU}  Start: $(date)"
echo "======================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null || true

BASE_MODEL="$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
[[ -f "${BASE_MODEL}/config.json" ]] || { echo "ERROR: base model not found: ${BASE_MODEL}"; exit 1; }
echo "Model: ${BASE_MODEL}"

TRAIN_JSONL="${ROOT}/output/SFT/rational_sft/llama70b/train.jsonl"
VAL_JSONL="${ROOT}/output/SFT/rational_sft/llama70b/val.jsonl"
[[ -f "$TRAIN_JSONL" ]] || { echo "ERROR: train data not found: $TRAIN_JSONL"; exit 1; }
[[ -f "$VAL_JSONL"   ]] || { echo "ERROR: val data not found: $VAL_JSONL"; exit 1; }
echo "Train: $(wc -l < $TRAIN_JSONL) rows | Val: $(wc -l < $VAL_JSONL) rows"

OUT_DIR="${ROOT}/models/llama70b_sft_rational"
mkdir -p "$OUT_DIR"

echo ""
echo "Training | GPUs: ${NGPU} | out: ${OUT_DIR}"
# effective batch = NGPU * batch_size * grad_accum = 4 * 1 * 16 = 64

accelerate launch \
  --config_file src/slurm/dpo/fsdp_llama70b.yaml \
  --num_processes "${NGPU}" \
  scripts/train_sft_linguasafe.py \
    --base-model              "${BASE_MODEL}" \
    --train-jsonl             "${TRAIN_JSONL}" \
    --val-jsonl               "${VAL_JSONL}" \
    --output-dir              "${OUT_DIR}" \
    --max-length              1024 \
    --epochs                  3 \
    --batch-size              1 \
    --grad-accum              16 \
    --lr                      2e-5 \
    --warmup-ratio            0.05 \
    --lora-r                  16 \
    --lora-alpha              32 \
    --lora-dropout            0.05 \
    --logging-steps           10 \
    --save-steps              50 \
    --report-to               none \
    --run-name                "sft_llama70b_rational" \
    --fsdp                    "full_shard auto_wrap" \
    --fsdp-transformer-cls    "LlamaDecoderLayer"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Adapter: ${OUT_DIR}"
echo "======================================================"
