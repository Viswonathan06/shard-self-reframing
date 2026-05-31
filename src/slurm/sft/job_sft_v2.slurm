#!/bin/bash
#SBATCH --job-name=sft_v2_llama31_8b
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err
#SBATCH --partition=a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00

set -euo pipefail

echo "=========================================="
echo "SFT v2 — chat-template + prompt masking + r=16 LoRA"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "=========================================="

ROOT_DIR="${ROOT_DIR:-/common/guam/Safeshift}"
cd "$ROOT_DIR"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate llm_safety

python --version
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| tfm', transformers.__version__, '| peft', peft.__version__, '| cuda', torch.cuda.is_available(), '| ngpu', torch.cuda.device_count())"

# Avoid network calls — Llama snapshot is already on local disk.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

BASE_MODEL="${BASE_MODEL:-/playpen/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
TRAIN_JSONL="${TRAIN_JSONL:-data/sft_scratchpad_v2/train_sft.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/sft_scratchpad_v2/val_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-models/llama31_sft_v2}"
EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGETS="${LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj}"

if [[ ! -d "$BASE_MODEL" ]]; then
  echo "ERROR: base model dir missing: $BASE_MODEL"
  exit 1
fi
[[ -f "$TRAIN_JSONL" ]] || { echo "ERROR: missing $TRAIN_JSONL"; exit 1; }
[[ -f "$VAL_JSONL"   ]] || { echo "ERROR: missing $VAL_JSONL";   exit 1; }

echo "BASE_MODEL=$BASE_MODEL"
echo "TRAIN=$TRAIN_JSONL  ($(wc -l < "$TRAIN_JSONL") rows)"
echo "VAL=$VAL_JSONL      ($(wc -l < "$VAL_JSONL") rows)"
echo "OUTPUT=$OUTPUT_DIR"

python scripts/sft_v2.py \
  --base-model "$BASE_MODEL" \
  --train-jsonl "$TRAIN_JSONL" \
  --val-jsonl   "$VAL_JSONL" \
  --output-dir  "$OUTPUT_DIR" \
  --epochs      "$EPOCHS" \
  --per-device-batch-size "$BATCH" \
  --grad-accum  "$GRAD_ACCUM" \
  --learning-rate "$LR" \
  --lora-r      "$LORA_R" \
  --lora-alpha  "$LORA_ALPHA" \
  --lora-targets "$LORA_TARGETS"

echo "=========================================="
echo "End: $(date)"
echo "Files in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR" | head -30
echo "=========================================="
