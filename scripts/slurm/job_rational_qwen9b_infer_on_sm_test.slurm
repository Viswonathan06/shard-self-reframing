#!/usr/bin/bash
# Run inference for rational_sft/qwen35_9b adapter on the self_model (non-thinking)
# test set. Zero leakage confirmed: SM test prompts are all in RS test set,
# none in RS train/val.
# Output goes to rational_sft/qwen35_9b/inference/qwen35_9b_on_sm_test.jsonl
#
# Usage:
#   sbatch scripts/slurm/job_rational_qwen9b_infer_on_sm_test.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=4:00:00
#SBATCH --job-name=rational_qwen9b_infer_sm
#SBATCH --output=$PROJECT_ROOT/logs/rational_qwen9b_infer_sm_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/rational_qwen9b_infer_sm_%j.err

set -euo pipefail

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_TRUST_REMOTE_CODE=1

BASE_MODEL="$HF_HOME/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
SFT_ADAPTER="$ROOT/output/SFT/rational_sft/qwen35_9b/models/qwen35_9b"
TEST_JSONL="$ROOT/output/SFT/self_model/qwen35_9b/test.jsonl"
INFERENCE_OUT="$ROOT/output/SFT/rational_sft/qwen35_9b/inference/qwen35_9b_on_sm_test.jsonl"

[[ -f "$BASE_MODEL/config.json" ]]          || { echo "ERROR: base model not found: $BASE_MODEL"; exit 1; }
[[ -f "$SFT_ADAPTER/adapter_config.json" ]] || { echo "ERROR: adapter not found: $SFT_ADAPTER"; exit 1; }
[[ -f "$TEST_JSONL" ]]                      || { echo "ERROR: test set not found: $TEST_JSONL"; exit 1; }

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
INFER_TP=1
for _tp in 2 4; do (( _tp <= NUM_GPUS )) && INFER_TP=$_tp; done

echo "========================================================"
echo "Inference: rational_sft/qwen35_9b on SM non-thinking test"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Test rows: $(wc -l < "$TEST_JSONL")"
echo "Adapter:   $SFT_ADAPTER"
echo "Output:    $INFERENCE_OUT"
echo "GPUs:      $NUM_GPUS  (TP=$INFER_TP)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

"$PYTHON" scripts/inference_sft_qwen_vllm.py \
    --base-model             "$BASE_MODEL" \
    --sft-adapter            "$SFT_ADAPTER" \
    --input-data             "$TEST_JSONL" \
    --output                 "$INFERENCE_OUT" \
    --tensor-parallel-size   "$INFER_TP" \
    --gpu-memory-utilization 0.85 \
    --max-model-len          8192 \
    --max-new-tokens         8192 \
    --temperature            0.3 \
    --lora-rank              16 \
    --system-prompt          "You are a helpful and safe assistant. When a user asks about a sensitive topic, respond helpfully while avoiding harm. Always address the user's underlying need even if you cannot address the exact phrasing of their request."

echo ""
echo ">>> Inference done — $(wc -l < "$INFERENCE_OUT") rows -> $INFERENCE_OUT"
echo "========================================================"
echo "Done at: $(date)"
echo "========================================================"
