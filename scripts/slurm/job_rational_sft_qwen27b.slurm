#!/usr/bin/bash
# Rational-SFT pipeline for Qwen 2.5 27B.
# Trains on SCR reasoning chains (rational_full_output) using the same
# train/val/test p1_id split as the self-model SFT for a fair comparison.
#
# Matches job_qlora_pipeline.slurm exactly: 4-bit NF4 QLoRA, sft_qlora.py,
# no DeepSpeed, 4 GPUs, max_length=4096.
#
# Usage:
#   sbatch scripts/slurm/job_rational_sft_qwen27b.slurm
#
# Skip training:
#   SKIP_TRAIN=1 sbatch scripts/slurm/job_rational_sft_qwen27b.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=24:00:00
#SBATCH --job-name=rational_sft_qwen27b
#SBATCH --output=$PROJECT_ROOT/logs/rational_sft_qwen27b_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/rational_sft_qwen27b_%j.err

set -euo pipefail

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
TORCHRUN="$CONDA_PREFIX/envs/vllm-serve/bin/torchrun"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_AIO=0
export DS_SKIP_CUDA_CHECK=1
export TMPDIR=/tmp
export VLLM_USE_V1=0

MODEL="qwen35_27b"
BASE_MODEL="$HF_HOME/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"

DATA_DIR="${ROOT}/output/SFT/rational_sft/qwen35_27b"
TRAIN_JSONL="${DATA_DIR}/train.jsonl"
VAL_JSONL="${DATA_DIR}/val.jsonl"
TEST_JSONL="${DATA_DIR}/test.jsonl"

SFT_ADAPTER="${DATA_DIR}/models/qwen35_27b"
INFERENCE_OUT="${DATA_DIR}/inference/qwen35_27b.jsonl"
JUDGE_DIR="${DATA_DIR}/judge/qwen35_27b"

# Reuse baseline already computed for self-model pipeline (same test set)
BASELINE="${ROOT}/output/SFT/self_model/qwen35_27b/modelwise_test_baselines/qwen35_27b.jsonl"

_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma-4 not found"; exit 1; }

mkdir -p "$SFT_ADAPTER" "$(dirname "$INFERENCE_OUT")" "$JUDGE_DIR"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
MASTER_PORT=$(( (SLURM_JOB_ID % 10000) + 20000 ))
INFER_TP=1
for _tp in 2 4 8; do (( _tp <= NUM_GPUS )) && INFER_TP=$_tp; done

echo "========================================================"
echo "Rational-SFT (QLoRA) pipeline: $MODEL"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Base:      $BASE_MODEL"
echo "Data:      $DATA_DIR"
echo "Judge:     $GEMMA4_PATH"
echo "GPUs:      $NUM_GPUS  (infer TP=$INFER_TP)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Stage 1: 4-bit QLoRA training ────────────────────────────────────────────
SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [[ "$SKIP_TRAIN" == "1" ]]; then
    [[ -f "$SFT_ADAPTER/adapter_config.json" ]] || \
        { echo "ERROR: SKIP_TRAIN=1 but adapter not found at $SFT_ADAPTER"; exit 1; }
    echo ">>> STAGE 1: Skipped (SKIP_TRAIN=1)"
else
    echo ">>> STAGE 1: QLoRA training (4-bit NF4, no DeepSpeed)"
    "$TORCHRUN" --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" scripts/sft_qlora.py \
        --base-model            "$BASE_MODEL" \
        --train-jsonl           "$TRAIN_JSONL" \
        --val-jsonl             "$VAL_JSONL" \
        --output-dir            "$SFT_ADAPTER" \
        --epochs                3 \
        --per-device-batch-size 1 \
        --grad-accum            $(( 64 / NUM_GPUS )) \
        --learning-rate         1e-4 \
        --warmup-ratio          0.05 \
        --lora-r                16 \
        --lora-alpha            32 \
        --lora-targets          "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
        --max-length            4096 \
        --logging-steps         1 \
        --seed                  42 \
        --wandb-project         linguasafe-sft-dpo \
        --run-name              "qlora_rational_qwen35_27b"
    echo ">>> Stage 1 complete — adapter at $SFT_ADAPTER"
fi
echo ""

# ── Stage 2: Inference ────────────────────────────────────────────────────────
echo ">>> STAGE 2: Inference (TP=$INFER_TP)"
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
    --system-prompt          "You are a helpful assistant." \
    --enable-thinking
echo ">>> Stage 2 complete — $INFERENCE_OUT"
echo ""

sleep 30

# ── Stage 3: Gemma-4 judge ────────────────────────────────────────────────────
echo ">>> STAGE 3: Judge"
_SLURM_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
[[ -n "$_SLURM_CUDA_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA_DEVICES"
JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENFORCE_EAGER=0
echo "Judge TP=$JUDGE_TP"

"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$INFERENCE_OUT" \
    --baseline       "$BASELINE" \
    --output-dir     "$JUDGE_DIR" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "$MODEL" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0
echo ">>> Stage 3 complete — $JUDGE_DIR"

echo ""
echo "========================================================"
echo "Rational-SFT pipeline done at: $(date)"
echo "========================================================"
