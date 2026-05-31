#!/usr/bin/bash
# QLoRA SFT pipeline: train → inference → judge for large models (27B–70B).
# Uses 4-bit NF4 quantization so the full model fits per-GPU on A6000s.
# No DeepSpeed — standard DDP via torchrun.
#
# Usage:
#   MODEL=qwen35_27b  sbatch scripts/slurm/job_qlora_pipeline.slurm
#   MODEL=llama70b    sbatch scripts/slurm/job_qlora_pipeline.slurm
#   MODEL=mistral24b  sbatch scripts/slurm/job_qlora_pipeline.slurm
#
# Self-model distillation (train on the model's own O2 wins):
#   MODEL=qwen35_27b SELF_MODEL=1 sbatch scripts/slurm/job_qlora_pipeline.slurm
#
# Skip training (adapter already saved), run inference + judge only:
#   MODEL=mistral24b SELF_MODEL=1 SKIP_TRAIN=1 sbatch scripts/slurm/job_qlora_pipeline.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=24:00:00
#SBATCH --job-name=qlora_sft
#SBATCH --output=$PROJECT_ROOT/logs/qlora_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/qlora_%j.err

echo "=== QLoRA pipeline starting — job ${SLURM_JOB_ID} on $(hostname) ==="
echo "MODEL=${MODEL:-UNSET}"

if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL is not set. Submit with: MODEL=qwen35_27b sbatch ..."
    exit 1
fi

set -euo pipefail

[[ "$MODEL" =~ ^(qwen35_27b|qwen35_27b_thinking|llama70b|mistral24b)$ ]] || { echo "ERROR: Unknown MODEL=$MODEL. Must be qwen35_27b, qwen35_27b_thinking, llama70b, or mistral24b."; exit 1; }
# MODEL_TAG overrides the output directory name without changing model identity.
# Use to avoid clobbering an existing run: MODEL_TAG=mistral24b_rerun
MODEL_TAG="${MODEL_TAG:-$MODEL}"
ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
TORCHRUN="$CONDA_PREFIX/envs/vllm-serve/bin/torchrun"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_AIO=0
export DS_SKIP_CUDA_CHECK=1
export TMPDIR=/tmp

# ── Model-specific paths ──────────────────────────────────────────────────────

# HF repo IDs — used to download the model if not already cached on this node.
# Models are downloaded to $HF_HOME (writable on all nodes).
declare -A HF_IDS=(
    [qwen35_27b]="Qwen/Qwen3.5-27B"
    [qwen35_27b_thinking]="Qwen/Qwen3.5-27B"
    [llama70b]="meta-llama/Llama-3.3-70B-Instruct"
    [mistral24b]="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
)

HF_ID="${HF_IDS[$MODEL]:?Unknown MODEL=$MODEL}"

USER_CACHE="$HF_HOME"
mkdir -p "$USER_CACHE"

# Derive cache dir name from HF ID (e.g. Qwen/Qwen3.5-27B → models--Qwen--Qwen3.5-27B)
_MODEL_DIR_NAME="models--$(echo "$HF_ID" | sed 's|/|--|g')"

# Search order: user cache → guam's cache → shared hub → download
# find returns non-zero when path missing — use || true to prevent set -e from triggering
_find_snapshot() {
    local cache_dir="$1/$_MODEL_DIR_NAME"
    find "$cache_dir/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true
}

_is_complete() { [[ -f "$1/config.json" ]] && ([[ -f "$1/tokenizer.json" ]] || [[ -f "$1/tokenizer.model" ]] || [[ -f "$1/tekken.json" ]]); }

BASE_MODEL=$(_find_snapshot "$USER_CACHE" || true)
[[ -n "$BASE_MODEL" ]] && ! _is_complete "$BASE_MODEL" && BASE_MODEL=""
if [[ -z "$BASE_MODEL" ]]; then
    BASE_MODEL=$(_find_snapshot "$HF_HOME" || true)
    [[ -n "$BASE_MODEL" ]] && ! _is_complete "$BASE_MODEL" && BASE_MODEL=""
fi
if [[ -z "$BASE_MODEL" ]]; then
    BASE_MODEL=$(_find_snapshot "$HF_HOME" || true)
    [[ -n "$BASE_MODEL" ]] && ! _is_complete "$BASE_MODEL" && BASE_MODEL=""
fi

if [[ -z "$BASE_MODEL" ]]; then
    echo "Model not found or incomplete in cache — downloading $HF_ID to $USER_CACHE ..."
    BASE_MODEL=$("$PYTHON" -c "
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='${HF_ID}', cache_dir='${USER_CACHE}')
print(path)
" 2>&1 | tail -1)
    echo "Downloaded to: $BASE_MODEL"
fi

[[ -f "$BASE_MODEL/config.json" ]] || { echo "ERROR: base model not found after download: $BASE_MODEL"; exit 1; }
[[ -f "$BASE_MODEL/tokenizer.json" || -f "$BASE_MODEL/tokenizer.model" || -f "$BASE_MODEL/tekken.json" ]] || { echo "ERROR: tokenizer missing at $BASE_MODEL"; exit 1; }
echo "Using base model: $BASE_MODEL"

# ── Training hyperparameters ──────────────────────────────────────────────────
# All large models use max_length=2048 by default (non-reasoning).
# Qwen3.5-27B is a reasoning model and uses the reasoning dataset + longer ctx.

SELF_MODEL="${SELF_MODEL:-1}"
COMBINED_DIR="output/SFT/qwen35_122b_teacher/combined_dna_linguasafe"
REASONING_DIR="output/SFT/qwen35_122b_teacher/combined_dna_linguasafe_reasoning"
SELF_MODEL_DIR="output/SFT/self_model/${MODEL}"

if [[ "$MODEL" == qwen35* ]]; then
    TRAIN_JSONL="$REASONING_DIR/train.jsonl"
    VAL_JSONL="$REASONING_DIR/val.jsonl"
    TEST_JSONL="$REASONING_DIR/test.jsonl"
    MAX_LENGTH=4096
elif [[ "$MODEL" == "llama70b" ]]; then
    TRAIN_JSONL="$COMBINED_DIR/train.jsonl"
    VAL_JSONL="$COMBINED_DIR/val.jsonl"
    TEST_JSONL="$COMBINED_DIR/test.jsonl"
    MAX_LENGTH=512  # 70B 4-bit ~35 GiB + lm_head ~2 GiB leaves ~10 GiB; 1024 OOMs lm_head forward
else
    TRAIN_JSONL="$COMBINED_DIR/train.jsonl"
    VAL_JSONL="$COMBINED_DIR/val.jsonl"
    TEST_JSONL="$COMBINED_DIR/test.jsonl"
    MAX_LENGTH=2048
fi

if [[ "$SELF_MODEL" == "1" ]]; then
    TRAIN_JSONL="$SELF_MODEL_DIR/train.jsonl"
    VAL_JSONL="$SELF_MODEL_DIR/val.jsonl"
    TEST_JSONL="$SELF_MODEL_DIR/test.jsonl"
    [[ -f "$TRAIN_JSONL" ]] || { echo "ERROR: self-model train split not found: $TRAIN_JSONL"; exit 1; }
fi

if [[ "$SELF_MODEL" == "1" ]]; then
    BASELINE="$SELF_MODEL_DIR/modelwise_test_baselines/${MODEL}.jsonl"
    SFT_ADAPTER="$SELF_MODEL_DIR/models/${MODEL_TAG}"
    INFERENCE_OUT="$SELF_MODEL_DIR/inference/${MODEL_TAG}.jsonl"
    JUDGE_DIR="$SELF_MODEL_DIR/judge/${MODEL_TAG}"
elif [[ "$MODEL" == qwen35* ]]; then
    # qwen35 models train on reasoning data — keep outputs co-located with that data
    BASELINE="$COMBINED_DIR/modelwise_test_baselines/${MODEL}.jsonl"
    SFT_ADAPTER="$REASONING_DIR/models/${MODEL_TAG}"
    INFERENCE_OUT="$REASONING_DIR/inference/${MODEL_TAG}.jsonl"
    JUDGE_DIR="$REASONING_DIR/judge/${MODEL_TAG}"
else
    BASELINE="$COMBINED_DIR/modelwise_test_baselines/${MODEL}.jsonl"
    SFT_ADAPTER="$COMBINED_DIR/models/${MODEL_TAG}"
    INFERENCE_OUT="$COMBINED_DIR/inference/${MODEL_TAG}.jsonl"
    JUDGE_DIR="$COMBINED_DIR/judge/${MODEL_TAG}"
fi

# ── Resolve Gemma4 judge path ─────────────────────────────────────────────────
_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma4 snapshot not found"; exit 1; }

mkdir -p "$SFT_ADAPTER" "$(dirname "$INFERENCE_OUT")" "$JUDGE_DIR"

# gpu1 physical GPUs 3 and 7 are broken (CUDA reports fewer devices than allocated).
# Strip them from CUDA_VISIBLE_DEVICES so we never land on them.
if [[ "$(hostname -s)" == "gpu1" ]]; then
    CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^3$' | grep -v '^7$' | paste -sd ',')
    export CUDA_VISIBLE_DEVICES
    echo "gpu1: filtered CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
[[ $NUM_GPUS -lt 1 ]] && { echo "ERROR: no usable GPUs after filtering"; exit 1; }
MASTER_PORT=$(( (SLURM_JOB_ID % 10000) + 20000 ))

echo "========================================================"
echo "QLoRA pipeline: $MODEL"
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Base model:     $BASE_MODEL"
echo "Self-model:     $SELF_MODEL"
echo "Max length:     $MAX_LENGTH"
echo "Judge model:    $GEMMA4_PATH"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Stage 1: QLoRA training ───────────────────────────────────────────────────

SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [[ "$SKIP_TRAIN" == "1" ]]; then
    [[ -f "$SFT_ADAPTER/adapter_config.json" ]] || { echo "ERROR: SKIP_TRAIN=1 but adapter not found at $SFT_ADAPTER"; exit 1; }
    echo ">>> STAGE 1: Skipped (SKIP_TRAIN=1) — using adapter at $SFT_ADAPTER"
else
    echo ">>> STAGE 1: QLoRA Training"
    # mistral_common tokenizer requires a dedicated training script
    SFT_TRAIN_SCRIPT="scripts/sft_qlora.py"
    [[ "$MODEL" == mistral* ]] && SFT_TRAIN_SCRIPT="scripts/sft_qlora_mistral.py"
    "$TORCHRUN" --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" "$SFT_TRAIN_SCRIPT" \
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
        --max-length            "${MAX_LENGTH}" \
        --logging-steps         1 \
        --seed                  42 \
        --wandb-project         linguasafe-sft-dpo \
        --run-name              "$( [[ "$SELF_MODEL" == "1" ]] && echo "qlora_selfmodel_${MODEL_TAG}" || echo "qlora_combined_${MODEL_TAG}" )"
    echo ">>> Stage 1 complete — adapter saved to $SFT_ADAPTER"
fi
echo ""

# ── Stage 2: Inference ───────────────────────────────────────────────────────
# Pick largest power-of-2 ≤ NUM_GPUS for tensor parallel
INFER_TP=1
for _tp in 2 4 8; do (( _tp <= NUM_GPUS )) && INFER_TP=$_tp; done
echo "Inference tensor-parallel-size: $INFER_TP"

SFT_SYSTEM_PROMPT="You are a helpful assistant."
# Mistral SFT training folds system prompt into the user message; passing a
# separate system prompt at inference causes [SYSTEM_PROMPT] tokens to leak
# into the output. Use empty system prompt to match the training format.
[[ "$MODEL" == mistral* ]] && SFT_SYSTEM_PROMPT=""

THINKING_FLAG=""
[[ "$MODEL" == *_thinking ]] && THINKING_FLAG="--enable-thinking"

TOKENIZER_MODE="auto"
[[ "$MODEL" == mistral* ]] && TOKENIZER_MODE="mistral"

# vLLM v1 worker processes lose CUDA_VISIBLE_DEVICES on some nodes; v0 is stable.
export VLLM_USE_V1=0

SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
if [[ "$SKIP_INFERENCE" == "1" ]]; then
    [[ -f "$INFERENCE_OUT" ]] || { echo "ERROR: SKIP_INFERENCE=1 but output not found at $INFERENCE_OUT"; exit 1; }
    echo ">>> STAGE 2: Skipped (SKIP_INFERENCE=1) — using $(wc -l < "$INFERENCE_OUT") rows from $INFERENCE_OUT"
else
    echo ">>> STAGE 2: Inference"
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
        --system-prompt          "$SFT_SYSTEM_PROMPT" \
        --tokenizer-mode         "$TOKENIZER_MODE" \
        $THINKING_FLAG
    echo ">>> Stage 2 complete — $(wc -l < "$INFERENCE_OUT") rows written to $INFERENCE_OUT"
fi
echo ""

# ── Stage 3: Gemma4 judge ─────────────────────────────────────────────────────

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
echo "Judge tensor-parallel-size: $JUDGE_TP"

"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$INFERENCE_OUT" \
    --baseline       "$BASELINE" \
    --output-dir     "$JUDGE_DIR" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "$MODEL_TAG" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0

echo ">>> Stage 3 complete — results in $JUDGE_DIR"
echo ""
echo "========================================================"
echo "QLoRA pipeline done: $MODEL"
echo "========================================================"
