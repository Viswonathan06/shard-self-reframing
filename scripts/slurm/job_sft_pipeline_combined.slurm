#!/usr/bin/bash
# Full SFT pipeline: baseline → train → inference → judge (Gemma4)
# Stage 0 generates baseline responses if not already present.
# Runs on the combined DNA+LinguaSafe dataset (teacher = Qwen3.5-122B).
#
# Usage:
#   MODEL=llama8b        sbatch scripts/slurm/job_sft_pipeline_combined.slurm
#   MODEL=qwen35_9b      sbatch scripts/slurm/job_sft_pipeline_combined.slurm
#   MODEL=qwen35_27b     sbatch scripts/slurm/job_sft_pipeline_combined.slurm
#   MODEL=mistral7b      sbatch scripts/slurm/job_sft_pipeline_combined.slurm
#   MODEL=phi4           sbatch scripts/slurm/job_sft_pipeline_combined.slurm
#
# Skip training (adapter already saved), run inference + judge only:
#   MODEL=phi4 SKIP_TRAIN=1 sbatch scripts/slurm/job_sft_pipeline_combined.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:8
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=20:00:00
#SBATCH --job-name=sft_pipeline_%x
#SBATCH --output=$PROJECT_ROOT/logs/sft_pipeline_%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/sft_pipeline_%x_%j.err

set -euo pipefail

MODEL="${MODEL:?Set MODEL=llama8b|qwen35_9b|mistral7b before sbatch}"
ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

# Python with torch/transformers/peft/vllm (vllm-serve conda env)
PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export HF_HUB_DISABLE_XET=1

# ── DeepSpeed ─────────────────────────────────────────────────────────────────
# ZeRO-2: shards optimizer states + gradients; each GPU holds full params.
#          Good for 8B–27B on 4×49 GiB.
# ZeRO-3: shards everything; needed for 70B+.
DS_CONFIG="$ROOT/scripts/ds_zero2.json"
TORCHRUN="$CONDA_PREFIX/envs/vllm-serve/bin/torchrun"

# Skip async-IO op check — it crashes on network filesystems during cleanup.
export DS_BUILD_AIO=0
# Redirect DS temp files off NFS so shutil.rmtree in is_compatible() succeeds.
export TMPDIR=/tmp
# System CUDA (13.0 on gpu5) doesn't match torch's CUDA (12.8); skip the check
# so CPUAdam can JIT-compile with the installed nvcc instead of erroring out.
export DS_SKIP_CUDA_CHECK=1
# Reduce fragmentation-induced OOM (allocator reuses existing segments first).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Model-specific paths ──────────────────────────────────────────────────────

declare -A BASE_MODELS=(
    [llama8b]="$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    [qwen35_9b]="$HF_HOME/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    [qwen35_9b_thinking]="$HF_HOME/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    [qwen35_27b]="$HF_HOME/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    [qwen35_27b_thinking]="$HF_HOME/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    [mistral7b]="$HF_HOME/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
    [phi4]="$HF_HOME/models--microsoft--phi-4/snapshots/932b33c0ec9ca189badeb22480721a8de9d0e006"
)

BASE_MODEL="${BASE_MODELS[$MODEL]:?Unknown MODEL=$MODEL}"

# ── Shared paths ──────────────────────────────────────────────────────────────

# SELF_MODEL=1  → train on the model's own O2-wins (self-distillation)
#              data lives under output/SFT/self_model/${MODEL}/
# SELF_MODEL=0  → train on qwen35_122b teacher data (default)
SELF_MODEL="${SELF_MODEL:-0}"

COMBINED_DIR="output/SFT/qwen35_122b_teacher/combined_dna_linguasafe"
REASONING_DIR="output/SFT/qwen35_122b_teacher/combined_dna_linguasafe_reasoning"
SELF_MODEL_DIR="output/SFT/self_model/${MODEL}"

# Reasoning models (Qwen3 series) train on data with <think> tokens and
# need a larger context window to avoid truncating the chain-of-thought.
# They also need a smaller per-device batch (1 vs 2) because ZeRO-2 keeps a
# full copy of the ~18 GB base model on each GPU, leaving little room for
# 10k-token activations. Grad-accum is doubled to keep the effective batch
# size (1×16×4 = 64) identical to the non-reasoning setting (2×8×4 = 64).
if [[ "$MODEL" == qwen35* ]]; then
    TRAIN_JSONL="$REASONING_DIR/train.jsonl"
    VAL_JSONL="$REASONING_DIR/val.jsonl"
    TEST_JSONL="$REASONING_DIR/test.jsonl"
    MAX_LENGTH=8192
    PER_DEVICE_BS=1
    GRAD_ACCUM=8
    # 27B in bf16 = ~54 GB — exceeds one GPU's 47 GB, so ZeRO-3 is required to
    # shard params across all 4 GPUs (~13.5 GB/GPU). Smaller qwen35 uses ZeRO-2.
    # Attention activations scale as seq_len²: even with ZeRO-3, 10k tokens OOMs
    # a 47 GB GPU. Data is p95=~4800 tokens, so 4096 covers nearly everything.
    if [[ "$MODEL" == qwen35_27b* ]]; then
        DS_CONFIG="$ROOT/scripts/ds_zero3.json"
        MAX_LENGTH=4096  # 6144 OOMs 8×47GB even with ZeRO-3+CPU optimizer; activations dominate
    fi
else
    TRAIN_JSONL="$COMBINED_DIR/train.jsonl"
    VAL_JSONL="$COMBINED_DIR/val.jsonl"
    TEST_JSONL="$COMBINED_DIR/test.jsonl"
    MAX_LENGTH=2048
    PER_DEVICE_BS=2
    GRAD_ACCUM=4
fi

# Override data paths when training on self-generated data.
# Hyperparams (BS, max_length, DeepSpeed config) are inherited from above.
if [[ "$SELF_MODEL" == "1" ]]; then
    TRAIN_JSONL="$SELF_MODEL_DIR/train.jsonl"
    VAL_JSONL="$SELF_MODEL_DIR/val.jsonl"
    TEST_JSONL="$SELF_MODEL_DIR/test.jsonl"
    [[ -f "$TRAIN_JSONL" ]] || { echo "ERROR: self-model train split not found: $TRAIN_JSONL"; exit 1; }
fi

if [[ "$SELF_MODEL" == "1" ]]; then
    BASELINE="$SELF_MODEL_DIR/modelwise_test_baselines/${MODEL}.jsonl"
    SFT_ADAPTER="$SELF_MODEL_DIR/models/${MODEL}"
    INFERENCE_OUT="$SELF_MODEL_DIR/inference/${MODEL}.jsonl"
    JUDGE_DIR="$SELF_MODEL_DIR/judge/${MODEL}"
elif [[ "$MODEL" == qwen35* ]]; then
    # qwen35 trains on reasoning data — keep outputs co-located with that data
    BASELINE="$COMBINED_DIR/modelwise_test_baselines/${MODEL}.jsonl"
    SFT_ADAPTER="$REASONING_DIR/models/${MODEL}"
    INFERENCE_OUT="$REASONING_DIR/inference/${MODEL}.jsonl"
    JUDGE_DIR="$REASONING_DIR/judge/${MODEL}"
else
    BASELINE="$COMBINED_DIR/modelwise_test_baselines/${MODEL}.jsonl"
    SFT_ADAPTER="$COMBINED_DIR/models/${MODEL}"
    INFERENCE_OUT="$COMBINED_DIR/inference/${MODEL}.jsonl"
    JUDGE_DIR="$COMBINED_DIR/judge/${MODEL}"
fi

# Resolve Gemma4 path without sourcing the env script — that script sets
# CUDA_VISIBLE_DEVICES, NCCL flags, and PYTHONPATH side-effects that corrupt
# the training environment. We just find the snapshot directly.
_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma4 snapshot not found in $_GEMMA4_BASE"; exit 1; }

mkdir -p "$SFT_ADAPTER" "$(dirname "$BASELINE")" "$(dirname "$INFERENCE_OUT")" "$JUDGE_DIR"

# ── Download base model if not cached on this node ────────────────────────────
if [[ ! -f "$BASE_MODEL/config.json" ]]; then
    echo "Base model not found at $BASE_MODEL — trying user cache..."
    HF_ID=$(basename "$(dirname "$(dirname "$BASE_MODEL")")" | sed 's/^models--//;s/--/\//')
    USER_CACHE="$HF_HOME"
    mkdir -p "$USER_CACHE"
    BASE_MODEL=$("$PYTHON" -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='${HF_ID}',
    cache_dir='${USER_CACHE}',
)
print(path)
" 2>/dev/null | tail -1)
    echo "Downloaded to: $BASE_MODEL"
fi

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
MASTER_PORT=$(( (SLURM_JOB_ID % 10000) + 20000 ))

# Determine inference tensor-parallel-size (largest power-of-2 ≤ NUM_GPUS).
# phi4 has 10 KV heads → max valid TP is 1; custom_all_reduce also fails >1 on phi4.
INFER_TP=1
for _tp in 2 4 8; do (( _tp <= NUM_GPUS )) && INFER_TP=$_tp; done
[[ "$MODEL" == "phi4" ]] && INFER_TP=1

SFT_SYSTEM_PROMPT="You are a helpful assistant."

THINKING_FLAG=""
[[ "$MODEL" == *_thinking ]] && THINKING_FLAG="--enable-thinking"

# vLLM v1 worker processes lose CUDA_VISIBLE_DEVICES on some nodes; v0 is stable.
export VLLM_USE_V1=0

echo "========================================================"
echo "SFT pipeline: $MODEL"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Base model:   $BASE_MODEL"
echo "Judge model:  $GEMMA4_PATH"
echo "GPUs:         $NUM_GPUS  (infer TP=$INFER_TP)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Stage 0: Baseline generation (skip if baseline already exists) ─────────────
if [[ -f "$BASELINE" ]]; then
    echo ">>> STAGE 0: Skipped — baseline already at $BASELINE"
else
    echo ">>> STAGE 0: Generating baseline (no adapter)"
    "$PYTHON" scripts/inference_sft_qwen_vllm.py \
        --base-model             "$BASE_MODEL" \
        --input-data             "$COMBINED_DIR/test.jsonl" \
        --output                 "$BASELINE" \
        --tensor-parallel-size   "$INFER_TP" \
        --gpu-memory-utilization 0.85 \
        --max-model-len          8192 \
        --max-new-tokens         8192 \
        --temperature            0.3 \
        --system-prompt          "$SFT_SYSTEM_PROMPT" \
        $THINKING_FLAG
    echo ">>> Stage 0 complete — $(wc -l < "$BASELINE") baseline rows written to $BASELINE"
fi
echo ""

# ── Stage 1: SFT training ─────────────────────────────────────────────────────

SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [[ "$SKIP_TRAIN" == "1" ]]; then
    [[ -f "$SFT_ADAPTER/adapter_config.json" ]] || { echo "ERROR: SKIP_TRAIN=1 but adapter not found at $SFT_ADAPTER"; exit 1; }
    echo ">>> STAGE 1: Skipped (SKIP_TRAIN=1) — using adapter at $SFT_ADAPTER"
else
    echo ">>> STAGE 1: Training"
    # Scale grad_accum to keep effective batch = 64 regardless of GPU count
    TARGET_BATCH=64
    GRAD_ACCUM=$(( TARGET_BATCH / PER_DEVICE_BS / NUM_GPUS ))
    [[ $GRAD_ACCUM -lt 1 ]] && GRAD_ACCUM=1
    echo "GPUs=$NUM_GPUS  per_device_bs=$PER_DEVICE_BS  grad_accum=$GRAD_ACCUM  eff_batch=$(( NUM_GPUS * PER_DEVICE_BS * GRAD_ACCUM ))"
    "$TORCHRUN" --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" scripts/sft_v2.py \
        --base-model            "$BASE_MODEL" \
        --train-jsonl           "$TRAIN_JSONL" \
        --val-jsonl             "$VAL_JSONL" \
        --output-dir            "$SFT_ADAPTER" \
        --epochs                3 \
        --per-device-batch-size "${PER_DEVICE_BS}" \
        --grad-accum            "${GRAD_ACCUM}" \
        --learning-rate         1e-4 \
        --warmup-ratio          0.05 \
        --lora-r                16 \
        --lora-alpha            32 \
        --lora-targets          "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
        --max-length            "${MAX_LENGTH}" \
        --logging-steps         1 \
        --seed                  42 \
        --wandb-project         linguasafe-sft-dpo \
        --run-name              "$( [[ "$SELF_MODEL" == "1" ]] && echo "sft_selfmodel_${MODEL}" || echo "sft_combined_${MODEL}" )" \
        --deepspeed             "$DS_CONFIG"
    echo ">>> Stage 1 complete — adapter saved to $SFT_ADAPTER"
fi  # end SKIP_TRAIN
echo ""

# ── Stage 2: Inference on heldout test set ────────────────────────────────────

echo "Inference tensor-parallel-size: $INFER_TP"

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
    $THINKING_FLAG

echo ">>> Stage 2 complete — inference saved to $INFERENCE_OUT"
echo ""
sleep 30  # allow vLLM inference process to fully release GPU memory before judge starts

# ── Stage 3: Gemma4 judge ─────────────────────────────────────────────────────

echo ">>> STAGE 3: Judge"
source scripts/shell/setup_gemma4_vllm_env.sh  # sets CUDA_VISIBLE_DEVICES=0,1,2,3, NCCL, PYTHONPATH
# TP must equal actual visible GPUs (setup script may restrict CUDA_VISIBLE_DEVICES)
JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=6144
echo "Judge tensor-parallel-size: $JUDGE_TP"

"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$INFERENCE_OUT" \
    --baseline       "$BASELINE" \
    --output-dir     "$JUDGE_DIR" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "$MODEL" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0

echo ">>> Stage 3 complete — results in $JUDGE_DIR"
echo ""
echo "========================================================"
echo "Pipeline done: $MODEL"
echo "========================================================"
