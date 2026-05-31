#!/usr/bin/bash
# QLoRA SFT for Llama-3.3-70B-Instruct using AWQ INT4 base model.
# Two independent runs controlled by DATA_MODE:
#
#   DATA_MODE=self     sbatch scripts/slurm/job_qlora_awq_llama70b.slurm
#     → trains on llama70b's own O2 wins (self-model distillation)
#
#   DATA_MODE=teacher  sbatch scripts/slurm/job_qlora_awq_llama70b.slurm
#     → trains on Qwen3.5-122B teacher outputs (cross-model distillation)
#
# Skip flags:
#   SKIP_TRAIN=1     SKIP_INFERENCE=1  — jump straight to judge
#   DATA_MODE=self   SKIP_TRAIN=1      — adapter already saved

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu3
#SBATCH --time=24:00:00
#SBATCH --job-name=qlora_awq_%x
#SBATCH --output=$PROJECT_ROOT/logs/qlora_awq_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/qlora_awq_%j.err

set -euo pipefail

DATA_MODE="${DATA_MODE:?Set DATA_MODE=self or DATA_MODE=teacher before sbatch}"
[[ "$DATA_MODE" =~ ^(self|teacher)$ ]] || { echo "ERROR: DATA_MODE must be 'self' or 'teacher'"; exit 1; }

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
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_BATCH_INVARIANT=1

# ── gpu1 broken-GPU filter ────────────────────────────────────────────────────
if [[ "$(hostname -s)" == "gpu1" ]]; then
    CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^3$' | grep -v '^7$' | paste -sd ',')
    export CUDA_VISIBLE_DEVICES
fi
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
MASTER_PORT=$(( (SLURM_JOB_ID % 10000) + 20000 ))

# ── AWQ model path ────────────────────────────────────────────────────────────
AWQ_HF_ID="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4"
USER_CACHE="$HF_HOME"
AWQ_DIR_NAME="models--$(echo "$AWQ_HF_ID" | sed 's|/|--|g')"

_find_snapshot() {
    local cache_dir="$1/$AWQ_DIR_NAME"
    find "$cache_dir/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true
}
_is_complete() { [[ -f "$1/config.json" ]] && [[ -f "$1/tokenizer.json" || -f "$1/tokenizer.model" ]]; }

BASE_MODEL=$(_find_snapshot "$USER_CACHE" || true)
[[ -n "$BASE_MODEL" ]] && ! _is_complete "$BASE_MODEL" && BASE_MODEL=""
for _cache in "$HF_HOME" "$HF_HOME"; do
    [[ -n "$BASE_MODEL" ]] && break
    BASE_MODEL=$(_find_snapshot "$_cache" || true)
    [[ -n "$BASE_MODEL" ]] && ! _is_complete "$BASE_MODEL" && BASE_MODEL=""
done

if [[ -z "$BASE_MODEL" ]]; then
    echo "AWQ model not found in cache — downloading $AWQ_HF_ID ..."
    BASE_MODEL=$("$PYTHON" -c "
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='${AWQ_HF_ID}', cache_dir='${USER_CACHE}')
print(path)
" 2>&1 | tail -1)
    echo "Downloaded to: $BASE_MODEL"
fi
[[ -f "$BASE_MODEL/config.json" ]] || { echo "ERROR: AWQ model not found: $BASE_MODEL"; exit 1; }

# ── Gemma-4 judge path ────────────────────────────────────────────────────────
GEMMA4_PATH=$(find "$HF_HOME/models--google--gemma-4-31B-it/snapshots" \
    -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma4 not found"; exit 1; }

# ── Data and output paths ─────────────────────────────────────────────────────
SELF_DIR="output/SFT/self_model/llama70b"
TEACHER_DIR="output/SFT/qwen35_122b_teacher/combined_dna_linguasafe"
LLAMA70B_TEACHER_OUT="output/SFT/qwen35_122b_teacher/llama70b"

if [[ "$DATA_MODE" == "self" ]]; then
    TRAIN_JSONL="$SELF_DIR/train.jsonl"
    VAL_JSONL="$SELF_DIR/val.jsonl"
    TEST_JSONL="$SELF_DIR/test.jsonl"
    BASELINE="$SELF_DIR/modelwise_test_baselines/llama70b.jsonl"
    SFT_ADAPTER="$SELF_DIR/models/llama70b_awq"
    INFERENCE_OUT="$SELF_DIR/inference/llama70b_awq.jsonl"
    JUDGE_DIR="$SELF_DIR/judge/llama70b_awq"
    RUN_NAME="qlora_awq_llama70b_self"
else
    TRAIN_JSONL="$TEACHER_DIR/train.jsonl"
    VAL_JSONL="$TEACHER_DIR/val.jsonl"
    TEST_JSONL="$TEACHER_DIR/test.jsonl"
    BASELINE="$TEACHER_DIR/modelwise_test_baselines/llama33_70b.jsonl"
    SFT_ADAPTER="$LLAMA70B_TEACHER_OUT/models/llama70b_awq"
    INFERENCE_OUT="$LLAMA70B_TEACHER_OUT/inference/llama70b_awq.jsonl"
    JUDGE_DIR="$LLAMA70B_TEACHER_OUT/judge/llama70b_awq"
    RUN_NAME="qlora_awq_llama70b_teacher"
fi

[[ -f "$TRAIN_JSONL" ]] || { echo "ERROR: train split not found: $TRAIN_JSONL"; exit 1; }
[[ -f "$BASELINE"    ]] || { echo "ERROR: baseline file not found: $BASELINE"; exit 1; }
mkdir -p "$SFT_ADAPTER" "$(dirname "$INFERENCE_OUT")" "$JUDGE_DIR"

echo "========================================================"
echo "QLoRA-AWQ SFT: llama70b  [DATA_MODE=$DATA_MODE]"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "GPUs:       $NUM_GPUS"
echo "Base model: $BASE_MODEL"
echo "Train:      $TRAIN_JSONL  ($(wc -l < "$TRAIN_JSONL") rows)"
echo "Val:        $VAL_JSONL    ($(wc -l < "$VAL_JSONL") rows)"
echo "Test:       $TEST_JSONL   ($(wc -l < "$TEST_JSONL") rows)"
echo "Adapter:    $SFT_ADAPTER"
echo "Judge:      $GEMMA4_PATH"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Stage 1: AWQ QLoRA training ───────────────────────────────────────────────
SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [[ "$SKIP_TRAIN" == "1" ]]; then
    [[ -f "$SFT_ADAPTER/adapter_config.json" ]] || { echo "ERROR: SKIP_TRAIN=1 but adapter not found"; exit 1; }
    echo ">>> STAGE 1: Skipped (SKIP_TRAIN=1)"
else
    echo ">>> STAGE 1: AWQ QLoRA Training"
    "$TORCHRUN" --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
        scripts/sft_qlora_awq.py \
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
        --max-length            1024 \
        --logging-steps         1 \
        --seed                  42 \
        --wandb-project         linguasafe-sft-dpo \
        --run-name              "$RUN_NAME"
    echo ">>> Stage 1 complete — adapter at $SFT_ADAPTER"
fi
echo ""

# ── Stage 2: Inference ───────────────────────────────────────────────────────
INFER_TP=1
for _tp in 2 4; do (( _tp <= NUM_GPUS )) && INFER_TP=$_tp; done

SKIP_INFERENCE="${SKIP_INFERENCE:-0}"
if [[ "$SKIP_INFERENCE" == "1" ]]; then
    [[ -f "$INFERENCE_OUT" ]] || { echo "ERROR: SKIP_INFERENCE=1 but output not found"; exit 1; }
    echo ">>> STAGE 2: Skipped (SKIP_INFERENCE=1) — $(wc -l < "$INFERENCE_OUT") rows"
else
    echo ">>> STAGE 2: Inference (TP=$INFER_TP)"
    "$PYTHON" scripts/inference_sft_qwen_vllm.py \
        --base-model             "$BASE_MODEL" \
        --sft-adapter            "$SFT_ADAPTER" \
        --input-data             "$TEST_JSONL" \
        --output                 "$INFERENCE_OUT" \
        --tensor-parallel-size   "$INFER_TP" \
        --gpu-memory-utilization 0.85 \
        --max-model-len          4096 \
        --max-new-tokens         2048 \
        --temperature            0.3 \
        --lora-rank              16 \
        --system-prompt          "You are a helpful assistant."
    echo ">>> Stage 2 complete — $(wc -l < "$INFERENCE_OUT") rows → $INFERENCE_OUT"
fi
echo ""

# ── Stage 3: Gemma-4 judge ────────────────────────────────────────────────────
echo ">>> STAGE 3: Gemma-4 Judge"
_SLURM_CUDA="${CUDA_VISIBLE_DEVICES:-}"
source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
[[ -n "$_SLURM_CUDA" ]] && export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA"
JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENFORCE_EAGER=0

"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$INFERENCE_OUT" \
    --baseline       "$BASELINE" \
    --output-dir     "$JUDGE_DIR" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "llama70b_awq_${DATA_MODE}" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0

echo ">>> Stage 3 complete — results in $JUDGE_DIR"
echo ""
echo "========================================================"
echo "QLoRA-AWQ pipeline done: llama70b [$DATA_MODE]  $(date)"
echo "========================================================"
