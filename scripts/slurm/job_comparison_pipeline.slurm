#!/usr/bin/bash
# Comparison pipeline: run inference + judge for both self-model and teacher-model
# SFT adapters on the shared common test set (output/SFT/comparison/{MODEL}/).
#
# The common test set is combined_dna_linguasafe/test.jsonl minus the model's
# self-model training prompts, ensuring neither adapter saw the test prompts.
#
# Usage:
#   MODEL=llama8b            sbatch scripts/slurm/job_comparison_pipeline.slurm
#   MODEL=qwen35_9b          sbatch scripts/slurm/job_comparison_pipeline.slurm
#   MODEL=phi4               sbatch scripts/slurm/job_comparison_pipeline.slurm
#   MODEL=mistral7b          sbatch scripts/slurm/job_comparison_pipeline.slurm
#   MODEL=qwen35_27b_thinking sbatch scripts/slurm/job_comparison_pipeline.slurm
#   MODEL=mistral24b         sbatch scripts/slurm/job_comparison_pipeline.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=comparison
#SBATCH --output=$PROJECT_ROOT/logs/comparison_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/comparison_%j.err

set -euo pipefail

MODEL="${MODEL:?Set MODEL before sbatch}"
ROOT="$PROJECT_ROOT"
cd "$ROOT"

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_AIO=0
export DS_SKIP_CUDA_CHECK=1
export TMPDIR=/tmp

# gpu1 physical GPUs 3 and 7 are broken; strip them before anything else.
if [[ "$(hostname -s)" == "gpu1" ]]; then
    CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^3$' | grep -v '^7$' | paste -sd ',')
    export CUDA_VISIBLE_DEVICES
    echo "gpu1: filtered CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# ── Model paths ───────────────────────────────────────────────────────────────
declare -A HF_IDS=(
    [llama8b]="meta-llama/Llama-3.1-8B-Instruct"
    [qwen35_9b]="Qwen/Qwen3.5-9B"
    [qwen35_27b_thinking]="Qwen/Qwen3.5-27B"
    [phi4]="microsoft/phi-4"
    [mistral7b]="mistralai/Mistral-7B-Instruct-v0.3"
    [mistral24b]="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
)

HF_ID="${HF_IDS[$MODEL]:?Unknown MODEL=$MODEL}"
_MODEL_DIR_NAME="models--$(echo "$HF_ID" | sed 's|/|--|g')"

_find_snapshot() {
    local cache_dir="$1/$_MODEL_DIR_NAME"
    find "$cache_dir/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true
}
_is_complete() { [[ -f "$1/config.json" ]] && ([[ -f "$1/tokenizer.json" ]] || [[ -f "$1/tokenizer.model" ]] || [[ -f "$1/tekken.json" ]]); }

# llama8b lives in guam's cache; search that first
BASE_MODEL=$(_find_snapshot "$HF_HOME" || true)
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
    echo "Model not found in cache — downloading $HF_ID ..."
    USER_CACHE="$HF_HOME"
    mkdir -p "$USER_CACHE"
    BASE_MODEL=$("$PYTHON" -c "
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='${HF_ID}', cache_dir='${USER_CACHE}')
print(path)
" 2>&1 | tail -1)
fi
[[ -f "$BASE_MODEL/config.json" ]] || { echo "ERROR: base model not found after search: $BASE_MODEL"; exit 1; }
echo "Using base model: $BASE_MODEL"

# ── Adapter paths ─────────────────────────────────────────────────────────────
SELF_ADAPTER="output/SFT/self_model/${MODEL}/models/${MODEL}"
# qwen35_27b_thinking trained on reasoning data — adapter lives in reasoning dir
if [[ "$MODEL" == "qwen35_27b_thinking" ]]; then
    TEACHER_ADAPTER="output/SFT/qwen35_122b_teacher/combined_dna_linguasafe_reasoning/models/${MODEL}"
else
    TEACHER_ADAPTER="output/SFT/qwen35_122b_teacher/combined_dna_linguasafe/models/${MODEL}"
fi

[[ -f "$SELF_ADAPTER/adapter_config.json" ]]    || { echo "ERROR: self-model adapter not found: $SELF_ADAPTER"; exit 1; }
[[ -f "$TEACHER_ADAPTER/adapter_config.json" ]] || { echo "ERROR: teacher-model adapter not found: $TEACHER_ADAPTER"; exit 1; }

# ── Comparison data paths ─────────────────────────────────────────────────────
COMP_DIR="output/SFT/comparison/${MODEL}"
TEST_JSONL="$COMP_DIR/test.jsonl"
SELF_INFER="$COMP_DIR/inference/self_model.jsonl"       # field: self_output + baseline_output
LARGE_INFER="$COMP_DIR/inference/large_model.jsonl"    # field: large_model_output + response
MERGED="$COMP_DIR/inference/comparison.jsonl"           # merged: self_output + large_model_output
JUDGE_DIR="$COMP_DIR/judge"

[[ -f "$TEST_JSONL" ]] || { echo "ERROR: common test set not found: $TEST_JSONL"; exit 1; }
mkdir -p "$COMP_DIR/inference" "$JUDGE_DIR"

# ── Shared settings ───────────────────────────────────────────────────────────
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

INFER_TP=1
for _tp in 2 4 8; do (( _tp <= NUM_GPUS )) && INFER_TP=$_tp; done
# phi4: 10 KV heads; mistral24b: use TP=2 max for LoRA compatibility
[[ "$MODEL" == "phi4"      ]] && INFER_TP=1
[[ "$MODEL" == "mistral24b" ]] && { (( INFER_TP > 2 )) && INFER_TP=2 || true; }

SYSTEM_PROMPT="You are a helpful assistant."

THINKING_FLAG=""
[[ "$MODEL" == *_thinking ]] && THINKING_FLAG="--enable-thinking"

TOKENIZER_MODE="auto"
[[ "$MODEL" == mistral* ]] && TOKENIZER_MODE="mistral"

export VLLM_USE_V1=0

# Resolve Gemma4 judge
_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma4 snapshot not found"; exit 1; }

echo "========================================================"
echo "Comparison pipeline: $MODEL"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Base model:   $BASE_MODEL"
echo "Self adapter: $SELF_ADAPTER"
echo "Teacher adp:  $TEACHER_ADAPTER"
echo "Test rows:    $(wc -l < "$TEST_JSONL")"
echo "Infer TP:     $INFER_TP"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

SKIP_INFER="${SKIP_INFER:-0}"

# ── Stage 1: Inference — self-model adapter ───────────────────────────────────
if [[ "$SKIP_INFER" == "1" ]]; then
    [[ -f "$SELF_INFER" ]]  || { echo "ERROR: SKIP_INFER=1 but $SELF_INFER not found"; exit 1; }
    [[ -f "$LARGE_INFER" ]] || { echo "ERROR: SKIP_INFER=1 but $LARGE_INFER not found"; exit 1; }
    echo ">>> STAGES 1+2: Skipped (SKIP_INFER=1)"
elif [[ -f "$SELF_INFER" && -f "$LARGE_INFER" ]]; then
    echo ">>> STAGES 1+2: Skipped — inference files already exist"
else
if [[ -f "$SELF_INFER" ]]; then
    echo ">>> STAGE 1: Skipped — self_model.jsonl already exists ($(wc -l < "$SELF_INFER") rows)"
else
    echo ">>> STAGE 1: Inference (self-model adapter)"
    "$PYTHON" scripts/inference_sft_qwen_vllm.py \
        --base-model             "$BASE_MODEL" \
        --sft-adapter            "$SELF_ADAPTER" \
        --input-data             "$TEST_JSONL" \
        --output                 "$SELF_INFER" \
        --output-key             self_output \
        --tensor-parallel-size   "$INFER_TP" \
        --gpu-memory-utilization 0.85 \
        --max-model-len          8192 \
        --max-new-tokens         8192 \
        --temperature            0.3 \
        --lora-rank              16 \
        --system-prompt          "$SYSTEM_PROMPT" \
        --tokenizer-mode         "$TOKENIZER_MODE" \
        $THINKING_FLAG
    echo ">>> Stage 1 complete — $(wc -l < "$SELF_INFER") rows"
    sleep 20
fi
echo ""

# ── Stage 2: Inference — large-model (teacher) adapter ───────────────────────
if [[ -f "$LARGE_INFER" ]]; then
    echo ">>> STAGE 2: Skipped — large_model.jsonl already exists ($(wc -l < "$LARGE_INFER") rows)"
else
    echo ">>> STAGE 2: Inference (large-model adapter)"
    "$PYTHON" scripts/inference_sft_qwen_vllm.py \
        --base-model             "$BASE_MODEL" \
        --sft-adapter            "$TEACHER_ADAPTER" \
        --input-data             "$TEST_JSONL" \
        --output                 "$LARGE_INFER" \
        --output-key             large_model_output \
        --tensor-parallel-size   "$INFER_TP" \
        --gpu-memory-utilization 0.85 \
        --max-model-len          8192 \
        --max-new-tokens         8192 \
        --temperature            0.3 \
        --lora-rank              16 \
        --system-prompt          "$SYSTEM_PROMPT" \
        --tokenizer-mode         "$TOKENIZER_MODE" \
        $THINKING_FLAG
    echo ">>> Stage 2 complete — $(wc -l < "$LARGE_INFER") rows"
    sleep 20
fi
fi  # end SKIP_INFER
echo ""

# ── Stage 3: Merge into single comparison JSONL ───────────────────────────────
echo ">>> STAGE 3: Merging outputs"
"$PYTHON" -c "
import json
self_path  = '$SELF_INFER'
large_path = '$LARGE_INFER'
out_path   = '$MERGED'
with open(self_path)  as f: self_rows  = {str(json.loads(l)['p1_id']): json.loads(l) for l in f if l.strip()}
with open(large_path) as f: large_rows = {str(json.loads(l)['p1_id']): json.loads(l) for l in f if l.strip()}
with open(out_path, 'w') as fout:
    for pid, sr in self_rows.items():
        lr = large_rows.get(pid, {})
        row = {'p1_id': pid, 'p1': sr.get('p1',''), 'category': sr.get('category',''), 'level': sr.get('level'),
               'self_output': sr.get('self_output',''), 'large_model_output': lr.get('large_model_output','')}
        fout.write(json.dumps(row, ensure_ascii=False) + '\n')
print(f'Merged {len(self_rows)} rows -> {out_path}')
"
echo ">>> Stage 3 complete"
echo ""

sleep 20

# ── Stage 4: Judge — large_model vs self_model (direct comparison) ────────────
# large_model.jsonl has 'response' → judge reads it as sft-responses
# self_model.jsonl  has 'baseline_output' → judge reads it as baseline
# Summary: "SFT Win%" = large_model preferred; "Baseline Win%" = self_model preferred
echo ">>> STAGE 4: Judge (large_model vs self_model)"
_SLURM_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
[[ -n "$_SLURM_CUDA_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA_DEVICES"
export VLLM_USE_V1=0
JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENFORCE_EAGER=0
echo "Judge TP: $JUDGE_TP"

"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$LARGE_INFER" \
    --baseline       "$SELF_INFER" \
    --output-dir     "$JUDGE_DIR" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "${MODEL}" \
    --sft-label      large_model \
    --baseline-label self_model \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0
echo ">>> Stage 4 complete"
echo ""

echo "========================================================"
echo "Comparison pipeline done: $MODEL"
echo "========================================================"
