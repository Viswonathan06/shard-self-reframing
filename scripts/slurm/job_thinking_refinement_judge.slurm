#!/usr/bin/bash
# Thinking-mode refinement + Gemma4 judge for Qwen3.5 models.
# Generates O2 with <think> tokens, then judges O2 vs O1 baseline.
#
# Usage:
#   MODEL=qwen35_27b sbatch scripts/slurm/job_thinking_refinement_judge.slurm
#   MODEL=qwen35_9b  sbatch scripts/slurm/job_thinking_refinement_judge.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=think_refine
#SBATCH --output=$PROJECT_ROOT/logs/think_refine_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/think_refine_%j.err

echo "=== Thinking refinement pipeline — job ${SLURM_JOB_ID} on $(hostname) ==="
echo "MODEL=${MODEL:-UNSET}"

if [[ -z "${MODEL:-}" ]]; then
    echo "ERROR: MODEL is not set. Submit with: MODEL=qwen35_27b sbatch ..."
    exit 1
fi

set -euo pipefail

[[ "$MODEL" =~ ^(qwen35_27b|qwen35_9b)$ ]] || {
    echo "ERROR: Unknown MODEL=$MODEL. Must be qwen35_27b or qwen35_9b."
    exit 1
}

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_AIO=0
export DS_SKIP_CUDA_CHECK=1
export TMPDIR=/tmp

# ── Model paths ───────────────────────────────────────────────────────────────
declare -A BASE_MODELS=(
    [qwen35_27b]="$HF_HOME/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    [qwen35_9b]="$HF_HOME/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
BASE_MODEL="${BASE_MODELS[$MODEL]}"
[[ -f "$BASE_MODEL/config.json" ]] || { echo "ERROR: model not found: $BASE_MODEL"; exit 1; }

# ── Tensor parallel: 27B needs TP=4 to fit in bf16 across 4×49 GiB ───────────
if [[ "$MODEL" == "qwen35_27b" ]]; then
    REFINE_TP=4
else
    REFINE_TP=2
fi

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

# ── Paths ─────────────────────────────────────────────────────────────────────
DNA_INPUT="output/DNA_Per_Model_Experiments/refinement_from_baseline_multimodel/${MODEL}/refinement_input.jsonl"
LS_INPUT="output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_multimodel/${MODEL}/refinement_input.jsonl"

DNA_REFINE_OUT="output/DNA_Per_Model_Experiments/refinement_from_baseline_multimodel/${MODEL}_thinking"
LS_REFINE_OUT="output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_multimodel/${MODEL}_thinking"

DNA_JUDGE_DIR="output/DNA_Per_Model_Experiments/evaluation/refinement_from_baseline_multimodel_${MODEL}_thinking_gemma4_judge"
LS_JUDGE_DIR="output/Linguasafe_Experiments/evaluation/baseline_vs_refinement_${MODEL}_thinking_gemma4_judge"

REFINE_SYSTEM="src/prompts/p2_response_system_prompt_refinement.txt"

[[ -f "$DNA_INPUT" ]] || { echo "ERROR: DNA input not found: $DNA_INPUT"; exit 1; }
[[ -f "$LS_INPUT"  ]] || { echo "ERROR: LS input not found: $LS_INPUT"; exit 1; }

mkdir -p "$DNA_REFINE_OUT" "$LS_REFINE_OUT" "$DNA_JUDGE_DIR" "$LS_JUDGE_DIR"

# ── Gemma4 judge path ─────────────────────────────────────────────────────────
_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma4 snapshot not found"; exit 1; }

echo "========================================================"
echo "Thinking refinement pipeline: $MODEL"
echo "Job ID:       ${SLURM_JOB_ID}"
echo "Base model:   $BASE_MODEL"
echo "Refine TP:    $REFINE_TP"
echo "Judge model:  $GEMMA4_PATH"
echo "DNA input:    $(wc -l < "$DNA_INPUT") rows"
echo "LS input:     $(wc -l < "$LS_INPUT") rows"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

export USE_LOCAL_MODEL=true
export LOCAL_MODEL="$BASE_MODEL"
export VLLM_TENSOR_PARALLEL_SIZE=$REFINE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=16384

# ── Stage 1a: DNA refinement with thinking ────────────────────────────────────
echo ">>> STAGE 1a: DNA refinement (thinking mode, $(wc -l < "$DNA_INPUT") rows)"
"$PYTHON" src/baselines/generate_p2_controlled_outputs.py \
    --input-jsonl        "$DNA_INPUT" \
    --output-dir         "$DNA_REFINE_OUT" \
    --system-prompt-file "$REFINE_SYSTEM" \
    --refinement-mode \
    --enable-thinking \
    --use-local-model \
    --local-model        "$BASE_MODEL" \
    --temperature        0.7 \
    --max-tokens         8192 \
    --batch-size         16 \
    --exit-on-null
echo ">>> Stage 1a done — $(wc -l < "$DNA_REFINE_OUT/baseline_outputs.jsonl") rows"
echo ""

# ── Stage 1b: LinguaSafe refinement with thinking ─────────────────────────────
echo ">>> STAGE 1b: LinguaSafe refinement (thinking mode, $(wc -l < "$LS_INPUT") rows)"
"$PYTHON" src/baselines/generate_p2_controlled_outputs.py \
    --input-jsonl        "$LS_INPUT" \
    --output-dir         "$LS_REFINE_OUT" \
    --system-prompt-file "$REFINE_SYSTEM" \
    --refinement-mode \
    --enable-thinking \
    --use-local-model \
    --local-model        "$BASE_MODEL" \
    --temperature        0.7 \
    --max-tokens         8192 \
    --batch-size         16 \
    --exit-on-null
echo ">>> Stage 1b done — $(wc -l < "$LS_REFINE_OUT/baseline_outputs.jsonl") rows"
echo ""

# ── Stage 2: Gemma4 judge ─────────────────────────────────────────────────────
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

echo ">>> STAGE 2a: Judge — DNA"
"$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
    --refinement-jsonl   "$DNA_REFINE_OUT/baseline_outputs.jsonl" \
    --output-dir         "$DNA_JUDGE_DIR" \
    --evals              all \
    --temperature        0.1 \
    --max-tokens         512 \
    --use-local-model \
    --judge-model        "$GEMMA4_PATH" \
    --batch-size         64
echo ">>> Stage 2a done — results in $DNA_JUDGE_DIR"
echo ""

echo ">>> STAGE 2b: Judge — LinguaSafe"
"$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
    --refinement-jsonl   "$LS_REFINE_OUT/baseline_outputs.jsonl" \
    --output-dir         "$LS_JUDGE_DIR" \
    --evals              all \
    --temperature        0.1 \
    --max-tokens         512 \
    --use-local-model \
    --judge-model        "$GEMMA4_PATH" \
    --batch-size         64
echo ">>> Stage 2b done — results in $LS_JUDGE_DIR"
echo ""

echo "========================================================"
echo "Thinking refinement pipeline done: $MODEL"
echo "Next steps:"
echo "  python scripts/build_selfmodel_sft_split.py --models ${MODEL}_thinking"
echo "========================================================"
