#!/usr/bin/bash
# Re-judge Phi-4 DNA refinement with properly formatted baseline (o1).
#
# Background: the original pipeline ran norm() on baseline_output, stripping
# all newlines. This job judges against baseline_outputs_fixed.jsonl, which
# restores newlines from the raw p1_baseline_outputs.jsonl.
#
# LinguaSafe already re-judged separately.
# Output: DNA_Per_Model_Experiments/evaluation/refinement_from_baseline_multimodel_phi4_fixed_gemma4_judge/

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=08:00:00
#SBATCH --job-name=phi4_rejudge
#SBATCH --output=$PROJECT_ROOT/logs/phi4_rejudge_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/phi4_rejudge_%j.err

set -euo pipefail

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0

FIXED_JSONL="output/DNA_Per_Model_Experiments/refinement_from_baseline_multimodel/phi4/baseline_outputs_fixed.jsonl"
JUDGE_DIR="output/DNA_Per_Model_Experiments/evaluation/refinement_from_baseline_multimodel_phi4_fixed_gemma4_judge"

[[ -f "$FIXED_JSONL" ]] || { echo "ERROR: fixed jsonl not found: $FIXED_JSONL"; exit 1; }
mkdir -p "$JUDGE_DIR"

# ── Gemma-4 path (shared cache) ───────────────────────────────────────────────
_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma4 not found in $_GEMMA4_BASE"; exit 1; }

source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true

JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENFORCE_EAGER=0

echo "========================================================"
echo "Phi-4 re-judge with fixed baseline formatting"
echo "Job:        ${SLURM_JOB_ID}"
echo "Input:      $FIXED_JSONL  ($(wc -l < "$FIXED_JSONL") rows)"
echo "Output dir: $JUDGE_DIR"
echo "Judge:      $GEMMA4_PATH"
echo "Judge TP:   $JUDGE_TP"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

"$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
    --refinement-jsonl  "$FIXED_JSONL" \
    --output-dir        "$JUDGE_DIR" \
    --evals             all \
    --temperature       0.1 \
    --max-tokens        512 \
    --use-local-model \
    --judge-model       "$GEMMA4_PATH" \
    --batch-size        64

echo ""
echo "Done — results in $JUDGE_DIR"
echo "Rows: $(wc -l < "$JUDGE_DIR/eval_results.jsonl" 2>/dev/null || echo 0)"
