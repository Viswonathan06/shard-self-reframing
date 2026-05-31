#!/usr/bin/bash
# Judge-only: run Gemma-4 judge on already-completed refinement outputs.
# Usage: MODEL=phi4 sbatch scripts/slurm/job_universal_only_judge_only.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=12:00:00
#SBATCH --job-name=judge_only_%x
#SBATCH --output=$PROJECT_ROOT/logs/judge_only_%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/judge_only_%x_%j.err

set -eo pipefail

MODEL="${MODEL:?Set MODEL before sbatch}"

ROOT="$PROJECT_ROOT"
cd "$ROOT"

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB="${HF_HOME}/hub"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_AIO=0
export DS_SKIP_CUDA_CHECK=1
export TMPDIR=/tmp
export VLLM_USE_V1=0

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

GEMMA4_PATH=$(find "$HF_HOME/models--google--gemma-4-31B-it/snapshots" \
    -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "${GEMMA4_PATH}/config.json" ]] || { echo "ERROR: Gemma-4 not found"; exit 1; }

DNA_REFINE_DIR="output/DNA_Per_Model_Experiments/refinement_without_cat/${MODEL}"
LS_REFINE_DIR="output/Linguasafe_Experiments/refinement_without_cat/${MODEL}"
DNA_JUDGE_DIR="output/DNA_Per_Model_Experiments/evaluation/without_cat_${MODEL}_gemma4_judge"
LS_JUDGE_DIR="output/Linguasafe_Experiments/evaluation/without_cat_${MODEL}_gemma4_judge"
mkdir -p "$DNA_JUDGE_DIR" "$LS_JUDGE_DIR"

source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
JUDGE_TP=$NUM_GPUS
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=6144
export VLLM_USE_V1=0

echo "========================================================"
echo "Judge-only: $MODEL  |  Gemma-4 TP=$JUDGE_TP"
echo "Node: $(hostname)  GPUs: $NUM_GPUS  Start: $(date)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_judge() {
    local label="$1" refine_dir="$2" judge_dir="$3"
    local refined_file="${refine_dir}/baseline_outputs.jsonl"
    echo ">>> Judge — $label"
    if [[ ! -f "$refined_file" ]]; then echo "  WARN: $refined_file not found — skipping"; return; fi
    local n_refined; n_refined=$(wc -l < "$refined_file")
    local eval_file="${judge_dir}/eval_results.jsonl"
    if [[ -f "$eval_file" ]] && [[ $(wc -l < "$eval_file") -ge "$n_refined" ]]; then
        echo "  SKIP: already done ($(wc -l < "$eval_file") rows)"; return
    fi
    "$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
        --refinement-jsonl  "$refined_file" \
        --output-dir        "$judge_dir" \
        --evals             all \
        --temperature       0.1 \
        --max-tokens        512 \
        --use-local-model \
        --judge-model       "$GEMMA4_PATH"
    echo "  Done: $judge_dir"
}

run_judge "DNA"        "$DNA_REFINE_DIR" "$DNA_JUDGE_DIR"
run_judge "LinguaSafe" "$LS_REFINE_DIR"  "$LS_JUDGE_DIR"

echo "========================================================"
echo "Judge-only done: $MODEL  $(date)"
echo "========================================================"
