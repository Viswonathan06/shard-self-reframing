#!/usr/bin/bash
# Judge TeacherSFT (rational_sft) vs SelfSFT (SHARD) on the overlapping
# prompts shared between the qwen35_122b_teacher test set and each model's
# self_model test set.  Runs all 6 models sequentially on one GPU node.
#
# Usage:
#   sbatch scripts/slurm/job_teacher_vs_selfsft_overlap_judge.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu2,gpu3
#SBATCH --time=08:00:00
#SBATCH --job-name=teacher_vs_selfsft_overlap
#SBATCH --output=$PROJECT_ROOT/logs/teacher_vs_selfsft_overlap_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/teacher_vs_selfsft_overlap_%j.err

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
export TMPDIR=/tmp
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export VLLM_USE_V1=0
export VLLM_TRUST_REMOTE_CODE=1

_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma-4 not found"; exit 1; }

_SLURM_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
[[ -n "$_SLURM_CUDA_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA_DEVICES"

JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
_tp=1; for _p2 in 2 4 8; do (( _p2 <= JUDGE_TP )) && _tp=$_p2; done; JUDGE_TP=$_tp
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENFORCE_EAGER=0

OVERLAP_DIR="$ROOT/output/SFT/overlap_judge"

echo "========================================================"
echo "TeacherSFT vs SelfSFT overlap judge"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Judge TP:  $JUDGE_TP"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

MODELS="llama8b phi4"

for MODEL in $MODELS; do
    TEACHER_FILE="$OVERLAP_DIR/$MODEL/teacher_sft.jsonl"
    SELF_FILE="$OVERLAP_DIR/$MODEL/self_sft.jsonl"
    JUDGE_OUT="$OVERLAP_DIR/$MODEL/judge"

    if [[ ! -f "$TEACHER_FILE" || ! -f "$SELF_FILE" ]]; then
        echo ">>> SKIP $MODEL: overlap files not found"
        continue
    fi

    mkdir -p "$JUDGE_OUT"
    N=$(wc -l < "$TEACHER_FILE")
    echo ">>> Judging $MODEL ($N rows): TeacherSFT vs SelfSFT"

    "$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
        --sft-responses   "$TEACHER_FILE" \
        --baseline        "$SELF_FILE" \
        --output-dir      "$JUDGE_OUT" \
        --judge-model     "$GEMMA4_PATH" \
        --model-tag       "${MODEL}_teacher_vs_selfsft" \
        --sft-label       "TeacherSFT" \
        --baseline-label  "SelfSFT" \
        --batch-size      8 \
        --max-new-tokens  512 \
        --temperature     0.0

    OUTFILE="$JUDGE_OUT/sft_vs_baseline_${MODEL}_teacher_vs_selfsft.jsonl"
    echo ">>> $MODEL done — $(wc -l < "$OUTFILE") rows"
    echo ""
done

echo "========================================================"
echo "All models done at: $(date)"
echo "========================================================"
