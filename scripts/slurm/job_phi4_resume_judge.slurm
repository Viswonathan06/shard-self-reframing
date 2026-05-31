#!/usr/bin/bash
# Run the Gemma-4 judge for phi4 self_model and rational_sft (full 396 rows).
# Previous runs only matched 103 rows because modelwise_test_baselines/phi4.jsonl
# had wrong coverage; correct baseline is Linguasafe_Experiments p1_baseline_outputs.jsonl.
#
# Usage:
#   sbatch scripts/slurm/job_phi4_resume_judge.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=6:00:00
#SBATCH --job-name=phi4_resume_judge
#SBATCH --output=$PROJECT_ROOT/logs/phi4_resume_judge_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/phi4_resume_judge_%j.err

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
export VLLM_USE_V1=0
export VLLM_TRUST_REMOTE_CODE=1

echo "=== JOB START ==="
echo "Host:     $(hostname)"
echo "Date:     $(date)"
echo "Job ID:   ${SLURM_JOB_ID:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "=== END DIAGNOSTICS ==="

# ── Gemma-4 path ──────────────────────────────────────────────────────────────
_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma-4 not found at $_GEMMA4_BASE"; exit 1; }
echo "Using Gemma-4 at: $GEMMA4_PATH"

_SLURM_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
[[ -n "$_SLURM_CUDA_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA_DEVICES"

JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c . || echo 0)
_tp=1; for _p2 in 2 4 8; do (( _p2 <= JUDGE_TP )) && _tp=$_p2; done; JUDGE_TP=$_tp
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENFORCE_EAGER=0

echo "Judge TP: $JUDGE_TP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── 1. self_model/phi4 judge (resume) ─────────────────────────────────────────
echo ">>> [1/2] Resuming judge: self_model/phi4"
SELF_INFER="$ROOT/output/SFT/self_model/phi4/inference/phi4.jsonl"
SELF_BASELINE="$ROOT/output/SFT/self_model/phi4/modelwise_test_baselines/phi4_merged_baseline.jsonl"
SELF_JUDGE_DIR="$ROOT/output/SFT/self_model/phi4/judge/phi4"

[[ -f "$SELF_INFER"    ]] || { echo "ERROR: missing $SELF_INFER"; exit 1; }
[[ -f "$SELF_BASELINE" ]] || { echo "ERROR: missing $SELF_BASELINE"; exit 1; }

"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$SELF_INFER" \
    --baseline       "$SELF_BASELINE" \
    --output-dir     "$SELF_JUDGE_DIR" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "phi4" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0

echo ">>> [1/2] Done — $(wc -l < "$SELF_JUDGE_DIR/sft_vs_baseline_phi4.jsonl") rows total"
echo ""

sleep 15

# ── 2. rational_sft/phi4 judge (resume) ───────────────────────────────────────
echo ">>> [2/2] Resuming judge: rational_sft/phi4"
RATIONAL_INFER="$ROOT/output/SFT/rational_sft/phi4/inference/phi4.jsonl"
RATIONAL_BASELINE="$ROOT/output/SFT/self_model/phi4/modelwise_test_baselines/phi4_merged_baseline.jsonl"
RATIONAL_JUDGE_DIR="$ROOT/output/SFT/rational_sft/phi4/judge/phi4"

[[ -f "$RATIONAL_INFER"    ]] || { echo "ERROR: missing $RATIONAL_INFER"; exit 1; }
[[ -f "$RATIONAL_BASELINE" ]] || { echo "ERROR: missing $RATIONAL_BASELINE"; exit 1; }

"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$RATIONAL_INFER" \
    --baseline       "$RATIONAL_BASELINE" \
    --output-dir     "$RATIONAL_JUDGE_DIR" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "phi4_rational_sft" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0

echo ">>> [2/2] Done — $(wc -l < "$RATIONAL_JUDGE_DIR/sft_vs_baseline_phi4_rational_sft.jsonl") rows total"
echo ""

echo "========================================================"
echo "phi4 resume judge complete at: $(date)"
echo "========================================================"
