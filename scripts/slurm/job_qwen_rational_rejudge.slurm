#!/usr/bin/bash
# Re-run Gemma-4 judge for qwen35_27b and qwen35_9b rational_sft.
# Previous runs used LS p1_baseline_outputs.jsonl for ALL rows, causing wrong
# baseline responses for ~15-20% of DNA rows (p1_id collision between datasets).
# This uses a merged baseline keyed by p1 text instead of p1_id.
#
# Usage:
#   sbatch scripts/slurm/job_qwen_rational_rejudge.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=8:00:00
#SBATCH --job-name=qwen_rational_rejudge
#SBATCH --output=$PROJECT_ROOT/logs/qwen_rational_rejudge_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/qwen_rational_rejudge_%j.err

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

# ── 1. qwen35_27b rational_sft judge ─────────────────────────────────────────
echo ">>> [1/2] Judge: rational_sft/qwen35_27b"
"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$ROOT/output/SFT/rational_sft/qwen35_27b/inference/qwen35_27b.jsonl" \
    --baseline       "$ROOT/output/SFT/self_model/qwen35_27b_thinking/modelwise_test_baselines/qwen35_27b_merged_baseline.jsonl" \
    --output-dir     "$ROOT/output/SFT/rational_sft/qwen35_27b/judge/qwen35_27b" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "qwen35_27b" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0
echo ">>> [1/2] Done — $(wc -l < "$ROOT/output/SFT/rational_sft/qwen35_27b/judge/qwen35_27b/sft_vs_baseline_qwen35_27b.jsonl") rows"
echo ""

sleep 15

# ── 2. qwen35_9b rational_sft judge ──────────────────────────────────────────
echo ">>> [2/2] Judge: rational_sft/qwen35_9b"
"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$ROOT/output/SFT/rational_sft/qwen35_9b/inference/qwen35_9b.jsonl" \
    --baseline       "$ROOT/output/SFT/self_model/qwen35_9b_thinking/modelwise_test_baselines/qwen35_9b_merged_baseline.jsonl" \
    --output-dir     "$ROOT/output/SFT/rational_sft/qwen35_9b/judge/qwen35_9b" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "qwen35_9b_rational_sft" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0
echo ">>> [2/2] Done — $(wc -l < "$ROOT/output/SFT/rational_sft/qwen35_9b/judge/qwen35_9b/sft_vs_baseline_qwen35_9b_rational_sft.jsonl") rows"
echo ""

echo "========================================================"
echo "qwen rational_sft rejudge complete at: $(date)"
echo "========================================================"
