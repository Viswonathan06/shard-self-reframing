#!/usr/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --job-name=judge_qwen27b
#SBATCH --output=$PROJECT_ROOT/logs/judge_qwen27b_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/judge_qwen27b_%j.err

set -euo pipefail
ROOT="$PROJECT_ROOT"
cd "$ROOT"
PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export HF_HOME=/playpen/huggingface
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1

_SLURM_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
[[ -n "$_SLURM_CUDA_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA_DEVICES"

export VLLM_USE_V1=0

GEMMA4_PATH=$(find $HF_HOME/models--google--gemma-4-31B-it/snapshots -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma4 not found"; exit 1; }
echo "Judge model: $GEMMA4_PATH"

JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENFORCE_EAGER=0
echo "Judge TP: $JUDGE_TP"

"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  output/SFT/self_model/qwen35_27b_thinking/inference/qwen35_27b_thinking.jsonl \
    --baseline       output/SFT/self_model/qwen35_27b_thinking/modelwise_test_baselines/qwen35_27b_thinking.jsonl \
    --output-dir     output/SFT/self_model/qwen35_27b_thinking/judge/qwen35_27b_thinking \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      qwen35_27b_thinking \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0

echo "Done."
