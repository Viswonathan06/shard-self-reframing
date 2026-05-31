#!/usr/bin/bash
# Re-run inference + judge for mistral7b self-model SFT (adapter already trained).
# Fixes the [INST] artifact by passing the training system prompt at inference.

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:8
#SBATCH --time=8:00:00
#SBATCH --job-name=reinfer_mistral7b
#SBATCH --output=$PROJECT_ROOT/logs/reinfer_mistral7b_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/reinfer_mistral7b_%j.err

set -euo pipefail

ROOT="$PROJECT_ROOT"
cd "$ROOT"

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_MODEL="$HF_HOME/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
SFT_ADAPTER="output/SFT/self_model/mistral7b/models/mistral7b"
TEST_JSONL="output/SFT/self_model/mistral7b/test.jsonl"
INFERENCE_OUT="output/SFT/self_model/mistral7b/inference/mistral7b.jsonl"
JUDGE_DIR="output/SFT/self_model/mistral7b/judge/mistral7b"
BASELINE="output/SFT/self_model/mistral7b/modelwise_test_baselines/mistral7b.jsonl"

# Must match the system prompt used during SFT training (sft_v2.py SYSTEM_PROMPT).
SFT_SYSTEM_PROMPT="You are a helpful assistant."

_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma4 not found"; exit 1; }

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
INFER_TP=1
for _tp in 2 4 8; do (( _tp <= NUM_GPUS )) && INFER_TP=$_tp; done

echo "Base model:  $BASE_MODEL"
echo "SFT adapter: $SFT_ADAPTER"
echo "Judge model: $GEMMA4_PATH"
echo "Inference TP: $INFER_TP"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo ">>> Inference (with system prompt to fix [INST] artifacts)"
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
    --system-prompt          "$SFT_SYSTEM_PROMPT"

echo ">>> Inference done — $(wc -l < "$INFERENCE_OUT") rows"
echo ""

echo ">>> Judge"
_SLURM_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
[[ -n "$_SLURM_CUDA_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA_DEVICES"
JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=6144
export DS_BUILD_AIO=0
export DS_SKIP_CUDA_CHECK=1

mkdir -p "$JUDGE_DIR"
"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$INFERENCE_OUT" \
    --baseline       "$BASELINE" \
    --output-dir     "$JUDGE_DIR" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "mistral7b" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0

echo ">>> Judge done — results in $JUDGE_DIR"
