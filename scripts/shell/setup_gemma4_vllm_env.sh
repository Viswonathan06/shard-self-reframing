#!/bin/bash
# Setup script for Gemma 4 31B with vLLM optimization
# Usage: source scripts/shell/setup_gemma4_vllm_env.sh

# Note: Don't use 'set -e' when sourcing as it can exit the shell
# set -euo pipefail

echo "=== Setting up Gemma 4 31B vLLM Environment ==="

# Project root
export SAFESHIFT_ROOT="/common/guam/Safeshift"
export PYTHONPATH="${SAFESHIFT_ROOT}:${PYTHONPATH:-}"

# HuggingFace cache setup
if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d /playpen/huggingface/hub ]] && [[ -w /playpen/huggingface/hub ]]; then
    export HF_HOME=/playpen/huggingface
    echo "✓ Using shared HF cache: ${HF_HOME}"
  else
    # Use user-specific playpen cache
    PLAYPEN_USER_HF="${PLAYPEN_USER_HF:-/playpen/${USER}/huggingface}"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
      echo "✓ Using user HF cache: ${HF_HOME}"
    else
      export HF_HOME="${HOME}/.cache/huggingface"
      echo "⚠ Fallback to home cache: ${HF_HOME}"
    fi
  fi
fi

# HuggingFace Token
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ -f "${HOME}/.hf_token" ]]; then
    export HF_TOKEN=$(cat "${HOME}/.hf_token")
    echo "✓ HF_TOKEN loaded from ~/.hf_token"
  else
    echo "⚠ HF_TOKEN not set - may be needed for Gemma 4 model access"
    echo "  Set with: export HF_TOKEN=hf_your_token_here"
  fi
fi

# Model paths
export GEMMA4_31B_SHARED_PATH="/playpen/huggingface/hub/models--google--gemma-4-31B-it"
export GEMMA4_31B_USER_PATH="${HF_HOME}/hub/models--google--gemma-4-31B-it"

# vLLM Configuration for Gemma 4 31B (optimized for 31B model)
export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE=4                # Use 4 GPUs for 31B model
export VLLM_GPU_MEMORY_UTILIZATION=0.75           # Reduced to avoid OOM issues
export VLLM_MAX_MODEL_LEN=6144                    # Reduced context for stability
export VLLM_MAX_NUM_BATCHED_TOKENS=6144          # Batch size optimization
export VLLM_BLOCK_SIZE=16                        # Memory block size
export VLLM_SWAP_SPACE=4                         # GB of CPU swap space
export VLLM_ENFORCE_EAGER=1                      # Use eager execution for stability

# Performance optimizations
export CUDA_VISIBLE_DEVICES=0,1,2,3              # Use first 4 GPUs
export NCCL_P2P_DISABLE=1                        # Sometimes needed for multi-GPU
export NCCL_IB_DISABLE=1                         # Disable InfiniBand if issues

# Model-specific settings
export MODEL_NAME="google/gemma-4-31B-it"
export MODEL_TEMPERATURE=0.3
export MODEL_MAX_TOKENS=2048
export MODEL_TOP_P=0.9

# Resolve model path function
resolve_gemma4_path() {
  # Check shared cache first
  if [[ -d "${GEMMA4_31B_SHARED_PATH}/snapshots" ]]; then
    local shared_snap=$(find "${GEMMA4_31B_SHARED_PATH}/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${shared_snap}" ]] && [[ -f "${shared_snap}/config.json" ]]; then
      echo "${shared_snap}"
      return 0
    fi
  fi
  
  # Check user cache
  if [[ -d "${GEMMA4_31B_USER_PATH}/snapshots" ]]; then
    local user_snap=$(find "${GEMMA4_31B_USER_PATH}/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${user_snap}" ]] && [[ -f "${user_snap}/config.json" ]]; then
      echo "${user_snap}"
      return 0
    fi
  fi
  
  echo ""
  return 1
}

# Set resolved model path
GEMMA4_MODEL_PATH=$(resolve_gemma4_path)
if [[ -n "${GEMMA4_MODEL_PATH}" ]]; then
  export GEMMA4_31B_PATH="${GEMMA4_MODEL_PATH}"
  echo "✓ Gemma 4 31B found at: ${GEMMA4_31B_PATH}"
else
  echo "❌ Gemma 4 31B model not found in cache"
  echo "   Check: ls -la ${GEMMA4_31B_SHARED_PATH}/snapshots/"
  echo "   Continuing anyway - you may need to download the model first"
  export GEMMA4_31B_PATH="${GEMMA4_31B_SHARED_PATH}"
fi

# Environment check
echo ""
echo "=== Environment Summary ==="
echo "Project root: ${SAFESHIFT_ROOT}"
echo "HF_HOME: ${HF_HOME}"
echo "Model path: ${GEMMA4_31B_PATH}"
echo "vLLM config: TP=${VLLM_TENSOR_PARALLEL_SIZE} GPU_MEM=${VLLM_GPU_MEMORY_UTILIZATION} MAX_LEN=${VLLM_MAX_MODEL_LEN}"
echo ""
echo "Ready to run Gemma 4 31B with vLLM! 🚀"
echo ""
echo "Next steps:"
echo "  1. Run a test: srun --gres=gpu:4 python -c \"from src.utils.model_client import ModelClient; print('vLLM ready!')\""
echo "  2. Submit job: sbatch scripts/slurm/job_gemma4_31b_test.slurm"
echo "  3. Interactive: srun --gres=gpu:4 --pty bash"