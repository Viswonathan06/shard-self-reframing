#!/bin/bash
# Safe setup script for Gemma 4 31B with vLLM optimization
# Usage: source scripts/shell/setup_gemma4_safe.sh

# Safe error handling - don't exit the shell on errors
setup_gemma4_environment() {
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
            if mkdir -p "${PLAYPEN_USER_HF}/hub" 2>/dev/null && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
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
            export HF_TOKEN=$(cat "${HOME}/.hf_token" 2>/dev/null || echo "")
            if [[ -n "$HF_TOKEN" ]]; then
                echo "✓ HF_TOKEN loaded from ~/.hf_token"
            else
                echo "⚠ HF_TOKEN file exists but is empty"
            fi
        else
            echo "⚠ HF_TOKEN not set - may be needed for Gemma 4 model access"
            echo "  Set with: export HF_TOKEN=hf_your_token_here"
        fi
    fi
    
    # Model paths
    export GEMMA4_31B_SHARED_PATH="/playpen/huggingface/hub/models--google--gemma-4-31B-it"
    export GEMMA4_31B_USER_PATH="${HF_HOME}/hub/models--google--gemma-4-31B-it"
    
    # vLLM Configuration for Gemma 4 31B (conservative settings)
    export USE_LOCAL_MODEL=true
    export VLLM_TRUST_REMOTE_CODE=1
    export VLLM_ONLY=1
    export VLLM_TENSOR_PARALLEL_SIZE=4                # Use 4 GPUs for 31B model
    export VLLM_GPU_MEMORY_UTILIZATION=0.70           # Conservative to avoid OOM
    export VLLM_MAX_MODEL_LEN=4096                    # Conservative context for stability
    export VLLM_MAX_NUM_BATCHED_TOKENS=4096          # Batch size optimization
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
    export MODEL_MAX_TOKENS=1024
    export MODEL_TOP_P=0.9
    
    # Try to find model path
    local model_path=""
    
    # Check shared cache first
    if [[ -d "${GEMMA4_31B_SHARED_PATH}/snapshots" ]]; then
        local shared_snap=$(find "${GEMMA4_31B_SHARED_PATH}/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
        if [[ -n "${shared_snap}" ]] && [[ -f "${shared_snap}/config.json" ]]; then
            model_path="${shared_snap}"
        fi
    fi
    
    # Check user cache if not found in shared
    if [[ -z "$model_path" ]] && [[ -d "${GEMMA4_31B_USER_PATH}/snapshots" ]]; then
        local user_snap=$(find "${GEMMA4_31B_USER_PATH}/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
        if [[ -n "${user_snap}" ]] && [[ -f "${user_snap}/config.json" ]]; then
            model_path="${user_snap}"
        fi
    fi
    
    # Set model path or fallback
    if [[ -n "$model_path" ]]; then
        export GEMMA4_31B_PATH="$model_path"
        export LOCAL_MODEL="$model_path"
        echo "✓ Gemma 4 31B found at: ${GEMMA4_31B_PATH}"
    else
        echo "⚠ Gemma 4 31B model not found in snapshots"
        echo "  Checked: ${GEMMA4_31B_SHARED_PATH}/snapshots/"
        echo "  Checked: ${GEMMA4_31B_USER_PATH}/snapshots/"
        echo "  Using repo ID as fallback: google/gemma-4-31B-it"
        export GEMMA4_31B_PATH="google/gemma-4-31B-it"
        export LOCAL_MODEL="google/gemma-4-31B-it"
    fi
    
    # Environment summary
    echo ""
    echo "=== Environment Summary ==="
    echo "Project root: ${SAFESHIFT_ROOT}"
    echo "HF_HOME: ${HF_HOME}"
    echo "Model path: ${GEMMA4_31B_PATH}"
    echo "vLLM config: TP=${VLLM_TENSOR_PARALLEL_SIZE} GPU_MEM=${VLLM_GPU_MEMORY_UTILIZATION} MAX_LEN=${VLLM_MAX_MODEL_LEN}"
    echo "HF_TOKEN: ${HF_TOKEN:+Set (${#HF_TOKEN} chars)}${HF_TOKEN:-Not set}"
    echo ""
    echo "✅ Environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Check GPU memory: python scripts/check_gpu_memory.py"
    echo "  2. Test model: python scripts/test_gemma4_quick.py"
    echo "  3. Submit job: sbatch scripts/slurm/job_gemma4_31b_test.slurm"
    
    return 0
}

# Call the function
setup_gemma4_environment