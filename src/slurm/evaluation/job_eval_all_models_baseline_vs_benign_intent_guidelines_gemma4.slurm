#!/usr/bin/bash
# Evaluation Job 2: DNA Baseline vs Benign Intent + Guidelines - ALL 6 MODELS
# Processes all 6 model comparisons sequentially in one job using Gemma 4 31B as judge
#
# Usage: sbatch src/slurm/evaluation/job_eval_all_models_baseline_vs_benign_intent_guidelines_gemma4.slurm
#
# This job processes:
#   llama8b, llama33_70b, mistral_7b, mistral_24b, qwen35_9b, qwen35_27b
#
# Each comparison: dna_baseline_multimodel vs dna_benign_intent_guidelines_multimodel
#
# Override settings:
#   MAX_RECORDS=100 sbatch ...         # Limit records for testing
#   EVALS=helpfulness sbatch ...       # Only helpfulness evaluation

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=eval_all_models_baseline_vs_benign_intent_guidelines_gemma4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Repo root detection
ROOT_DIR="${ROOT_DIR:-}"
if [[ -z "$ROOT_DIR" ]] && [[ -d "${SLURM_SUBMIT_DIR:-}/src/evaluation_ranking" ]]; then
  ROOT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
fi
if [[ -z "$ROOT_DIR" ]]; then
  _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  _ROOT_CAND="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
  if [[ -d "${_ROOT_CAND}/src/evaluation_ranking" ]]; then
    ROOT_DIR="${_ROOT_CAND}"
  fi
fi
if [[ -z "$ROOT_DIR" ]] || [[ ! -d "$ROOT_DIR/src/evaluation_ranking" ]]; then
  echo "ERROR: Could not resolve repo root. Submit from repo or set ROOT_DIR."
  exit 1
fi

cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "=== DNA Evaluation Job 2: Baseline vs Benign Intent + Guidelines - ALL MODELS ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Root: ${ROOT_DIR}"
echo "Start time: $(date)"

# All models to process
MODELS=("llama8b" "llama33_70b" "mistral_7b" "mistral_24b" "qwen35_9b" "qwen35_27b")

# Parameters
EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

echo "Models to process: ${MODELS[*]}"
echo "Evaluations: ${EVALS}"
echo "Max records per model: ${MAX_RECORDS:-all}"

# Setup Gemma 4 31B as judge
echo ""
echo "=== Setting up Gemma 4 31B Judge ==="

# HF cache setup
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d /playpen/huggingface/hub ]] && [[ -w /playpen/huggingface/hub ]]; then
    export HF_HOME=/playpen/huggingface
  else
    PLAYPEN_USER_HF="${PLAYPEN_USER_HF:-/playpen/${USER}/huggingface}"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" 2>/dev/null && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
    else
      export HF_HOME="${HOME}/.cache/huggingface"
    fi
  fi
fi

# Gemma 4 31B setup
export USE_LOCAL_MODEL=true
export JUDGE_MODEL="google/gemma-4-31B-it"
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.70    # Conservative for 31B model
export VLLM_MAX_MODEL_LEN=4096             # Context length
export VLLM_ENFORCE_EAGER=1                # Stability

# Try to find cached model first
GEMMA4_SHARED_PATH="/playpen/huggingface/hub/models--google--gemma-4-31B-it"
GEMMA4_USER_PATH="${HF_HOME}/hub/models--google--gemma-4-31B-it"

resolve_gemma4_path() {
  # Check shared cache first
  if [[ -d "${GEMMA4_SHARED_PATH}/snapshots" ]]; then
    local shared_snap=$(find "${GEMMA4_SHARED_PATH}/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${shared_snap}" ]] && [[ -f "${shared_snap}/config.json" ]]; then
      echo "${shared_snap}"
      return 0
    fi
  fi
  
  # Check user cache
  if [[ -d "${GEMMA4_USER_PATH}/snapshots" ]]; then
    local user_snap=$(find "${GEMMA4_USER_PATH}/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${user_snap}" ]] && [[ -f "${user_snap}/config.json" ]]; then
      echo "${user_snap}"
      return 0
    fi
  fi
  
  echo ""
  return 1
}

GEMMA4_MODEL_PATH=$(resolve_gemma4_path)
if [[ -n "${GEMMA4_MODEL_PATH}" ]]; then
  export LOCAL_MODEL="${GEMMA4_MODEL_PATH}"
  echo "✓ Using cached Gemma 4 31B: ${GEMMA4_MODEL_PATH}"
else
  export LOCAL_MODEL="google/gemma-4-31B-it"
  echo "⚠ Using repo ID (will download): google/gemma-4-31B-it"
fi

# Python setup
if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
  echo "✓ Using project .venv"
else
  PYTHON_CMD="python3"
  echo "✓ Using system python"
fi

# GPU diagnostics
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

# Process each model
declare -a SUCCESSFUL_MODELS
declare -a FAILED_MODELS
TOTAL_MODELS=${#MODELS[@]}
CURRENT_MODEL=0

for model in "${MODELS[@]}"; do
  CURRENT_MODEL=$((CURRENT_MODEL + 1))
  
  echo ""
  echo "========================================"
  echo "Processing Model $CURRENT_MODEL/$TOTAL_MODELS: $model"
  echo "========================================"
  echo "Start time: $(date)"
  
  # Input files
  O1_BASELINE="${ROOT_DIR}/output/DNA_New_Experiments/dna_baseline_multimodel/${model}/p1_baseline_outputs.jsonl"
  O2_BENIGN_GUIDELINES="${ROOT_DIR}/output/DNA_New_Experiments/dna_benign_intent_guidelines_multimodel/${model}/p1_baseline_outputs.jsonl"

  # Output directory
  EVAL_OUT_DIR="${ROOT_DIR}/output/DNA_New_Experiments/evaluation/baseline_vs_benign_intent_guidelines_${model}_gemma4_judge"

  # Validate input files
  if [[ ! -f "${O1_BASELINE}" ]]; then
    echo "❌ ERROR: Baseline file not found: ${O1_BASELINE}"
    FAILED_MODELS+=("${model}")
    continue
  fi
  if [[ ! -f "${O2_BENIGN_GUIDELINES}" ]]; then
    echo "❌ ERROR: Benign intent + guidelines file not found: ${O2_BENIGN_GUIDELINES}"
    FAILED_MODELS+=("${model}")
    continue
  fi

  echo "Evaluating:"
  echo "  Baseline (O1):               ${O1_BASELINE}"
  echo "  Benign Intent + Guidelines (O2): ${O2_BENIGN_GUIDELINES}"
  echo "  Output dir:                  ${EVAL_OUT_DIR}"

  # Create output directory
  mkdir -p "${EVAL_OUT_DIR}"

  # Build evaluation command
  CMD_ARGS=(
    "--o1-jsonl" "${O1_BASELINE}"
    "--o2-jsonl" "${O2_BENIGN_GUIDELINES}"
    "--output-dir" "${EVAL_OUT_DIR}"
    "--evals" "${EVALS}"
    "--temperature" "${TEMPERATURE}"
    "--max-tokens" "${MAX_TOKENS}"
    "--use-local-model"
    "--judge-model" "${LOCAL_MODEL}"
  )

  if [[ -n "${MAX_RECORDS}" ]]; then
    CMD_ARGS+=("--max-records" "${MAX_RECORDS}")
  fi

  # Execute evaluation
  echo ""
  echo "Running evaluation for $model..."
  echo "Command: ${PYTHON_CMD} src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py ${CMD_ARGS[*]}"
  
  if "${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"; then
    echo "✅ SUCCESS: $model evaluation completed"
    SUCCESSFUL_MODELS+=("${model}")
    
    # Show summary if available
    summary_file="${EVAL_OUT_DIR}/eval_summary.json"
    if [[ -f "${summary_file}" ]]; then
      echo "Summary for $model:"
      head -20 "$summary_file" | grep -E "(helpfulness|harmlessness|total)" || true
    fi
  else
    echo "❌ FAILED: $model evaluation failed"
    FAILED_MODELS+=("${model}")
  fi
  
  echo "Completed $model at: $(date)"
  echo ""
done

# Final summary
echo ""
echo "========================================"
echo "JOB COMPLETION SUMMARY"
echo "========================================"
echo "End time: $(date)"
echo "Total models processed: $TOTAL_MODELS"
echo "Successful: ${#SUCCESSFUL_MODELS[@]}"
echo "Failed: ${#FAILED_MODELS[@]}"

if [[ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]]; then
  echo ""
  echo "✅ Successfully processed models:"
  printf '  %s\n' "${SUCCESSFUL_MODELS[@]}"
  echo ""
  echo "Results available in:"
  for model in "${SUCCESSFUL_MODELS[@]}"; do
    echo "  output/DNA_New_Experiments/evaluation/baseline_vs_benign_intent_guidelines_${model}_gemma4_judge/"
  done
fi

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo ""
  echo "❌ Failed models:"
  printf '  %s\n' "${FAILED_MODELS[@]}"
  echo ""
  echo "To retry failed models individually:"
  for model in "${FAILED_MODELS[@]}"; do
    echo "  MODEL=${model} sbatch src/slurm/evaluation/job_eval_baseline_vs_benign_intent_guidelines_gemma4.slurm"
  done
fi

echo ""
echo "Job completed with ${#SUCCESSFUL_MODELS[@]}/${TOTAL_MODELS} models successful"

# Exit with error code if any models failed
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  exit 1
else
  exit 0
fi