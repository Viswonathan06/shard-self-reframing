#!/usr/bin/bash
# Evaluation: DNA Baseline vs DNA Benign Intent + Guidelines using Gemma 4 31B as judge  
# Compares pure baseline (O1) vs baseline + benign intent + guidelines (O1)
#
# Usage: sbatch src/slurm/evaluation/job_eval_baseline_vs_benign_intent_guidelines_gemma4.slurm
#
# Override model:
#   MODEL=llama8b sbatch ...           # Default: llama8b
#   MODEL=qwen35_27b sbatch ...
#   MODEL=mistral_7b sbatch ...
#
# Override evaluation types:
#   EVALS=helpfulness,harmfulness sbatch ...  # Default: all
#
# Override record limit:
#   MAX_RECORDS=100 sbatch ...         # Default: all records

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_baseline_vs_benign_intent_guidelines_gemma4
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

echo "=== DNA Baseline vs Benign Intent + Guidelines Evaluation (Gemma 4 31B Judge) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Root: ${ROOT_DIR}"

# Parameters
MODEL="${MODEL:-llama8b}"
EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

# Input files
O1_BASELINE="${ROOT_DIR}/output/DNA_New_Experiments/dna_baseline_multimodel/${MODEL}/p1_baseline_outputs.jsonl"
O2_BENIGN_GUIDELINES="${ROOT_DIR}/output/DNA_New_Experiments/dna_benign_intent_guidelines_multimodel/${MODEL}/p1_baseline_outputs.jsonl"

# Output directory
EVAL_OUT_DIR="${ROOT_DIR}/output/DNA_New_Experiments/evaluation/baseline_vs_benign_intent_guidelines_${MODEL}_gemma4_judge"

# Validate input files
if [[ ! -f "${O1_BASELINE}" ]]; then
  echo "ERROR: Baseline file not found: ${O1_BASELINE}"
  exit 1
fi
if [[ ! -f "${O2_BENIGN_GUIDELINES}" ]]; then
  echo "ERROR: Benign intent + guidelines file not found: ${O2_BENIGN_GUIDELINES}"
  exit 1
fi

echo "Evaluating:"
echo "  Baseline (O1):               ${O1_BASELINE}"
echo "  Benign Intent + Guidelines (O2): ${O2_BENIGN_GUIDELINES}"
echo "  Output dir:                  ${EVAL_OUT_DIR}"
echo "  Model:                       ${MODEL}"
echo "  Evaluations:                 ${EVALS}"

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

# Log the command
echo "=== Running Evaluation ==="
echo "Command: ${PYTHON_CMD} src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py ${CMD_ARGS[*]}"
echo "Note: Using SINGLE RANDOMIZED comparison (not AB+BA position swapping)"
echo ""

# Execute evaluation (single randomized comparison)
"${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: ${EVAL_OUT_DIR}"
echo "Check the following files:"
echo "  - ${EVAL_OUT_DIR}/evaluation_results.jsonl"
echo "  - ${EVAL_OUT_DIR}/evaluation_summary.json"

exit $?