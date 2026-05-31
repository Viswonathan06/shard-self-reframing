#!/usr/bin/bash
# LinguaSafe Evaluation: helpful_assistant (O1) vs refinement_from_baseline (O2) - ALL 6 MODELS
# Judge: Gemma 4 31B
#
# Each model is compared against its own outputs:
#   O1 = baseline_output field in linguasafe_refinement_from_baseline_multimodel/<model>/baseline_outputs.jsonl
#   O2 = output field in the same file
#
# Usage: sbatch src/slurm/evaluation/job_eval_linguasafe_baseline_vs_refinement_gemma4.slurm
#
# Overrides:
#   MAX_RECORDS=100 sbatch ...    # Limit records for testing
#   EVALS=helpfulness sbatch ...  # Only run one eval dimension

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=linguasafe_eval_baseline_vs_refinement_gemma4
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

echo "=== LinguaSafe Evaluation: Baseline vs Refinement - ALL MODELS ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Root: ${ROOT_DIR}"
echo "Start time: $(date)"

MODELS=("llama8b" "llama33_70b" "mistral_7b" "mistral_24b" "qwen35_9b" "qwen35_27b")

EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

echo "Models: ${MODELS[*]}"
echo "Evaluations: ${EVALS}"
echo "Max records: ${MAX_RECORDS:-all}"

# ── HF cache ────────────────────────────────────────────────────────────────
if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
  else
    PLAYPEN_USER_HF="/playpen/${USER}/huggingface"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" 2>/dev/null && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
    else
      export HF_HOME="${HOME}/.cache/huggingface"
    fi
  fi
fi

# ── Gemma 4 31B judge setup ──────────────────────────────────────────────────
export USE_LOCAL_MODEL=true
export JUDGE_MODEL="google/gemma-4-31B-it"
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_MAX_MODEL_LEN=4096
export VLLM_ENFORCE_EAGER=1

GEMMA4_SHARED_PATH="$HF_HOME/models--google--gemma-4-31B-it"
GEMMA4_USER_PATH="${HF_HOME}/hub/models--google--gemma-4-31B-it"

resolve_gemma4_path() {
  for base in "${GEMMA4_SHARED_PATH}" "${GEMMA4_USER_PATH}"; do
    if [[ -d "${base}/snapshots" ]]; then
      local snap
      snap=$(find "${base}/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
      if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then
        echo "${snap}"; return 0
      fi
    fi
  done
  echo ""; return 1
}

GEMMA4_MODEL_PATH=$(resolve_gemma4_path) || true
if [[ -n "${GEMMA4_MODEL_PATH}" ]]; then
  export LOCAL_MODEL="${GEMMA4_MODEL_PATH}"
  echo "Using cached Gemma 4 31B: ${GEMMA4_MODEL_PATH}"
else
  export LOCAL_MODEL="google/gemma-4-31B-it"
  echo "Gemma 4 31B not cached — will download from HuggingFace Hub"
fi

# ── Python ───────────────────────────────────────────────────────────────────
if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

LINGUASAFE_EXPERIMENTS_ROOT="${ROOT_DIR}/output/Linguasafe_Experiments"
EVAL_ROOT="${LINGUASAFE_EXPERIMENTS_ROOT}/evaluation"
mkdir -p "${EVAL_ROOT}"

declare -a SUCCESSFUL_MODELS
declare -a FAILED_MODELS
TOTAL_MODELS=${#MODELS[@]}
CURRENT_MODEL=0

for model in "${MODELS[@]}"; do
  CURRENT_MODEL=$((CURRENT_MODEL + 1))

  echo ""
  echo "========================================"
  echo "Model $CURRENT_MODEL/$TOTAL_MODELS: $model"
  echo "Start: $(date)"
  echo "========================================"

  # The refinement file contains both baseline_output (O1) and output (O2)
  # for this specific model — judge compares each model against its own outputs.
  REFINEMENT_FILE="${LINGUASAFE_EXPERIMENTS_ROOT}/linguasafe_refinement_from_baseline_multimodel/${model}/baseline_outputs.jsonl"
  EVAL_OUT_DIR="${EVAL_ROOT}/baseline_vs_refinement_${model}_gemma4_judge"

  if [[ ! -f "${REFINEMENT_FILE}" ]]; then
    echo "ERROR: Refinement file not found: ${REFINEMENT_FILE}"
    FAILED_MODELS+=("${model}")
    continue
  fi

  echo "  O1 (baseline_output) + O2 (output): ${REFINEMENT_FILE}"
  echo "  Output dir: ${EVAL_OUT_DIR}"

  mkdir -p "${EVAL_OUT_DIR}"

  CMD_ARGS=(
    "--refinement-jsonl" "${REFINEMENT_FILE}"
    "--output-dir"       "${EVAL_OUT_DIR}"
    "--evals"            "${EVALS}"
    "--temperature"      "${TEMPERATURE}"
    "--max-tokens"       "${MAX_TOKENS}"
    "--use-local-model"
    "--judge-model"      "${LOCAL_MODEL}"
  )
  [[ -n "${MAX_RECORDS}" ]] && CMD_ARGS+=("--max-records" "${MAX_RECORDS}")

  if "${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"; then
    echo "SUCCESS: ${model}"
    SUCCESSFUL_MODELS+=("${model}")
    summary="${EVAL_OUT_DIR}/eval_summary.json"
    if [[ -f "${summary}" ]]; then
      python3 -c "
import json
d = json.load(open('${summary}'))
for k in ('helpfulness','harmlessness'):
    if k in d:
        v = d[k]
        print(f'  {k}: o1_wins={v.get(\"o1_wins\",\"?\")}, o2_wins={v.get(\"o2_wins\",\"?\")}, ties={v.get(\"ties\",\"?\")}')
" 2>/dev/null || true
    fi
  else
    echo "FAILED: ${model}"
    FAILED_MODELS+=("${model}")
  fi

  echo "Done: ${model} at $(date)"
done

echo ""
echo "========================================"
echo "SUMMARY: ${#SUCCESSFUL_MODELS[@]}/${TOTAL_MODELS} models successful"
echo "End: $(date)"
[[ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]] && printf '  OK:   %s\n' "${SUCCESSFUL_MODELS[@]}"
[[ ${#FAILED_MODELS[@]} -gt 0 ]]    && printf '  FAIL: %s\n' "${FAILED_MODELS[@]}"

echo ""
echo "Results: ${EVAL_ROOT}/baseline_vs_refinement_<model>_gemma4_judge/"

[[ ${#FAILED_MODELS[@]} -gt 0 ]] && exit 1 || exit 0
