#!/usr/bin/bash
# All models: Benign Intent + Guidelines vs Refinement — Gemma 4 31B judge
#
# O1 (map): dna_benign_intent_guidelines_multimodel/${model}/p1_baseline_outputs.jsonl (field: output)
# O2 (driver rows): dna_refinement_from_baseline_multimodel/${model}/baseline_outputs.jsonl (field: output)
#
# Usage:
#   sbatch src/slurm/evaluation/job_eval_all_models_benign_guidelines_vs_refinement_gemma4.slurm
#   MAX_RECORDS=100 sbatch ...
#
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=eval_all_benign_guidelines_vs_refinement_gemma4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

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
  echo "ERROR: Could not resolve repo root."
  exit 1
fi

cd "$ROOT_DIR" || exit 1
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

MODELS=("llama8b" "llama33_70b" "mistral_7b" "mistral_24b" "qwen35_9b" "qwen35_27b")
EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

echo "=== All models: benign+guidelines vs refinement (Gemma 4 31B) ==="
echo "Job: ${SLURM_JOB_ID:-local}  Host: $(hostname)"
echo "Start: $(date)"

if [[ -z "${HF_HOME}" ]]; then
  if [[ -d /playpen/huggingface/hub ]] && [[ -w /playpen/huggingface/hub ]]; then
    export HF_HOME=/playpen/huggingface
  elif [[ -d "/playpen/${USER}/hf_cache" ]]; then
    export HF_HOME="/playpen/${USER}/hf_cache"
  else
    export HF_HOME="${HOME}/.cache/huggingface"
  fi
fi

export USE_LOCAL_MODEL=true
export JUDGE_MODEL="google/gemma-4-31B-it"
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_MAX_MODEL_LEN=4096
export VLLM_ENFORCE_EAGER=1

GEMMA4_SHARED_PATH="/playpen/huggingface/hub/models--google--gemma-4-31B-it"
GEMMA4_PLAYPEN="${PLAYPEN:-/playpen}/guam/hf_cache/hub/models--google--gemma-4-31B-it"
GEMMA4_USER_PATH="${HF_HOME}/hub/models--google--gemma-4-31B-it"

resolve_gemma4_path() {
  local p snap
  for p in "${GEMMA4_SHARED_PATH}/snapshots" "${GEMMA4_PLAYPEN}/snapshots" "${GEMMA4_USER_PATH}/snapshots"; do
    if [[ -d "$p" ]]; then
      snap=$(find "$p" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
      if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then
        echo "${snap}"
        return 0
      fi
    fi
  done
  echo ""
  return 1
}

GEMMA4_MODEL_PATH=$(resolve_gemma4_path)
if [[ -n "${GEMMA4_MODEL_PATH}" ]]; then
  export LOCAL_MODEL="${GEMMA4_MODEL_PATH}"
  echo "✓ Gemma 4: ${GEMMA4_MODEL_PATH}"
else
  export LOCAL_MODEL="google/gemma-4-31B-it"
fi

if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

SUCCESSFUL=()
FAILED=()
for model in "${MODELS[@]}"; do
  echo ""
  echo "========== ${model} =========="
  O1="${ROOT_DIR}/output/DNA_New_Experiments/dna_benign_intent_guidelines_multimodel/${model}/p1_baseline_outputs.jsonl"
  O2="${ROOT_DIR}/output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/${model}/baseline_outputs.jsonl"
  OUT="${ROOT_DIR}/output/DNA_New_Experiments/evaluation/benign_guidelines_vs_refinement_${model}_gemma4_judge"

  if [[ ! -f "${O1}" ]] || [[ ! -f "${O2}" ]]; then
    echo "❌ Skip ${model}: missing O1 or O2"
    FAILED+=("${model}")
    continue
  fi

  mkdir -p "${OUT}"
  CMD_ARGS=(
    "--o1-jsonl" "${O1}"
    "--o2-jsonl" "${O2}"
    "--output-dir" "${OUT}"
    "--evals" "${EVALS}"
    "--temperature" "${TEMPERATURE}"
    "--max-tokens" "${MAX_TOKENS}"
    "--use-local-model"
    "--judge-model" "${LOCAL_MODEL}"
  )
  [[ -n "${MAX_RECORDS}" ]] && CMD_ARGS+=("--max-records" "${MAX_RECORDS}")

  if "${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"; then
    echo "✅ ${model}"
    SUCCESSFUL+=("${model}")
  else
    echo "❌ ${model}"
    FAILED+=("${model}")
  fi
done

echo ""
echo "Successful (${#SUCCESSFUL[@]}): ${SUCCESSFUL[*]}"
echo "Failed (${#FAILED[@]}): ${FAILED[*]}"
echo "End: $(date)"
[[ ${#FAILED[@]} -eq 0 ]]
