#!/usr/bin/bash
# Evaluation: DNA Benign Intent + Guidelines vs DNA Refinement (from baseline, no extra guidelines pass)
# Judge: Gemma 4 31B. O1 = benign+guidelines response, O2 = refinement output (paired by p1_id via o2-jsonl rows).
#
# Usage:
#   sbatch src/slurm/evaluation/job_eval_benign_guidelines_vs_refinement_gemma4.slurm
#   MODEL=qwen35_27b sbatch ...
#   MAX_RECORDS=50 EVALS=helpfulness sbatch ...
#
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_benign_guidelines_vs_refinement_gemma4
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
  echo "ERROR: Could not resolve repo root. Submit from repo or set ROOT_DIR."
  exit 1
fi

cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

MODEL="${MODEL:-llama8b}"
EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

echo "=== Benign Intent + Guidelines vs Refinement (Gemma 4 31B) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Root: ${ROOT_DIR}"
echo "MODEL: ${MODEL}"

# O1: benign + guidelines (per-p1 output). O2 driver: refinement outputs (must share p1_id with O1 map).
O1_BENIGN_GUIDELINES="${ROOT_DIR}/output/DNA_New_Experiments/dna_benign_intent_guidelines_multimodel/${MODEL}/p1_baseline_outputs.jsonl"
O2_REFINEMENT="${ROOT_DIR}/output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/${MODEL}/baseline_outputs.jsonl"

EVAL_OUT_DIR="${ROOT_DIR}/output/DNA_New_Experiments/evaluation/benign_guidelines_vs_refinement_${MODEL}_gemma4_judge"

if [[ ! -f "${O1_BENIGN_GUIDELINES}" ]]; then
  echo "ERROR: Benign + guidelines file not found: ${O1_BENIGN_GUIDELINES}"
  exit 1
fi
if [[ ! -f "${O2_REFINEMENT}" ]]; then
  echo "ERROR: Refinement file not found: ${O2_REFINEMENT}"
  exit 1
fi

echo "  O1 (benign+guidelines): ${O1_BENIGN_GUIDELINES}"
echo "  O2 (refinement):        ${O2_REFINEMENT}"
echo "  Output:                 ${EVAL_OUT_DIR}"

# Gemma 4 31B judge env (same pattern as other DNA eval jobs)
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d /playpen/huggingface/hub ]] && [[ -w /playpen/huggingface/hub ]]; then
    export HF_HOME=/playpen/huggingface
  else
    PLAYPEN_USER_HF="${PLAYPEN_USER_HF:-/playpen/${USER}/huggingface}"
    if [[ -d "/playpen/${USER}/hf_cache" ]]; then
      export HF_HOME="/playpen/${USER}/hf_cache"
    elif mkdir -p "${PLAYPEN_USER_HF}/hub" 2>/dev/null && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
    else
      export HF_HOME="${HOME}/.cache/huggingface"
    fi
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
  local p
  for p in "${GEMMA4_SHARED_PATH}/snapshots" "${GEMMA4_PLAYPEN}/snapshots" "${GEMMA4_USER_PATH}/snapshots"; do
    if [[ -d "$p" ]]; then
      local snap
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
  echo "✓ Cached Gemma 4 31B: ${GEMMA4_MODEL_PATH}"
else
  export LOCAL_MODEL="google/gemma-4-31B-it"
  echo "⚠ Using repo id: google/gemma-4-31B-it"
fi

if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

mkdir -p "${EVAL_OUT_DIR}"

CMD_ARGS=(
  "--o1-jsonl" "${O1_BENIGN_GUIDELINES}"
  "--o2-jsonl" "${O2_REFINEMENT}"
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

echo "${PYTHON_CMD} src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py ${CMD_ARGS[*]}"
"${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"

echo ""
echo "Done. See: ${EVAL_OUT_DIR}/eval_results.jsonl , ${EVAL_OUT_DIR}/eval_summary.json"
exit $?
