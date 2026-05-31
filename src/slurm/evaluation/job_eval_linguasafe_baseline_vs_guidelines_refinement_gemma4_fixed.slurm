#!/usr/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=eval_linguasafe_baseline_vs_guidelines_refinement_gemma4
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
  [[ -d "${_ROOT_CAND}/src/evaluation_ranking" ]] && ROOT_DIR="${_ROOT_CAND}"
fi
[[ -z "$ROOT_DIR" ]] || [[ ! -d "$ROOT_DIR/src/evaluation_ranking" ]] && echo "ERROR: repo root not found" && exit 1

cd "$ROOT_DIR" || exit 1
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== LinguaSafe: Baseline vs Guidelines Refinement Eval ==="
echo "Job: ${SLURM_JOB_ID:-local} | Host: $(hostname) | Start: $(date)"

MODELS_STR="${MODELS:-llama8b llama33_70b qwen35_9b qwen35_27b mistral_7b mistral_24b}"
read -ra MODELS <<< "${MODELS_STR}"
EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
RESTART_EVAL_FOR="${RESTART_EVAL_FOR:-}"

echo "Models: ${MODELS[*]}"

export HF_HOME=/playpen/huggingface
export HF_HUB_CACHE=/playpen/huggingface/hub
export USE_LOCAL_MODEL=true
export JUDGE_MODEL="google/gemma-4-31B-it"
export VLLM_TRUST_REMOTE_CODE=1
# export VLLM_ONLY=1  # Disabled to allow HuggingFace fallback
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_ENFORCE_EAGER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

GEMMA4_PATH=""
for snap in /playpen/huggingface/hub/models--google--gemma-4-31B-it/snapshots/*/; do
  [[ -d "$snap" ]] && GEMMA4_PATH="${snap%/}" && break
done
[[ -n "$GEMMA4_PATH" ]] && export LOCAL_MODEL="$GEMMA4_PATH" || export LOCAL_MODEL="google/gemma-4-31B-it"
echo "Judge: ${LOCAL_MODEL}"

if [[ -x "/common/guam/anaconda3/bin/python3" ]]; then
  PYTHON_CMD="/common/guam/anaconda3/bin/python3"
elif [[ -x "${HOME}/anaconda3/bin/python3" ]]; then
  PYTHON_CMD="${HOME}/anaconda3/bin/python3"
else
  PYTHON_CMD="$(command -v python3)"
fi
echo "Python: $("${PYTHON_CMD}" -c 'import sys; print(sys.executable)')"

nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv

SUCCESSFUL_MODELS=()
FAILED_MODELS=()
TOTAL=${#MODELS[@]}; N=0

for model in "${MODELS[@]}"; do
  N=$((N+1))
  echo ""
  echo "=== Model $N/$TOTAL: $model | $(date) ==="
  REFINEMENT_FILE="${ROOT_DIR}/output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_guidelines_multimodel/${model}/checkpoint.jsonl"
  EVAL_OUT_DIR="${ROOT_DIR}/output/Linguasafe_Experiments/evaluation/baseline_vs_guidelines_refinement_${model}_gemma4_judge"

  if [[ ! -f "${REFINEMENT_FILE}" ]]; then
    echo "ERROR: not found: ${REFINEMENT_FILE}"; FAILED_MODELS+=("${model}"); continue
  fi
  echo "Input: ${REFINEMENT_FILE} ($(wc -l < "${REFINEMENT_FILE}") lines)"
  mkdir -p "${EVAL_OUT_DIR}"

  EXTRA_ARGS=()
  echo " ${RESTART_EVAL_FOR} " | grep -qw "${model}" && EXTRA_ARGS+=("--restart-eval") || true
  [[ -n "${MAX_RECORDS}" ]] && EXTRA_ARGS+=("--max-records" "${MAX_RECORDS}")

  if "${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
      --refinement-jsonl "${REFINEMENT_FILE}" \
      --output-dir       "${EVAL_OUT_DIR}" \
      --evals            "${EVALS}" \
      --temperature      "${TEMPERATURE}" \
      --max-tokens       "${MAX_TOKENS}" \
      --use-local-model \
      --judge-model      "${LOCAL_MODEL}" \
      "${EXTRA_ARGS[@]}"; then
    echo "SUCCESS: ${model}"; SUCCESSFUL_MODELS+=("${model}")
    [[ -f "${EVAL_OUT_DIR}/eval_checkpoint.jsonl" ]] && echo "  checkpoint lines: $(wc -l < "${EVAL_OUT_DIR}/eval_checkpoint.jsonl")"
    [[ -f "${EVAL_OUT_DIR}/eval_summary.json" ]] && grep -E "(helpfulness|harmlessness|win|total)" "${EVAL_OUT_DIR}/eval_summary.json" | head -6 || true
  else
    echo "FAILED: ${model}"; FAILED_MODELS+=("${model}")
  fi
done

echo ""; echo "=== DONE: ${#SUCCESSFUL_MODELS[@]}/${TOTAL} | $(date) ==="
if ((${#FAILED_MODELS[@]} > 0)); then
  printf 'Failed models: %s\n' "${FAILED_MODELS[@]}"
  exit 1
fi
exit 0
