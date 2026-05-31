#!/usr/bin/bash
# Resume: baseline vs benign_intent_guidelines eval for llama8b and qwen35_9b
# Both models stopped at 3072/3267 records; this job resumes from the checkpoint.
#
# Usage:
#   sbatch src/slurm/evaluation/job_eval_resume_baseline_vs_benign_intent_guidelines_gemma4.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_resume_big
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== Resume: baseline vs benign_intent_guidelines (llama8b, qwen35_9b) ==="
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"

# ── Gemma 4 31B judge ────────────────────────────────────────────────────────
export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_ENFORCE_EAGER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

_COMMON_GEMMA_REV="/common/guam/hf_cache/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343"
_DEFAULT_GEMMA4_REV="$HF_HOME/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343"

_gemma_ok() {
  local d="${1%/}"
  [[ -d "$d" ]] && [[ -r "${d}/config.json" ]] || return 1
  local f; shopt -s nullglob
  for f in "${d}"/*.safetensors; do [[ -r "$f" ]] && { shopt -u nullglob; return 0; }; done
  shopt -u nullglob; return 1
}

_resolve_gemma4() {
  local root f rev
  _gemma_ok "${_COMMON_GEMMA_REV}" && echo "${_COMMON_GEMMA_REV}" && return 0
  for root in \
      "/common/guam/hf_cache/models--google--gemma-4-31b-it/snapshots" \
      "/common/guam/hf_cache/hub/models--google--gemma-4-31b-it/snapshots"; do
    [[ -d "$root" ]] || continue
    shopt -s nullglob
    for f in "${root}"/*/model.safetensors.index.json; do
      rev="$(dirname "$f")"
      if _gemma_ok "${rev}"; then echo "${rev}"; shopt -u nullglob; return 0; fi
    done
    shopt -u nullglob
  done
  _gemma_ok "${_DEFAULT_GEMMA4_REV}" && echo "${_DEFAULT_GEMMA4_REV}" && return 0
  for root in \
      "$HF_HOME/models--google--gemma-4-31b-it/snapshots" \
      "$HF_HOME/hub/models--google--gemma-4-31b-it/snapshots" \
      "$HF_HOME/models--google--gemma-4-31B-it/snapshots"; do
    [[ -d "$root" ]] || continue
    shopt -s nullglob
    for f in "${root}"/*/model.safetensors.index.json; do
      rev="$(dirname "$f")"
      if _gemma_ok "${rev}"; then echo "${rev}"; shopt -u nullglob; return 0; fi
    done
    shopt -u nullglob
  done
  return 1
}

GEMMA4_PATH="$(_resolve_gemma4)" || { echo "ERROR: Gemma4 judge not found"; exit 1; }
echo "Judge: ${GEMMA4_PATH}"

if [[ "${GEMMA4_PATH}" == *"/hub/models--google--gemma-"* ]]; then
  export HF_HOME="$(dirname "$(dirname "$(dirname "$(dirname "${GEMMA4_PATH}")")")")"
  export HF_HUB_CACHE="${HF_HOME}/hub"
else
  export HF_HOME="$(dirname "$(dirname "$(dirname "${GEMMA4_PATH}")")")"
  [[ -d "${HF_HOME}/hub" ]] && export HF_HUB_CACHE="${HF_HOME}/hub" || export HF_HUB_CACHE="${HF_HOME}"
fi
export HF_HUB_OFFLINE=1
export LOCAL_MODEL="${GEMMA4_PATH}"

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

EVALS="${EVALS:-all}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

MODELS=("llama8b" "qwen35_9b")
SUCCESSFUL_MODELS=()
FAILED_MODELS=()

for model in "${MODELS[@]}"; do
  echo ""
  echo "========================================"
  echo "Model: $model  ($(date))"
  echo "========================================"

  O1="${ROOT_DIR}/output/Linguasafe_Experiments/helpful_assistant_multimodel/${model}/checkpoint.jsonl"
  O2="${ROOT_DIR}/output/Linguasafe_Experiments/benign_intent_guidelines_multimodel/${model}/checkpoint.jsonl"
  OUT_DIR="${ROOT_DIR}/output/Linguasafe_Experiments/evaluation/baseline_vs_benign_intent_guidelines_${model}_gemma4_judge"

  [[ -f "${O1}" ]] || { echo "ERROR: O1 not found: ${O1}"; FAILED_MODELS+=("${model}"); continue; }
  [[ -f "${O2}" ]] || { echo "ERROR: O2 not found: ${O2}"; FAILED_MODELS+=("${model}"); continue; }

  mkdir -p "${OUT_DIR}"

  if "${PYTHON}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
      --o1-jsonl   "${O1}" \
      --o2-jsonl   "${O2}" \
      --output-dir "${OUT_DIR}" \
      --evals      "${EVALS}" \
      --temperature "${TEMPERATURE}" \
      --max-tokens  "${MAX_TOKENS}" \
      --use-local-model \
      --judge-model "${LOCAL_MODEL}"; then
    echo "SUCCESS: $model"
    SUCCESSFUL_MODELS+=("${model}")
  else
    echo "FAILED: $model"
    FAILED_MODELS+=("${model}")
  fi
done

echo ""
echo "========================================"
echo "DONE: ${#SUCCESSFUL_MODELS[@]}/${#MODELS[@]} models  ($(date))"
[[ ${#FAILED_MODELS[@]} -gt 0 ]] && printf 'FAILED: %s\n' "${FAILED_MODELS[@]}" && exit 1 || exit 0
