#!/usr/bin/bash
# Evaluation: LinguaSafe helpful_assistant_multimodel (baseline) vs benign_intent_guidelines_multimodel
# Processes all 6 models sequentially using Gemma 4 31B as judge.
#
# Usage:
#   sbatch src/slurm/evaluation/job_eval_linguasafe_baseline_vs_benign_intent_guidelines_gemma4.slurm
#
# Override model subset:
#   MODELS="llama8b qwen35_9b" sbatch ...
# Re-run one model from scratch (wipes its checkpoint/results):
#   MODELS=qwen35_9b RESTART_EVAL_FOR=qwen35_9b sbatch ...

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=eval_ls_baseline_vs_benign_guidelines
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

echo "=== LinguaSafe Eval: baseline vs benign_intent_guidelines — ALL MODELS ==="
echo "Job ID: ${SLURM_JOB_ID:-local}  Host: $(hostname)  Start: $(date)"
echo "Root: ${ROOT_DIR}"

MODELS_STR="${MODELS:-llama8b llama33_70b qwen35_9b qwen35_27b mistral_7b mistral_24b}"
read -ra MODELS <<< "${MODELS_STR}"

EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
RESTART_EVAL_FOR="${RESTART_EVAL_FOR:-}"

echo "Models: ${MODELS[*]}"

# ── Gemma 4 31B judge setup ─────────────────────────────────────────────────
echo ""
echo "=== Gemma 4 31B Judge ==="

export USE_LOCAL_MODEL=true
export JUDGE_MODEL="google/gemma-4-31B-it"
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_ENFORCE_EAGER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

unset HF_HOME HF_HUB_CACHE 2>/dev/null || true

_COMMON_GEMMA_REV="/common/guam/hf_cache/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343"
_DEFAULT_GEMMA4_REV="$HF_HOME/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343"

_gemma_revision_ok_for_vllm() {
  local d="${1%/}" f
  [[ -d "$d" ]] || return 1
  [[ -r "${d}/config.json" ]] || return 1
  shopt -s nullglob
  for f in "${d}"/*.safetensors; do
    if [[ -r "$f" ]]; then shopt -u nullglob; return 0; fi
  done
  shopt -u nullglob
  return 1
}

_resolve_gemma4_model_path() {
  local cand root f rev
  if [[ -n "${GEMMA4_SNAPSHOT_DIR:-}" ]]; then
    cand="${GEMMA4_SNAPSHOT_DIR%/}"
    if _gemma_revision_ok_for_vllm "${cand}"; then echo "${cand}"; return 0; fi
    if [[ -d "$cand" ]]; then
      shopt -s nullglob
      for f in "${cand}"/*/config.json "${cand}"/*/model.safetensors.index.json; do
        rev="$(dirname "${f}")"
        if _gemma_revision_ok_for_vllm "${rev}"; then echo "${rev}"; shopt -u nullglob; return 0; fi
      done
      shopt -u nullglob
    fi
  fi
  if _gemma_revision_ok_for_vllm "${_COMMON_GEMMA_REV}"; then echo "${_COMMON_GEMMA_REV}"; return 0; fi
  for root in \
      "/common/guam/hf_cache/models--google--gemma-4-31b-it/snapshots" \
      "/common/guam/hf_cache/hub/models--google--gemma-4-31b-it/snapshots" \
      "/common/guam/hf_cache/hub/models--google--gemma-4-31B-it/snapshots"; do
    [[ -d "$root" ]] || continue
    shopt -s nullglob
    for f in "${root}"/*/config.json "${root}"/*/model.safetensors.index.json; do
      rev="$(dirname "${f}")"
      if _gemma_revision_ok_for_vllm "${rev}"; then echo "${rev}"; shopt -u nullglob; return 0; fi
    done
    shopt -u nullglob
  done
  if _gemma_revision_ok_for_vllm "${_DEFAULT_GEMMA4_REV}"; then echo "${_DEFAULT_GEMMA4_REV}"; return 0; fi
  for root in \
      "$HF_HOME/models--google--gemma-4-31b-it/snapshots" \
      "$HF_HOME/hub/models--google--gemma-4-31b-it/snapshots" \
      "$HF_HOME/models--google--gemma-4-31B-it/snapshots" \
      "/playpen/${USER}/hf_cache/models--google--gemma-4-31b-it/snapshots"; do
    [[ -d "$root" ]] || continue
    shopt -s nullglob
    for f in "${root}"/*/config.json "${root}"/*/model.safetensors.index.json; do
      rev="$(dirname "${f}")"
      if _gemma_revision_ok_for_vllm "${rev}"; then echo "${rev}"; shopt -u nullglob; return 0; fi
    done
    shopt -u nullglob
  done
  return 1
}

GEMMA4_MODEL_PATH=""
GEMMA4_MODEL_PATH="$(_resolve_gemma4_model_path)" || true
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH%/}"

if [[ -z "${GEMMA4_MODEL_PATH}" ]] || ! _gemma_revision_ok_for_vllm "${GEMMA4_MODEL_PATH}"; then
  echo "ERROR: Gemma judge dir is missing readable weight shards (*.safetensors)."
  echo "  Host: $(hostname)  Picked: ${GEMMA4_MODEL_PATH:-<none>}"
  exit 1
fi

if [[ "${GEMMA4_MODEL_PATH}" == *"/hub/models--google--gemma-"* ]]; then
  export HF_HOME="$(dirname "$(dirname "$(dirname "$(dirname "${GEMMA4_MODEL_PATH}")")")")"
  export HF_HUB_CACHE="${HF_HOME}/hub"
else
  export HF_HOME="$(dirname "$(dirname "$(dirname "${GEMMA4_MODEL_PATH}")")")"
  [[ -d "${HF_HOME}/hub" ]] && export HF_HUB_CACHE="${HF_HOME}/hub" || export HF_HUB_CACHE="${HF_HOME}"
fi
export LOCAL_MODEL="${GEMMA4_MODEL_PATH}"
export HF_HUB_OFFLINE=1

echo "Judge: ${LOCAL_MODEL}"

PYTHON_CMD="$CONDA_PREFIX/envs/vllm-serve/bin/python"
echo "Python: $("${PYTHON_CMD}" -c 'import sys; print(sys.executable)')"

echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

# ── Per-model eval loop ──────────────────────────────────────────────────────
SUCCESSFUL_MODELS=()
FAILED_MODELS=()
TOTAL_MODELS=${#MODELS[@]}
CURRENT_MODEL=0

for model in "${MODELS[@]}"; do
  CURRENT_MODEL=$((CURRENT_MODEL + 1))
  echo ""
  echo "========================================"
  echo "Model $CURRENT_MODEL/$TOTAL_MODELS: $model  ($(date))"
  echo "========================================"

  O1_BASELINE="${ROOT_DIR}/output/Linguasafe_Experiments/helpful_assistant_multimodel/${model}/checkpoint.jsonl"
  O2_GUIDELINES="${ROOT_DIR}/output/Linguasafe_Experiments/benign_intent_guidelines_multimodel/${model}/checkpoint.jsonl"
  EVAL_OUT_DIR="${ROOT_DIR}/output/Linguasafe_Experiments/evaluation/baseline_vs_benign_intent_guidelines_${model}_gemma4_judge"

  if [[ ! -f "${O1_BASELINE}" ]]; then
    echo "ERROR: baseline not found: ${O1_BASELINE}"; FAILED_MODELS+=("${model}"); continue
  fi
  if [[ ! -f "${O2_GUIDELINES}" ]]; then
    echo "ERROR: benign_intent_guidelines not found: ${O2_GUIDELINES}"; FAILED_MODELS+=("${model}"); continue
  fi

  echo "  O1 (baseline):              ${O1_BASELINE}"
  echo "  O2 (benign_intent+guidelines): ${O2_GUIDELINES}"
  echo "  Output:                     ${EVAL_OUT_DIR}"

  mkdir -p "${EVAL_OUT_DIR}"

  CMD_ARGS=(
    "--o1-jsonl"   "${O1_BASELINE}"
    "--o2-jsonl"   "${O2_GUIDELINES}"
    "--output-dir" "${EVAL_OUT_DIR}"
    "--evals"      "${EVALS}"
    "--temperature" "${TEMPERATURE}"
    "--max-tokens" "${MAX_TOKENS}"
    "--use-local-model"
    "--judge-model" "${LOCAL_MODEL}"
  )
  [[ -n "${MAX_RECORDS}" ]] && CMD_ARGS+=("--max-records" "${MAX_RECORDS}")

  if [[ -n "${RESTART_EVAL_FOR}" ]]; then
    for _rem in ${RESTART_EVAL_FOR}; do
      if [[ "${_rem}" == "${model}" ]]; then
        CMD_ARGS+=("--restart-eval")
        echo "  (--restart-eval for ${model})"
        break
      fi
    done
  fi

  if "${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"; then
    echo "SUCCESS: $model"
    SUCCESSFUL_MODELS+=("${model}")
    summary="${EVAL_OUT_DIR}/eval_summary.json"
    [[ -f "${summary}" ]] && head -20 "${summary}" | grep -E "(helpfulness|harmlessness|total)" || true
  else
    echo "FAILED: $model"
    FAILED_MODELS+=("${model}")
  fi
done

echo ""
echo "========================================"
echo "DONE: ${#SUCCESSFUL_MODELS[@]}/${TOTAL_MODELS} models  ($(date))"
echo "========================================"
[[ ${#FAILED_MODELS[@]} -gt 0 ]] && printf 'FAILED: %s\n' "${FAILED_MODELS[@]}" && exit 1 || exit 0
