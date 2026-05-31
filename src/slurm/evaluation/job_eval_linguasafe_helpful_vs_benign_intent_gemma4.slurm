#!/usr/bin/bash
# Evaluation: LinguaSafe helpful_assistant_multimodel (baseline) vs benign_intent_multimodel
# Processes model comparisons sequentially using Gemma 4 31B as judge (default 6 models).
#
# Usage:
#   sbatch src/slurm/evaluation/job_eval_linguasafe_helpful_vs_benign_intent_gemma4.slurm
# Interactive 2×GPU on gpu1 for 48h, then run the same script with bash (set TP=2; raise GPU mem for KV):
#   srun -p a6000 --qos=gpu_access -N1 -n1 --nodelist=gpu1 --cpus-per-task=16 --gres=gpu:2 --time=48:00:00 \\
#     bash -lc 'cd /common/guam/Safeshift && export VLLM_TENSOR_PARALLEL_SIZE=2 VLLM_GPU_MEMORY_UTILIZATION=0.90 \\
#       VLLM_MAX_MODEL_LEN=3072 MODELS="mistral_24b qwen35_27b" \\
#       && bash src/slurm/evaluation/job_eval_linguasafe_helpful_vs_benign_intent_gemma4.slurm'
# One-time (any node that can read playpen): copy the *entire* HF model dir (snapshots + blobs + refs).
# Trailing slashes matter: source must include blobs/ or symlinks in snapshots/ stay broken.
#   mkdir -p /common/guam/hf_cache && rsync -aL --info=progress2 \
#     $HF_HOME/models--google--gemma-4-31b-it/ \
#     /common/guam/hf_cache/models--google--gemma-4-31b-it/
# Verify on any GPU node: ls /common/guam/hf_cache/.../snapshots/145dc*/model*.safetensors
# Override judge dir if this node does not mount the default playpen path:
#   GEMMA4_SNAPSHOT_DIR=/path/to/snapshots/<rev> sbatch ...
# Re-run one model from scratch (deletes its eval checkpoint+results first):
#   MODELS=qwen35_9b RESTART_EVAL_FOR=qwen35_9b sbatch src/slurm/evaluation/job_eval_linguasafe_helpful_vs_benign_intent_gemma4.slurm
#
# Default models: llama8b, llama33_70b, qwen35_9b, qwen35_27b, mistral_7b, mistral_24b (override MODELS=...)
# Comparison: helpful_assistant_multimodel vs benign_intent_multimodel

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=eval_linguasafe_helpful_vs_benign_gemma4
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
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== LinguaSafe Evaluation: helpful_assistant (baseline) vs benign_intent - ALL MODELS ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Root: ${ROOT_DIR}"
echo "Start time: $(date)"

# Models present in BOTH helpful_assistant_multimodel and benign_intent_multimodel.
# Override per run: MODELS="mistral_7b llama33_70b" sbatch ...
MODELS_STR="${MODELS:-llama8b llama33_70b qwen35_9b qwen35_27b mistral_7b mistral_24b}"
read -ra MODELS <<< "${MODELS_STR}"

# Parameters
EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
# Space-separated names (e.g. qwen35_9b) to wipe checkpoint/results and re-run full eval for those models only.
RESTART_EVAL_FOR="${RESTART_EVAL_FOR:-}"

echo "Models to process: ${MODELS[*]}"
echo "Evaluations: ${EVALS}"
echo "Max records per model: ${MAX_RECORDS:-all}"

# Setup Gemma 4 31B as judge — one path only (same idea as baseline_vs_guidelines: set LOCAL_MODEL, no search).
# Do not inherit HF_HOME from login env (/common/guam/hf_cache etc.); vLLM uses HF_* from the snapshot tree.
echo ""
echo "=== Setting up Gemma 4 31B Judge ==="

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

# Same revision hash as playpen default; under /common use an rsync -aL materialized tree (no broken blob symlinks).
_COMMON_GEMMA_REV="/common/guam/hf_cache/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343"
_DEFAULT_GEMMA4_REV="$HF_HOME/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343"

# HF cache: config.json may be readable while weight shards are still missing/broken symlinks.
# vLLM requires at least one readable *.safetensors in the revision directory.
_gemma_revision_ok_for_vllm() {
  local d="${1%/}" f
  [[ -d "$d" ]] || return 1
  [[ -r "${d}/config.json" ]] || return 1
  shopt -s nullglob
  for f in "${d}"/*.safetensors; do
    if [[ -r "$f" ]]; then
      shopt -u nullglob
      return 0
    fi
  done
  shopt -u nullglob
  return 1
}

_resolve_gemma4_model_path() {
  local cand root f rev
  if [[ -n "${GEMMA4_SNAPSHOT_DIR:-}" ]]; then
    cand="${GEMMA4_SNAPSHOT_DIR%/}"
    if _gemma_revision_ok_for_vllm "${cand}"; then
      echo "${cand}"
      return 0
    fi
    if [[ -d "$cand" ]]; then
      shopt -s nullglob
      for f in "${cand}"/*/config.json "${cand}"/*/model.safetensors.index.json; do
        rev="$(dirname "${f}")"
        if _gemma_revision_ok_for_vllm "${rev}"; then
          echo "${rev}"
          shopt -u nullglob
          return 0
        fi
      done
      shopt -u nullglob
    fi
  fi
  if _gemma_revision_ok_for_vllm "${_COMMON_GEMMA_REV}"; then
    echo "${_COMMON_GEMMA_REV}"
    return 0
  fi
  for root in \
    "/common/guam/hf_cache/models--google--gemma-4-31b-it/snapshots" \
    "/common/guam/hf_cache/hub/models--google--gemma-4-31b-it/snapshots" \
    "/common/guam/hf_cache/hub/models--google--gemma-4-31B-it/snapshots"; do
    [[ -d "$root" ]] || continue
    shopt -s nullglob
    for f in "${root}"/*/config.json "${root}"/*/model.safetensors.index.json; do
      rev="$(dirname "${f}")"
      if _gemma_revision_ok_for_vllm "${rev}"; then
        echo "${rev}"
        shopt -u nullglob
        return 0
      fi
    done
    shopt -u nullglob
  done
  if _gemma_revision_ok_for_vllm "${_DEFAULT_GEMMA4_REV}"; then
    echo "${_DEFAULT_GEMMA4_REV}"
    return 0
  fi
  for root in \
    "$HF_HOME/models--google--gemma-4-31b-it/snapshots" \
    "$HF_HOME/hub/models--google--gemma-4-31b-it/snapshots" \
    "$HF_HOME/hub/models--google--gemma-4-31B-it/snapshots" \
    "/playpen/${USER}/hf_cache/models--google--gemma-4-31b-it/snapshots" \
    "/playpen/${USER}/hf_cache/hub/models--google--gemma-4-31B-it/snapshots" \
    "$HF_HOME/models--google--gemma-4-31B-it/snapshots"; do
    [[ -d "$root" ]] || continue
    shopt -s nullglob
    for f in "${root}"/*/config.json "${root}"/*/model.safetensors.index.json; do
      rev="$(dirname "${f}")"
      if _gemma_revision_ok_for_vllm "${rev}"; then
        echo "${rev}"
        shopt -u nullglob
        return 0
      fi
    done
    shopt -u nullglob
  done
  return 1
}

GEMMA4_MODEL_PATH=""
GEMMA4_MODEL_PATH="$(_resolve_gemma4_model_path)" || true
GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH%/}"

if [[ -z "${GEMMA4_MODEL_PATH}" ]] || ! _gemma_revision_ok_for_vllm "${GEMMA4_MODEL_PATH}"; then
  echo "ERROR: Gemma judge dir is missing readable weight shards (*.safetensors) for vLLM."
  echo "  Host: $(hostname)  Picked: ${GEMMA4_MODEL_PATH:-<none>}"
  echo "  config.json alone is not enough — copy the full HF model tree (snapshots + blobs)."
  echo "  Check: ls \"\${GEMMA4_SNAPSHOT_DIR:-/common/.../snapshots/145dc...}\"/*.safetensors"
  echo "  One-time (from a node that sees playpen blobs):"
  echo "    mkdir -p /common/guam/hf_cache && rsync -aL $HF_HOME/models--google--gemma-4-31b-it/ /common/guam/hf_cache/models--google--gemma-4-31b-it/"
  exit 1
fi

if [[ "${GEMMA4_MODEL_PATH}" == *"/hub/models--google--gemma-"* ]]; then
  export HF_HOME="$(dirname "$(dirname "$(dirname "$(dirname "${GEMMA4_MODEL_PATH}")")")")"
  export HF_HUB_CACHE="${HF_HOME}/hub"
else
  export HF_HOME="$(dirname "$(dirname "$(dirname "${GEMMA4_MODEL_PATH}")")")"
  if [[ -d "${HF_HOME}/hub" ]]; then
    export HF_HUB_CACHE="${HF_HOME}/hub"
  else
    export HF_HUB_CACHE="${HF_HOME}"
  fi
fi
export LOCAL_MODEL="${GEMMA4_MODEL_PATH}"
export HF_HUB_OFFLINE=1

echo "✓ Judge LOCAL_MODEL=${LOCAL_MODEL}"
echo "✓ HF_HOME=${HF_HOME}  HF_HUB_CACHE=${HF_HUB_CACHE}"

# Python: prefer conda base (same stack as vLLM jobs); avoid ~/.bashrc hijacking python3.
if [[ -x "/common/guam/anaconda3/bin/python3" ]]; then
  PYTHON_CMD="/common/guam/anaconda3/bin/python3"
elif [[ -x "${HOME}/anaconda3/bin/python3" ]]; then
  PYTHON_CMD="${HOME}/anaconda3/bin/python3"
else
  PYTHON_CMD="$(command -v python3)"
fi
echo "Python: $("${PYTHON_CMD}" -c 'import sys; print(sys.executable)')"

# GPU diagnostics
echo ""
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

# Process each model (explicit init: safe with bash set -u)
SUCCESSFUL_MODELS=()
FAILED_MODELS=()
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
  O1_BASELINE="${ROOT_DIR}/output/Linguasafe_Experiments/helpful_assistant_multimodel/${model}/checkpoint.jsonl"
  O2_BENIGN_INTENT="${ROOT_DIR}/output/Linguasafe_Experiments/benign_intent_multimodel/${model}/checkpoint.jsonl"

  # Output directory
  EVAL_OUT_DIR="${ROOT_DIR}/output/Linguasafe_Experiments/evaluation/helpful_vs_benign_intent_${model}_gemma4_judge"

  # Validate input files
  if [[ ! -f "${O1_BASELINE}" ]]; then
    echo "❌ ERROR: Baseline file not found: ${O1_BASELINE}"
    FAILED_MODELS+=("${model}")
    continue
  fi
  if [[ ! -f "${O2_BENIGN_INTENT}" ]]; then
    echo "❌ ERROR: Benign intent file not found: ${O2_BENIGN_INTENT}"
    FAILED_MODELS+=("${model}")
    continue
  fi

  echo "Evaluating:"
  echo "  helpful_assistant (O1): ${O1_BASELINE}"
  echo "  benign_intent     (O2): ${O2_BENIGN_INTENT}"
  echo "  Output dir:         ${EVAL_OUT_DIR}"

  mkdir -p "${EVAL_OUT_DIR}"

  CMD_ARGS=(
    "--o1-jsonl" "${O1_BASELINE}"
    "--o2-jsonl" "${O2_BENIGN_INTENT}"
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

  if [[ -n "${RESTART_EVAL_FOR}" ]]; then
    for _rem in ${RESTART_EVAL_FOR}; do
      if [[ "${_rem}" == "${model}" ]]; then
        CMD_ARGS+=("--restart-eval")
        echo "RESTART_EVAL_FOR: --restart-eval for ${model} (full dataset, removes prior checkpoint/results)."
        break
      fi
    done
  fi

  echo ""
  echo "Running evaluation for $model..."
  echo "Command: ${PYTHON_CMD} src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py ${CMD_ARGS[*]}"

  if "${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"; then
    echo "✅ SUCCESS: $model evaluation completed"
    SUCCESSFUL_MODELS+=("${model}")

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
    echo "  output/Linguasafe_Experiments/evaluation/helpful_vs_benign_intent_${model}_gemma4_judge/"
  done
fi

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo ""
  echo "❌ Failed models:"
  printf '  %s\n' "${FAILED_MODELS[@]}"
fi

echo ""
echo "Job completed with ${#SUCCESSFUL_MODELS[@]}/${TOTAL_MODELS} models successful"

[[ ${#FAILED_MODELS[@]} -gt 0 ]] && exit 1 || exit 0
