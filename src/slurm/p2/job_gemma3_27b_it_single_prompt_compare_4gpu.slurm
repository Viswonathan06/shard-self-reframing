#!/usr/bin/bash
# One allocation: Gemma-3-27B-IT — P1-only baseline (O1) then P2-controlled (O2) on the
# same first N rows of a shared P1→P2 JSONL (default: OpenAI-one / Qwen35 P1→P2 run).
#
# Token limits (override with env):
#   MAX_TOKENS_P1=500  — matches paper / baseline_common “response generation” cap.
#   MAX_TOKENS_P2=700  — matches job_p2_small_models_controlled_4gpu.slurm (--max-tokens 700).
#
# Run:
#   sbatch src/slurm/p2/job_gemma3_27b_it_single_prompt_compare_4gpu.slurm
#
# Examples:
#   MAX_RECORDS=1 sbatch ...                                    # first row only (default)
#   P1_P2_JSONL=/path/to.jsonl P1_OUT_DIR=... P2_OUT_DIR=... sbatch ...
#   MAX_TOKENS_P1=512 MAX_TOKENS_P2=512 sbatch ...
#   EXIT_ON_NULL=0 sbatch ...    # keep going if a completion is null
#   SKIP_PHASE1=1 sbatch ...     # only run P2 controlled (reuse existing P1 outputs dir)
#   SKIP_PHASE2=1 sbatch ...     # only run P1 baseline
#
# Same vLLM / HF / model resolution behavior as job_gemma3_27b_it_controlled_openai_one_4gpu.slurm.

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=gemma27b_p1p2_1row
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [[ -z "${HF_HOME}" ]]; then
  if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
  else
    PLAYPEN_USER_HF="${PLAYPEN_USER_HF:-/playpen/${USER}/huggingface}"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
      echo "NOTE: $HF_HOME not writable; HF_HOME=${HF_HOME} (user playpen cache)."
    else
      export HF_HOME="${HOME}/.cache/huggingface"
      echo "NOTE: Using HF_HOME=${HF_HOME} (home cache)."
    fi
  fi
fi

export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
export VLLM_ONLY="${VLLM_ONLY:-1}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export EXIT_ON_NULL="${EXIT_ON_NULL:-1}"

GEMMA3_CLEAR_HUB_CACHE="${GEMMA3_CLEAR_HUB_CACHE:-0}"

MAX_RECORDS="${MAX_RECORDS:-1}"
MAX_TOKENS_P1="${MAX_TOKENS_P1:-500}"
MAX_TOKENS_P2="${MAX_TOKENS_P2:-700}"

P1_P2_JSONL="${P1_P2_JSONL:-${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines_openai_one_qwen35_27b/p1_p2_outputs_openai_one.jsonl}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${ROOT_DIR}/src/prompts/p2_response_system_prompt_controlled.txt}"

COMPARE_ROOT="${COMPARE_ROOT:-${ROOT_DIR}/output/gemma27b_single_prompt_compare_${SLURM_JOB_ID:-local}}"
P1_OUT_DIR="${P1_OUT_DIR:-${COMPARE_ROOT}/p1_baseline}"
P2_OUT_DIR="${P2_OUT_DIR:-${COMPARE_ROOT}/p2_controlled}"

WANDB_ARG=""
[[ -n "${WANDB:-}" ]] && WANDB_ARG="--wandb"
USE_HF_ARG=""
if [[ -n "${USE_HF:-}" ]]; then
  USE_HF_ARG="--use-hf"
  export VLLM_ONLY=0
fi

if [[ ! -f "${P1_P2_JSONL}" ]]; then
  echo "ERROR: P1_P2_JSONL not found: ${P1_P2_JSONL}"
  exit 1
fi
if [[ ! -f "${SYSTEM_PROMPT_FILE}" ]]; then
  echo "ERROR: System prompt file not found: ${SYSTEM_PROMPT_FILE}"
  exit 1
fi

if [[ -n "${CONDA_PYTHON:-}" ]]; then
  PYTHON_CMD="${CONDA_PYTHON}"
  echo "Using CONDA_PYTHON=${PYTHON_CMD}"
elif [[ -n "${USE_CONDA:-}" ]] && [ -f "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_CMD="$CONDA_PREFIX/bin/python"
  echo "Using conda python: ${PYTHON_CMD}"
elif [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
  echo "Using project .venv"
elif [ -f "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
  PYTHON_CMD="python3"
fi

echo "=== GPU diagnostics (user: $USER, job: ${SLURM_JOB_ID:-N/A}) ==="
nvidia-smi 2>/dev/null || true
echo "=== end GPU diagnostics ==="

HF_HUB="${HF_HOME}/hub"
mkdir -p "${HF_HUB}"

if [[ "${GEMMA3_CLEAR_HUB_CACHE:-0}" =~ ^(1|true|yes|YES)$ ]]; then
  _GEMMA_HUB_DIR="${HF_HUB}/models--google--gemma-3-27b-it"
  if [[ -d "${_GEMMA_HUB_DIR}" ]]; then
    echo "GEMMA3_CLEAR_HUB_CACHE=1: removing ${_GEMMA_HUB_DIR}"
    rm -rf "${_GEMMA_HUB_DIR}"
  fi
  export GEMMA3_SKIP_SHARED_SNAPSHOT=1
fi

resolve_model_path() {
  local path="$1"
  if [[ -d "${path}" ]] && [[ -f "${path}/config.json" ]]; then
    echo "${path}"
    return 0
  fi
  if [[ -d "${path}/snapshots" ]]; then
    local snap
    snap=$(find "${path}/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then
      echo "${snap}"
      return 0
    fi
  fi
  echo ""
}

ensure_model() {
  local repo_id="$1"
  local path="$2"
  local resolved
  resolved=$(resolve_model_path "${path}")
  if [[ -n "${resolved}" ]]; then
    echo "${resolved}"
    return 0
  fi
  echo "Downloading missing model: ${repo_id} -> ${HF_HUB}" 1>&2
  local downloaded
  downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
repo_id = sys.argv[1]
cache_dir = sys.argv[2]
token = os.getenv('HF_TOKEN')
try:
    path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, resume_download=True, token=token)
    print(path)
except Exception as e:
    print(f'DOWNLOAD_FAILED: {e}', file=sys.stderr)
    sys.exit(1)
" "${repo_id}" "${HF_HUB}")
  if [[ $? -ne 0 ]] || [[ -z "${downloaded}" ]]; then
    return 1
  fi
  echo "${downloaded}"
  return 0
}

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. Gated Gemma download may fail if not already cached."
fi

SHARED_GEMMA_HUB="$HF_HOME/models--google--gemma-3-27b-it"
if [[ -z "${GEMMA3_27B_PATH:-}" ]] && ! [[ "${GEMMA3_SKIP_SHARED_SNAPSHOT:-0}" =~ ^(1|true|yes|YES)$ ]]; then
  shared_resolved=$(resolve_model_path "${SHARED_GEMMA_HUB}")
  if [[ -n "${shared_resolved}" ]] && [[ -f "${shared_resolved}/config.json" ]]; then
    GEMMA3_27B_PATH="${shared_resolved}"
    echo "NOTE: GEMMA3_27B_PATH unset; using shared hub snapshot: ${GEMMA3_27B_PATH}"
  fi
fi
GEMMA3_27B="${GEMMA3_27B_PATH:-${HF_HUB}/models--google--gemma-3-27b-it}"
GEMMA3_27B=$(ensure_model "google/gemma-3-27b-it" "${GEMMA3_27B}") || {
  echo "ERROR: Could not resolve or download google/gemma-3-27b-it"
  exit 1
}

MAX_REC_ARGS=(--max-records "${MAX_RECORDS}")

echo ""
echo "=== Gemma-3-27B single-prompt compare ==="
echo "Shared JSONL:  ${P1_P2_JSONL}"
echo "MAX_RECORDS:   ${MAX_RECORDS}"
echo "MAX_TOKENS_P1: ${MAX_TOKENS_P1}  |  MAX_TOKENS_P2: ${MAX_TOKENS_P2}"
echo "P1 out:        ${P1_OUT_DIR}"
echo "P2 out:        ${P2_OUT_DIR}"
echo "System prompt: ${SYSTEM_PROMPT_FILE}"
echo "Model:         ${GEMMA3_27B}"
echo ""

mkdir -p "${P1_OUT_DIR}" "${P2_OUT_DIR}"

if [[ -z "${SKIP_PHASE1:-}" ]]; then
  echo "---------- Phase 1: P1 baseline (O1) ----------"
  $PYTHON_CMD src/baselines/generate_p1_baseline_outputs.py \
    --input-jsonl "${P1_P2_JSONL}" \
    --output-dir "${P1_OUT_DIR}" \
    --use-local-model \
    --local-model "${GEMMA3_27B}" \
    --temperature 0.3 \
    --max-tokens "${MAX_TOKENS_P1}" \
    "${MAX_REC_ARGS[@]}" \
    ${USE_HF_ARG} \
    ${WANDB_ARG}
else
  echo "SKIP_PHASE1 set — skipping P1 baseline."
fi

if [[ -z "${SKIP_PHASE2:-}" ]]; then
  echo "---------- Phase 2: P2 controlled (O2) ----------"
  $PYTHON_CMD src/baselines/generate_p2_controlled_outputs.py \
    --input-jsonl "${P1_P2_JSONL}" \
    --output-dir "${P2_OUT_DIR}" \
    --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
    --use-local-model \
    --local-model "${GEMMA3_27B}" \
    --temperature 0.3 \
    --max-tokens "${MAX_TOKENS_P2}" \
    "${MAX_REC_ARGS[@]}" \
    ${USE_HF_ARG} \
    ${WANDB_ARG}
else
  echo "SKIP_PHASE2 set — skipping P2 controlled."
fi

echo ""
echo "Done. O1: ${P1_OUT_DIR}/p1_baseline_outputs.jsonl"
echo "      O2: ${P2_OUT_DIR}/baseline_outputs.jsonl"
