#!/usr/bin/bash
# P2 controlled outputs (single model): Gemma-3-27B-IT.
# Uses control signals (response_mode; mention_level derived from it) in the P1→P2 JSONL.
#
# Run:
#   sbatch src/slurm/p2/job_gemma3_27b_it_controlled_openai_one_4gpu.slurm
#
# Optional overrides:
#   Default P2_INPUT: output/p1_p2_llama_controlled.jsonl (override for guidelines/Qwen P1→P2 JSONL).
#   P2_INPUT=... P2_OUT_DIR=... HF_HOME=... HF_TOKEN=... sbatch ...
#   MAX_TOKENS=2000 VLLM_TENSOR_PARALLEL_SIZE=2 VLLM_GPU_MEMORY_UTILIZATION=0.80 VLLM_MAX_MODEL_LEN=8192 sbatch ...
#   MAX_RECORDS=1 sbatch ...   (only first N rows of P2_INPUT)
#   EXIT_ON_NULL=0 sbatch ...   (default: stop job on first null output; unset writes null rows and continues)
#   GEMMA3_27B_PATH=/path/to/snapshot  (or keep default HF cache resolution)
#   GEMMA3_CLEAR_HUB_CACHE=1 sbatch ...   (rm hub copy under HF_HOME, skip shared snapshot, re-download)
#
# If vLLM fails with: ValueError: Following weights were not initialized from checkpoint
# (many language_model.model.layers.*): your HF snapshot is usually incomplete/corrupt.
# Fix: GEMMA3_CLEAR_HUB_CACHE=1 once, or set GEMMA3_27B_PATH to a known-good snapshot;
# otherwise this script prefers a readable shared /playpen/huggingface snapshot when GEMMA3_27B_PATH is unset.
# Engine init flakiness: VLLM_ENABLE_V1_MULTIPROCESSING=0 (see job_evaluate_safety_helpfulness.slurm).

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=p2_gemma3_27b_it_ctrl_openai_one
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# Hugging Face cache: prefer shared /playpen/huggingface if writable; otherwise use a per-user playpen cache.
# (Fixes permission errors like: .../hub/models--*/refs/main: [Errno 13] Permission denied)
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
export MAX_TOKENS="${MAX_TOKENS:-2000}"
# First null completion exits Python with 1 (Slurm shows FAILED); row is not checkpointed. Set EXIT_ON_NULL=0 to disable.
export EXIT_ON_NULL="${EXIT_ON_NULL:-1}"

# Set to 1 once (here or GEMMA3_CLEAR_HUB_CACHE=1 sbatch) to rm HF hub copy and re-download; use 0 for normal reruns.
GEMMA3_CLEAR_HUB_CACHE="${GEMMA3_CLEAR_HUB_CACHE:-0}"

P2_INPUT="${P2_INPUT:-${ROOT_DIR}/output/p1_p2_llama_controlled.jsonl}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${ROOT_DIR}/src/prompts/p2_response_system_prompt_controlled.txt}"
P2_OUT_DIR="${P2_OUT_DIR:-${ROOT_DIR}/output/p2_controlled/qwen27b/p2_gemma3_27b_it_controlled_openai_one}"
WANDB_ARG=""
[[ -n "${WANDB}" ]] && WANDB_ARG="--wandb"
USE_HF_ARG=""
if [[ -n "${USE_HF}" ]]; then
  USE_HF_ARG="--use-hf"
  # Allow HF backend to run even if VLLM_ONLY was set externally
  export VLLM_ONLY=0
fi

if [[ ! -f "${P2_INPUT}" ]]; then
  echo "ERROR: Input JSONL not found: ${P2_INPUT}"
  exit 1
fi
if [[ ! -f "${SYSTEM_PROMPT_FILE}" ]]; then
  echo "ERROR: System prompt file not found: ${SYSTEM_PROMPT_FILE}"
  exit 1
fi

# Python
# If you want conda (recommended for your setup), pass one of:
#   CONDA_PYTHON=/path/to/python sbatch ...
#   USE_CONDA=1 sbatch ...    (uses $CONDA_PREFIX/bin/python as fallback)
if [[ -n "${CONDA_PYTHON}" ]]; then
  PYTHON_CMD="${CONDA_PYTHON}"
  echo "Using CONDA_PYTHON=${PYTHON_CMD}"
elif [[ -n "${USE_CONDA}" ]] && [ -f "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_CMD="$CONDA_PREFIX/bin/python"
  echo "Using conda python: ${PYTHON_CMD}"
elif [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
  echo "Using project .venv (numpy<2.4 for vLLM)"
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

# Wipe local hub entry and force a full download into HF_HOME (skips shared snapshot for this run).
if [[ "${GEMMA3_CLEAR_HUB_CACHE:-0}" =~ ^(1|true|yes|YES)$ ]]; then
  _GEMMA_HUB_DIR="${HF_HUB}/models--google--gemma-3-27b-it"
  if [[ -d "${_GEMMA_HUB_DIR}" ]]; then
    echo "GEMMA3_CLEAR_HUB_CACHE=1: removing ${_GEMMA_HUB_DIR}"
    rm -rf "${_GEMMA_HUB_DIR}"
  else
    echo "GEMMA3_CLEAR_HUB_CACHE=1: no directory at ${_GEMMA_HUB_DIR} (nothing to remove)"
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
    echo "" 1>&2
    return 1
  fi
  echo "${downloaded}"
  return 0
}

if [[ -z "${HF_TOKEN}" ]]; then
  echo "WARNING: HF_TOKEN not set. Gated Gemma download may fail if not already cached."
fi

# Prefer readable shared snapshot when GEMMA3_27B_PATH is unset (avoids partial user-cache weights).
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

echo "Input: ${P2_INPUT}"
echo "System prompt: ${SYSTEM_PROMPT_FILE}"
echo "Output dir: ${P2_OUT_DIR}"
echo "Model: ${GEMMA3_27B}"
echo "vLLM: TP=${VLLM_TENSOR_PARALLEL_SIZE} gpu_mem_util=${VLLM_GPU_MEMORY_UTILIZATION} max_model_len=${VLLM_MAX_MODEL_LEN} max_new_tokens=${MAX_TOKENS} EXIT_ON_NULL=${EXIT_ON_NULL}"
echo ""

mkdir -p "${P2_OUT_DIR}"

MAX_RECORDS="${MAX_RECORDS:-}"
MAX_REC_ARGS=()
if [[ -n "${MAX_RECORDS}" ]]; then
  MAX_REC_ARGS+=(--max-records "${MAX_RECORDS}")
fi

$PYTHON_CMD src/baselines/generate_p2_controlled_outputs.py \
  --input-jsonl "${P2_INPUT}" \
  --output-dir "${P2_OUT_DIR}" \
  --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
  --use-local-model \
  --local-model "${GEMMA3_27B}" \
  --temperature 0.3 \
  --max-tokens "${MAX_TOKENS}" \
  "${MAX_REC_ARGS[@]}" \
  ${USE_HF_ARG} \
  ${WANDB_ARG}

exit $?

