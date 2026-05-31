#!/usr/bin/bash
# P2 controlled outputs (single model): Qwen3.5-9B on Qwen3.5-27B guideline rows (openai_one).
# Same control-signal user prompt as Gemma controlled job; vLLM tensor parallel across 4 GPUs.
#
# Run:
#   sbatch src/slurm/p2/job_qwen35_9b_controlled_openai_one_4gpu.slurm
#
# Optional overrides:
#   P2_INPUT=... P2_OUT_DIR=... HF_HOME=... sbatch ...
#   VLLM_TENSOR_PARALLEL_SIZE=2  (must match #SBATCH --gres=gpu:N)
#   MAX_TOKENS=2000 MAX_RECORDS=10 sbatch ...
#   QWEN35_9B_PATH=/path/to/snapshot sbatch ...
#
# Hugging Face hub disk (e.g. Errno 28 on /playpen/$USER/huggingface):
#   The job always prints df + du of each models--* under HF_HOME/hub.
#   HF_HUB_PRUNE_OTHERS=1 sbatch ...   # rm every hub/models--* except this run's weights
#   HF_KEEP_HUB_DIRS="models--Qwen--Qwen3.5-27B" sbatch ...   # extra basenames to keep (comma-separated)
#   HF_HUB_PRUNE_MIN_FREE_GB=30      # if set, only prune when free space on HF_HOME fs is below this (requires HF_HUB_PRUNE_OTHERS=1)

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=p2_qwen35_9b_ctrl_q27b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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
      echo "NOTE: $HF_HOME not writable; HF_HOME=${HF_HOME}"
    else
      export HF_HOME="${HOME}/.cache/huggingface"
      echo "NOTE: HF_HOME=${HF_HOME} (home cache)."
    fi
  fi
fi

export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
export VLLM_ONLY="${VLLM_ONLY:-1}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export MAX_TOKENS="${MAX_TOKENS:-700}"
export EXIT_ON_NULL="${EXIT_ON_NULL:-1}"

_P2_OPENAI_QWEN27B="${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines_openai_one_qwen35_27b/p1_p2_outputs_openai_one.jsonl"
_P2_OPENAI_QWEN27B_LEGACY="${ROOT_DIR}/output/p1_p2_guidelines_openai_one_qwen35_27b/p1_p2_outputs_openai_one.jsonl"
P2_INPUT="${P2_INPUT:-${_P2_OPENAI_QWEN27B}}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${ROOT_DIR}/src/prompts/p2_response_system_prompt_controlled.txt}"
P2_OUT_DIR="${P2_OUT_DIR:-${ROOT_DIR}/output/p2_qwen35_9b_controlled_openai_one_qwen27b_guidelines}"
WANDB_ARG=""
[[ -n "${WANDB}" ]] && WANDB_ARG="--wandb"
USE_HF_ARG=""
if [[ -n "${USE_HF}" ]]; then
  USE_HF_ARG="--use-hf"
  export VLLM_ONLY=0
fi

if [[ ! -f "${P2_INPUT}" ]]; then
  if [[ -f "${_P2_OPENAI_QWEN27B}" ]]; then
    echo "NOTE: P2_INPUT not found at ${P2_INPUT}; using ${_P2_OPENAI_QWEN27B}"
    P2_INPUT="${_P2_OPENAI_QWEN27B}"
  elif [[ -f "${_P2_OPENAI_QWEN27B_LEGACY}" ]]; then
    echo "NOTE: P2_INPUT not found at ${P2_INPUT}; using legacy path ${_P2_OPENAI_QWEN27B_LEGACY}"
    P2_INPUT="${_P2_OPENAI_QWEN27B_LEGACY}"
  fi
fi
if [[ ! -f "${P2_INPUT}" ]]; then
  echo "ERROR: Input JSONL not found: ${P2_INPUT}"
  echo "Tried: ${_P2_OPENAI_QWEN27B}"
  echo "       ${_P2_OPENAI_QWEN27B_LEGACY}"
  exit 1
fi
if [[ ! -f "${SYSTEM_PROMPT_FILE}" ]]; then
  echo "ERROR: System prompt file not found: ${SYSTEM_PROMPT_FILE}"
  exit 1
fi

if [[ -n "${CONDA_PYTHON}" ]]; then
  PYTHON_CMD="${CONDA_PYTHON}"
elif [[ -n "${USE_CONDA}" ]] && [ -f "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_CMD="$CONDA_PREFIX/bin/python"
elif [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
elif [ -f "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
  PYTHON_CMD="python3"
fi

echo "=== GPU diagnostics (job: ${SLURM_JOB_ID:-N/A}) ==="
nvidia-smi 2>/dev/null || true

HF_HUB="${HF_HOME}/hub"
mkdir -p "${HF_HUB}"

# --- Hub usage report + optional prune of other cached models (frees space before download) ---
report_hf_hub_models() {
  echo "=== Hugging Face hub: ${HF_HUB} ==="
  if [[ -d "${HF_HOME}" ]]; then
    df -h "${HF_HOME}" 2>/dev/null || df -h "$(dirname "${HF_HOME}")" 2>/dev/null || true
  fi
  if [[ ! -d "${HF_HUB}" ]]; then
    echo "(no hub dir yet)"
    echo "=== end hub report ==="
    return 0
  fi
  local d
  for d in "${HF_HUB}"/models--*; do
    [[ -e "${d}" ]] || continue
    du -sh "${d}" 2>/dev/null || true
  done
  echo "=== end hub report ==="
}

# Basename under hub for the model this job needs (must match HF cache layout).
HF_HUB_KEEP_QWEN9B="models--Qwen--Qwen3.5-9B"

should_prune_hf_hub() {
  if ! [[ "${HF_HUB_PRUNE_OTHERS:-0}" =~ ^(1|true|yes|YES)$ ]]; then
    return 1
  fi
  local min_gb="${HF_HUB_PRUNE_MIN_FREE_GB:-}"
  if [[ -z "${min_gb}" ]]; then
    return 0
  fi
  local avail_kb
  if [[ -d "${HF_HOME}" ]]; then
    avail_kb=$(df -Pk "${HF_HOME}" 2>/dev/null | awk 'NR==2 {print $4}')
  else
    avail_kb=$(df -Pk "${HF_HUB}" 2>/dev/null | awk 'NR==2 {print $4}')
  fi
  if [[ -z "${avail_kb}" ]] || ! [[ "${avail_kb}" =~ ^[0-9]+$ ]]; then
    echo "WARNING: could not read free space; pruning anyway (HF_HUB_PRUNE_OTHERS=1)."
    return 0
  fi
  # avail_kb * 1024 bytes; threshold in GB
  local avail_gb=$((avail_kb / 1024 / 1024))
  if (( avail_gb < min_gb )); then
    echo "NOTE: free space on HF_HOME fs ≈ ${avail_gb} GiB (< ${min_gb} GiB); pruning other hub models."
    return 0
  fi
  echo "NOTE: free space on HF_HOME fs ≈ ${avail_gb} GiB (>= ${min_gb} GiB); skip prune."
  return 1
}

prune_hf_hub_except_kept() {
  local keep_extra="${HF_KEEP_HUB_DIRS:-}"
  echo "=== HF hub prune: removing other models under ${HF_HUB} (keeping ${HF_HUB_KEEP_QWEN9B}) ==="
  local d base
  for d in "${HF_HUB}"/models--*; do
    [[ -d "${d}" ]] || continue
    base=$(basename "${d}")
    if [[ "${base}" == "${HF_HUB_KEEP_QWEN9B}" ]]; then
      echo "KEEP ${base}"
      continue
    fi
    local skip=0
    local token
    IFS=',' read -ra _toks <<< "${keep_extra}"
    for token in "${_toks[@]}"; do
      token=$(echo "${token}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
      [[ -z "${token}" ]] && continue
      if [[ "${base}" == "${token}" ]]; then
        echo "KEEP ${base} (HF_KEEP_HUB_DIRS)"
        skip=1
        break
      fi
    done
    if [[ "${skip}" -eq 1 ]]; then
      continue
    fi
    echo "REMOVE ${d}"
    rm -rf "${d}"
  done
  echo "=== hub prune done ==="
}

report_hf_hub_models
if should_prune_hf_hub; then
  prune_hf_hub_except_kept
  report_hf_hub_models
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

QWEN35_9B_REPO="${QWEN35_9B_REPO:-Qwen/Qwen3.5-9B}"
QWEN35_9B_HINT="${QWEN35_9B_PATH:-${HF_HUB}/models--Qwen--Qwen3.5-9B}"
QWEN35_9B=$(ensure_model "${QWEN35_9B_REPO}" "${QWEN35_9B_HINT}") || {
  echo "ERROR: Could not resolve or download ${QWEN35_9B_REPO}"
  exit 1
}

echo "Input: ${P2_INPUT}"
echo "System prompt: ${SYSTEM_PROMPT_FILE}"
echo "Output dir: ${P2_OUT_DIR}"
echo "Model: ${QWEN35_9B}"
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
  --local-model "${QWEN35_9B}" \
  --temperature 0.3 \
  --max-tokens "${MAX_TOKENS}" \
  "${MAX_REC_ARGS[@]}" \
  ${USE_HF_ARG} \
  ${WANDB_ARG}

exit $?
