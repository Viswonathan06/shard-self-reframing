#!/usr/bin/bash
# P1→P2 guidelines with Qwen3.5-27B (same Python args as job_p1_to_p2_guidelines_gpu.slurm).
# Defaults: universal src/prompts/guidelines.txt + category .txt under output/guidelines/guidelines_openai_one
# (OpenAI-generated category guidelines). Override: GUIDELINES_DIR=.../output/guidelines/guidelines_v4 sbatch ...
# Recent cluster runs (e.g. job 5852) used this file; previously GUIDELINES_DIR pointed at guidelines_v4.
# Default 4 GPUs / TP=4 (Qwen3.5-27B has 24 heads: TP must divide 24). If Slurm grants fewer GPUs than TP,
# model_client clamps TP to torch.cuda.device_count() (fixes device=4,num_gpus=4 when TP was 6).
# For 6 GPUs: #SBATCH --gres=gpu:6 and VLLM_TENSOR_PARALLEL_SIZE=6 sbatch ...
# If vLLM OOM or foreign PID on a GPU, use #SBATCH --exclusive or adjust VLLM_GPU_MEMORY_UTILIZATION / VLLM_MAX_MODEL_LEN.
# Hugging Face has Qwen/Qwen3.5-27B only (no separate *-Instruct repo as of 2026-03).
# Override checkpoint: QWEN35_27B_PATH=... or QWEN35_27B_REPO=org/name sbatch ...
# Writable playpen cache (default if shared hub is read-only): PLAYPEN_USER_HF=/playpen/$USER/huggingface
# vLLM-only run for Qwen3.5: install once in this repo's .venv (pip metadata will warn — ignore if inference works):
#   pip install 'vllm==0.18.0'
#   pip install --upgrade --no-deps 'transformers>=5.2,<6' 'huggingface-hub>=1.5'
# Then submit this job. VLLM_ONLY=1 skips the slow HuggingFace pipeline fallback.

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --job-name=p1_to_p2_qwen35_27b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || {
    echo "ERROR: Cannot cd to $ROOT_DIR"
    exit 1
}

source ~/.bashrc 2>/dev/null || true

LOGS_DIR="${ROOT_DIR}/logs"
ARCHIVE_DIR="${LOGS_DIR}/archive"
mkdir -p "$ARCHIVE_DIR"

if [ -d "$LOGS_DIR" ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    ARCHIVE_SUBDIR="${ARCHIVE_DIR}/${TIMESTAMP}"
    mkdir -p "$ARCHIVE_SUBDIR"
    if find "$LOGS_DIR" -maxdepth 1 -type f \( -name "*.out" -o -name "*.err" \) | grep -q .; then
        echo "Archiving existing logs to ${ARCHIVE_SUBDIR}..."
        find "$LOGS_DIR" -maxdepth 1 -type f \( -name "*.out" -o -name "*.err" \) -exec mv {} "$ARCHIVE_SUBDIR/" \;
        echo "✓ Archived existing logs"
    fi
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
# snapshot_download writes under HF_HOME/hub. Shared /playpen/huggingface is often not user-writable.
# If so, use a personal playpen tree (no $HOME quota): PLAYPEN_USER_HF default /playpen/$USER/huggingface.
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
  else
    PLAYPEN_USER_HF="${PLAYPEN_USER_HF:-/playpen/${USER}/huggingface}"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
      echo "NOTE: $HF_HOME not writable; HF_HOME=${HF_HOME} (playpen user cache)."
    else
      echo "ERROR: Need a writable Hugging Face cache on playpen. Could not create/use:"
      echo "  ${PLAYPEN_USER_HF}/hub"
      echo "Ask your admin for a writable /playpen/${USER} (or set HF_HOME to any writable playpen path) then resubmit."
      exit 1
    fi
  fi
fi
mkdir -p "${HF_HOME}/hub"
export USE_LOCAL_MODEL=true
export VLLM_ONLY="${VLLM_ONLY:-1}"
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
# Match Slurm GPU count so every allocated GPU participates in tensor parallel.
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
# If a GPU still has another process (nvidia-smi), lower util and/or max len so KV init fits (OOM often shows a foreign PID).
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
# Lower (e.g. 2048) if KV init OOM persists; raise on a clean allocation if prompts truncate.
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-3072}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
echo "vLLM: TP=${VLLM_TENSOR_PARALLEL_SIZE} gpu_mem_util=${VLLM_GPU_MEMORY_UTILIZATION} max_model_len=${VLLM_MAX_MODEL_LEN} VLLM_ONLY=${VLLM_ONLY} TRUST_REMOTE_CODE=${VLLM_TRUST_REMOTE_CODE}"

if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    source "${ROOT_DIR}/.venv/bin/activate"
    PYTHON_CMD="python"
    echo "Using project .venv (numpy<2.4 for vLLM)"
elif command -v python &> /dev/null && [[ $(which python) == *"anaconda3"* ]]; then
    PYTHON_CMD="python"
elif [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    PYTHON_CMD="python3"
fi

echo "=== GPU diagnostics (user: $USER, job: ${SLURM_JOB_ID:-N/A}) ==="
nvidia-smi 2>/dev/null || true
echo "=== end GPU diagnostics ==="

HF_HUB="${HF_HOME}/hub"
QWEN35_27B_REPO="${QWEN35_27B_REPO:-Qwen/Qwen3.5-27B}"
# Repo root under hub (snapshots resolved below); override with full snapshot path via QWEN35_27B_PATH
QWEN35_27B_HINT="${QWEN35_27B_PATH:-${HF_HUB}/models--Qwen--Qwen3.5-27B}"

resolve_model_path() {
  local path="$1"
  if [[ -d "${path}" ]] && [[ -f "${path}/config.json" ]]; then
    echo "${path}"
    return 0
  fi
  if [[ -d "${path}/snapshots" ]]; then
    local snap
    snap=$(find "${path}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
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
  echo "Downloading: ${repo_id} -> ${HF_HUB}" 1>&2
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

MODEL_SNAPSHOT=$(ensure_model "${QWEN35_27B_REPO}" "${QWEN35_27B_HINT}") || {
  echo "ERROR: Could not resolve or download ${QWEN35_27B_REPO}"
  exit 1
}
echo "Using model: ${MODEL_SNAPSHOT}"

cleanup() { true; }
trap cleanup EXIT ERR
LANGUAGE=${LANGUAGE:-en}

GUIDELINES_DIR=${GUIDELINES_DIR:-${ROOT_DIR}/output/guidelines/guidelines_openai_one}
UNIVERSAL_GUIDELINES=${UNIVERSAL_GUIDELINES:-${ROOT_DIR}/src/prompts/guidelines.txt}
OUT_SUFFIX="_qwen35_27b"
if [[ "${GUIDELINES_DIR}" == *"guidelines_v4"* ]]; then
    P1_P2_OUT="${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines_v4${OUT_SUFFIX}"
elif [[ "${GUIDELINES_DIR}" == *"guidelines_v3"* ]]; then
    P1_P2_OUT="${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines_v3${OUT_SUFFIX}"
elif [[ "${GUIDELINES_DIR}" == *"guidelines_v2"* ]]; then
    P1_P2_OUT="${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines_v2${OUT_SUFFIX}"
elif [[ "${GUIDELINES_DIR}" == *"guidelines_openai_one"* ]]; then
    P1_P2_OUT="${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines_openai_one${OUT_SUFFIX}"
else
    P1_P2_OUT="${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines${OUT_SUFFIX}"
fi
[ -d "${P1_P2_OUT}" ] || mkdir -p "${P1_P2_OUT}"
echo "Guidelines dir: ${GUIDELINES_DIR}"
echo "Universal guidelines: ${UNIVERSAL_GUIDELINES}"
echo "P1→P2 output dir: ${P1_P2_OUT}"

WANDB_ARG=""
[ -n "${WANDB}" ] && WANDB_ARG="--wandb"

$PYTHON_CMD src/prompt_generation/run_p1_to_p2_guidelines.py \
    --csv-path "${ROOT_DIR}/dataset/linguasafe.csv" \
    --language "${LANGUAGE}" \
    --guidelines-dir "${GUIDELINES_DIR}" \
    --universal-guidelines "${UNIVERSAL_GUIDELINES}" \
    --output-dir "${P1_P2_OUT}" \
    --use-local-model \
    --local-model "${MODEL_SNAPSHOT}" \
    --temperature 0.3 \
    --max-tokens 512 \
    ${WANDB_ARG}

EXIT_CODE=$?
cleanup
exit $EXIT_CODE
