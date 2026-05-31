#!/usr/bin/bash
# O1 baseline: send p1 directly to Gemma-3-27B-IT (no control signals, no p2).
# Produces p1_baseline_outputs.jsonl for use as O1 in evaluation.
#
# Run:
#   sbatch src/slurm/p2/job_gemma3_27b_it_p1_baseline_openai_one_4gpu.slurm
#
# Optional overrides:
#   P1_INPUT=... P1_OUT_DIR=... MAX_TOKENS=2000 sbatch ...
#   MAX_RECORDS=1 sbatch ...   (only first N rows of the input JSONL)
#   VLLM_TENSOR_PARALLEL_SIZE=4 VLLM_GPU_MEMORY_UTILIZATION=0.80 VLLM_MAX_MODEL_LEN=8192 sbatch ...
#   GEMMA3_27B_PATH=/path/to/snapshot sbatch ...

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH --job-name=p1_baseline_gemma3_27b_it
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ---------- HF cache ----------
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
  else
    PLAYPEN_USER_HF="${PLAYPEN_USER_HF:-/playpen/${USER}/huggingface}"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" 2>/dev/null && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
    else
      export HF_HOME="${HOME}/.cache/huggingface"
    fi
  fi
fi

# ---------- vLLM / model config ----------
export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
export VLLM_ONLY="${VLLM_ONLY:-1}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export MAX_TOKENS="${MAX_TOKENS:-2000}"
export EXIT_ON_NULL="${EXIT_ON_NULL:-1}"

# ---------- Paths ----------
P1_INPUT="${P1_INPUT:-${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines_openai_one_qwen35_27b/p1_p2_outputs_openai_one.jsonl}"
P1_OUT_DIR="${P1_OUT_DIR:-${ROOT_DIR}/output/p1_baseline_gemma3_27b_it_openai_one}"

WANDB_ARG=""
[[ -n "${WANDB}" ]] && WANDB_ARG="--wandb"
USE_HF_ARG=""
if [[ -n "${USE_HF}" ]]; then
  USE_HF_ARG="--use-hf"
  export VLLM_ONLY=0
fi

if [[ ! -f "${P1_INPUT}" ]]; then
  echo "ERROR: Input JSONL not found: ${P1_INPUT}"
  exit 1
fi

# ---------- Python ----------
if [[ -n "${CONDA_PYTHON}" ]]; then
  PYTHON_CMD="${CONDA_PYTHON}"
  echo "Using CONDA_PYTHON=${PYTHON_CMD}"
elif [[ -n "${USE_CONDA}" ]] && [ -f "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_CMD="$CONDA_PREFIX/bin/python"
  echo "Using conda python: ${PYTHON_CMD}"
elif [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
  echo "Using project .venv"
else
  PYTHON_CMD="python3"
fi

# ---------- Resolve Gemma model path ----------
echo "=== GPU diagnostics (user: $USER, job: ${SLURM_JOB_ID:-N/A}) ==="
nvidia-smi 2>/dev/null || true
echo "=== end GPU diagnostics ==="

HF_HUB="${HF_HOME}/hub"
mkdir -p "${HF_HUB}"

GEMMA3_27B="${GEMMA3_27B_PATH:-${HF_HUB}/models--google--gemma-3-27b-it}"

resolve_model_path() {
  local path="$1"
  if [[ -d "${path}" ]] && [[ -f "${path}/config.json" ]]; then
    echo "${path}"; return 0
  fi
  if [[ -d "${path}/snapshots" ]]; then
    local snap
    snap=$(find "${path}/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then
      echo "${snap}"; return 0
    fi
  fi
  echo ""
}

ensure_model() {
  local repo_id="$1" path="$2" resolved
  resolved=$(resolve_model_path "${path}")
  if [[ -n "${resolved}" ]]; then echo "${resolved}"; return 0; fi
  echo "Downloading: ${repo_id} -> ${HF_HUB}" 1>&2
  local downloaded
  downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2],
                         resume_download=True, token=os.getenv('HF_TOKEN'))
print(path)
" "${repo_id}" "${HF_HUB}")
  if [[ $? -ne 0 ]] || [[ -z "${downloaded}" ]]; then return 1; fi
  echo "${downloaded}"
}

GEMMA3_27B=$(ensure_model "google/gemma-3-27b-it" "${GEMMA3_27B}") || {
  echo "ERROR: Could not resolve or download google/gemma-3-27b-it"; exit 1
}

# ---------- Run ----------
echo ""
echo "=== P1 baseline generation (O1) ==="
echo "Input:      ${P1_INPUT}"
echo "Output dir: ${P1_OUT_DIR}"
echo "Model:      ${GEMMA3_27B}"
echo "vLLM: TP=${VLLM_TENSOR_PARALLEL_SIZE} gpu_mem=${VLLM_GPU_MEMORY_UTILIZATION} max_len=${VLLM_MAX_MODEL_LEN} max_tokens=${MAX_TOKENS} EXIT_ON_NULL=${EXIT_ON_NULL}"
echo ""

mkdir -p "${P1_OUT_DIR}"

MAX_RECORDS="${MAX_RECORDS:-}"
MAX_REC_ARGS=()
if [[ -n "${MAX_RECORDS}" ]]; then
  MAX_REC_ARGS+=(--max-records "${MAX_RECORDS}")
fi

$PYTHON_CMD src/baselines/generate_p1_baseline_outputs.py \
  --input-jsonl "${P1_INPUT}" \
  --output-dir "${P1_OUT_DIR}" \
  --use-local-model \
  --local-model "${GEMMA3_27B}" \
  --temperature 0.3 \
  --max-tokens "${MAX_TOKENS}" \
  "${MAX_REC_ARGS[@]}" \
  ${USE_HF_ARG} \
  ${WANDB_ARG}

exit $?
