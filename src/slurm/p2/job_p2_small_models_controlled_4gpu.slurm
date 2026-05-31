#!/usr/bin/bash
# P2 controlled outputs: Llama-8B, Mistral-7B, Qwen3.5-9B, Gemma-3-12B-IT, Llama-3.3-70B (sequential).
# Uses p1_p2_outputs_v4.jsonl and control signals (response_mode; mention_level derived).

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=p2_small_controlled
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || {
  echo "ERROR: Cannot cd to $ROOT_DIR"
  exit 1
}

source ~/.bashrc 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/playpen/huggingface}"
export USE_LOCAL_MODEL=true
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"

P2_INPUT="${P2_INPUT:-${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines_v4/p1_p2_outputs_v4.jsonl}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-${ROOT_DIR}/src/prompts/p2_response_system_prompt_controlled.txt}"

if [[ ! -f "${P2_INPUT}" ]]; then
  echo "ERROR: Input JSONL not found: ${P2_INPUT}"
  exit 1
fi
if [[ ! -f "${SYSTEM_PROMPT_FILE}" ]]; then
  echo "ERROR: System prompt file not found: ${SYSTEM_PROMPT_FILE}"
  exit 1
fi

# Same layout as job_p2_all_models_4gpu: P2_OUTPUT_PARENT=deepseek -> output/deepseek/... (DeepSeek stays out of p2_controlled)
P2_OUTPUT_TAG="${P2_OUTPUT_TAG:-}"
P2_OUTPUT_PARENT="${P2_OUTPUT_PARENT:-}"
if [[ -z "${P2_OUTPUT_PARENT}" ]] && [[ "${P2_OUTPUT_TAG}" == "deepseek" ]]; then
  P2_OUTPUT_PARENT="deepseek"
fi
if [[ -n "${P2_OUTPUT_PARENT}" ]]; then
  P2_OUT_ROOT="${ROOT_DIR}/output/${P2_OUTPUT_PARENT}"
  P2_OUT_SUFFIX=""
elif [[ -n "${P2_OUTPUT_TAG}" ]]; then
  P2_OUT_ROOT="${ROOT_DIR}/output/p2_controlled/guidelines_v4"
  P2_OUT_SUFFIX="_${P2_OUTPUT_TAG}"
else
  P2_OUT_ROOT="${ROOT_DIR}/output/p2_controlled/guidelines_v4"
  P2_OUT_SUFFIX=""
fi

# Python
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
elif [ -f "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
  PYTHON_CMD="python3"
fi

# Use the exact snapshot paths from other working slurm files.
HF_HUB="${HF_HOME}/hub"
LLAMA8B="${LLAMA8B_PATH:-${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659}"
MISTRAL7B="${MISTRAL7B_PATH:-${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a}"
# Qwen3.5 / Gemma: repo roots under hub (ensure_model resolves snapshots/ or downloads)
QWEN9B="${QWEN9B_PATH:-${HF_HUB}/models--Qwen--Qwen3.5-9B}"
GEMMA3_12B="${GEMMA3_12B_PATH:-${HF_HUB}/models--google--gemma-3-12b-it}"
# Llama-3.3-70B: same default as p1_to_p2 guidelines jobs (override with LLAMA33_70B_PATH or LOCAL_MODEL_SNAPSHOT)
LLAMA33_70B="${LLAMA33_70B_PATH:-${LOCAL_MODEL_SNAPSHOT:-$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}}"
mkdir -p "${HF_HUB}"

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

# Ensure model path exists. If missing, download from Hugging Face and return snapshot path.
# Uses HF_TOKEN directly in snapshot_download(token=...) to avoid writing login state.
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

run_model() {
  local name="$1"
  local model_path="$2"
  local out_subdir="$3"
  local resolved
  resolved=$(resolve_model_path "${model_path}")
  if [[ -z "${resolved}" ]]; then
    echo "SKIP ${name}: model path not found/invalid (${model_path})"
    return 0
  fi

  echo "========== ${name} =========="
  echo "Model path: ${resolved}"
  mkdir -p "${P2_OUT_ROOT}/${out_subdir}"

  $PYTHON_CMD src/baselines/generate_p2_controlled_outputs.py \
    --input-jsonl "${P2_INPUT}" \
    --output-dir "${P2_OUT_ROOT}/${out_subdir}" \
    --system-prompt-file "${SYSTEM_PROMPT_FILE}" \
    --use-local-model \
    --local-model "${resolved}" \
    --temperature 0.3 \
    --max-tokens 2000

  echo "Exit ${name}: $?"
  echo ""
}

echo "=== Controlled P2 outputs (8B / Mistral-7B / Qwen3.5-9B / Gemma-3-12B-IT / 70B) ==="
echo "Input: ${P2_INPUT}"
echo "System prompt file: ${SYSTEM_PROMPT_FILE}"
echo "Using vLLM TP=${VLLM_TENSOR_PARALLEL_SIZE} on 4 GPUs"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "WARNING: HF_TOKEN not set. Gated model downloads (Llama/Mistral/Gemma) may fail if not already cached."
fi
echo ""

# Resolve or download models first
LLAMA8B=$(ensure_model "meta-llama/Llama-3.1-8B-Instruct" "${LLAMA8B}") || { echo "SKIP Llama-3.1-8B: download/path failed."; LLAMA8B=""; }
MISTRAL7B=$(ensure_model "mistralai/Mistral-7B-Instruct-v0.3" "${MISTRAL7B}") || { echo "SKIP Mistral-7B-v0.3: download/path failed."; MISTRAL7B=""; }
QWEN9B=$(ensure_model "Qwen/Qwen3.5-9B" "${QWEN9B}") || { echo "SKIP Qwen3.5-9B: download/path failed."; QWEN9B=""; }
GEMMA3_12B=$(ensure_model "google/gemma-3-12b-it" "${GEMMA3_12B}") || { echo "SKIP Gemma-3-12B-IT: download/path failed."; GEMMA3_12B=""; }
LLAMA33=$(ensure_model "meta-llama/Llama-3.3-70B-Instruct" "${LLAMA33_70B}") || { echo "SKIP Llama-3.3-70B: download/path failed."; LLAMA33=""; }

run_model "Llama-3.1-8B" "${LLAMA8B}" "p2_llama8b_controlled${P2_OUT_SUFFIX}"
run_model "Mistral-7B-v0.3" "${MISTRAL7B}" "p2_mistral7b_controlled${P2_OUT_SUFFIX}"
run_model "Qwen3.5-9B" "${QWEN9B}" "p2_qwen9b_controlled${P2_OUT_SUFFIX}"
run_model "Gemma-3-12B-IT" "${GEMMA3_12B}" "p2_gemma3_12b_controlled${P2_OUT_SUFFIX}"
run_model "Llama-3.3-70B-Instruct" "${LLAMA33}" "p2_llama33_70b_controlled${P2_OUT_SUFFIX}"

echo "Done at: $(date)"
