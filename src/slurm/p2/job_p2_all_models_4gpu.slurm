#!/usr/bin/bash
# P2 → outputs: Llama-8B, Mistral-7B, Qwen3.5-9B, Gemma-3-12B-IT in one job (sequential).
# Uses general system prompt: "You are a helpful assistant. Provide a clear, helpful response to the user's question."
# Saves p1, p2, and each model's output together in baseline_outputs.jsonl per model.
# Same vLLM setup across 4 GPUs (tensor parallel). Downloads models from Hugging Face if not present.

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=p2_all_models
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
# Qwen3.5 needs remote code; PyPI vLLM often wants transformers>=5.2 — see appendix/setup_utilities/install_vllm_qwen35_venv.sh
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"

# Hugging Face token required for gated models (Llama, Mistral, Gemma). Set in ~/.bashrc or: HF_TOKEN=... sbatch ...
if [[ -z "${HF_TOKEN}" ]]; then
  echo "WARNING: HF_TOKEN not set. Gated models (Llama, Mistral, Gemma) may fail to download."
  echo "Set HF_TOKEN in ~/.bashrc or run: HF_TOKEN=your_token sbatch ..."
fi

# P2 input: use legacy P2s from output/p1_to_p2/ by default (override with P2_INPUT if needed).
P2_INPUT="${P2_INPUT:-${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines/p1_p2_outputs.jsonl}"
if [[ ! -f "${P2_INPUT}" ]]; then
  echo "ERROR: P2 input not found: ${P2_INPUT}"
  echo "Expected P2s at: output/p1_to_p2/p1_p2_guidelines/p1_p2_outputs.jsonl"
  echo "Set P2_INPUT to the correct path if using a different location."
  exit 1
fi
echo "Using P2 input: ${P2_INPUT}"

# Optional: separate outputs from default P2 runs.
# - P2_OUTPUT_TAG=foo -> output/p2_baselines/p2_llama8b_foo, ... (legacy suffix)
# - P2_OUTPUT_TAG=deepseek OR P2_OUTPUT_PARENT=deepseek -> output/deepseek/... (DeepSeek runs stay here; not under p2_baselines)
P2_OUTPUT_TAG="${P2_OUTPUT_TAG:-}"
P2_OUTPUT_PARENT="${P2_OUTPUT_PARENT:-}"
if [[ -z "${P2_OUTPUT_PARENT}" ]] && [[ "${P2_OUTPUT_TAG}" == "deepseek" ]]; then
  P2_OUTPUT_PARENT="deepseek"
fi
if [[ -n "${P2_OUTPUT_PARENT}" ]]; then
  P2_OUT_ROOT="${ROOT_DIR}/output/${P2_OUTPUT_PARENT}"
  P2_OUT_SUFFIX=""
elif [[ -n "${P2_OUTPUT_TAG}" ]]; then
  P2_OUT_ROOT="${ROOT_DIR}/output/p2_baselines"
  P2_OUT_SUFFIX="_${P2_OUTPUT_TAG}"
else
  P2_OUT_ROOT="${ROOT_DIR}/output/p2_baselines"
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

HF="${HF_HOME}/hub"
mkdir -p "${HF}"

# Ensure model is on disk: use existing path/HF cache if valid, else download from Hugging Face.
# Usage: ensure_model "repo_id" "optional_existing_path"
# Prints the snapshot path to stdout; returns 0 on success, 1 on failure.
repo_to_hub_dir() {
  local repo_id="$1"
  local org="${repo_id%%/*}"
  local name="${repo_id##*/}"
  echo "${HF}/models--${org}--${name}"
}

resolve_model_path() {
  local path="$1"
  if [[ -z "${path}" ]]; then
    echo ""
    return 0
  fi
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
  local existing="$2"
  local resolved
  resolved=$(resolve_model_path "${existing}")
  if [[ -n "${resolved}" ]]; then
    echo "${resolved}"
    return 0
  fi

  # Also check the standard HF_HOME cache location regardless of existing-path hint.
  local hub_path
  hub_path=$(repo_to_hub_dir "${repo_id}")
  resolved=$(resolve_model_path "${hub_path}")
  if [[ -n "${resolved}" ]]; then
    echo "${resolved}"
    return 0
  fi

  echo "Downloading ${repo_id} to ${HF}..." 1>&2
  local path
  path=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download, login
import os
import sys
repo_id = sys.argv[1]
cache_dir = sys.argv[2]
token = os.getenv('HF_TOKEN')
if token:
    login(token=token)
try:
    path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, resume_download=True)
    print(path)
except Exception as e:
    print(f'DOWNLOAD_FAILED: {e}', file=sys.stderr)
    sys.exit(1)
" "$repo_id" "$HF")
  if [[ $? -ne 0 ]] || [[ -z "${path}" ]] || [[ "${path}" == DOWNLOAD_FAILED* ]]; then
    echo "" 1>&2
    return 1
  fi
  echo "${path}"
  return 0
}

echo "=== P2 → outputs (all 4 models) ==="
echo "P2 input: ${P2_INPUT}"
echo "Cache: ${HF}"
echo "Output dirs: ${P2_OUT_ROOT}/p2_llama8b${P2_OUT_SUFFIX}, ${P2_OUT_ROOT}/p2_mistral7b${P2_OUT_SUFFIX}, ${P2_OUT_ROOT}/p2_qwen9b${P2_OUT_SUFFIX}, ${P2_OUT_ROOT}/p2_gemma3_12b${P2_OUT_SUFFIX}"
echo "System prompt: You are a helpful assistant. Provide a clear, helpful response to the user's question."
echo ""

# Default model paths: prefer HF_HOME/hub cache; override with *_PATH if needed.
LLAMA8B="${LLAMA8B_PATH:-${HF}/models--meta-llama--Llama-3.1-8B-Instruct}"
MISTRAL7B="${MISTRAL7B_PATH:-${HF}/models--mistralai--Mistral-7B-Instruct-v0.2}"
QWEN9B="${QWEN9B_PATH:-${HF}/models--Qwen--Qwen3.5-9B}"
GEMMA3_12B="${GEMMA3_12B_PATH:-${HF}/models--google--gemma-3-12b-it}"

# Resolve to snapshot if needed; download only if path missing
LLAMA8B=$(ensure_model "meta-llama/Llama-3.1-8B-Instruct" "${LLAMA8B}") || { echo "SKIP Llama-3.1-8B: download or path failed."; LLAMA8B=""; }
MISTRAL7B=$(ensure_model "mistralai/Mistral-7B-Instruct-v0.2" "${MISTRAL7B}") || { echo "SKIP Mistral-7B: download or path failed."; MISTRAL7B=""; }
QWEN9B=$(ensure_model "Qwen/Qwen3.5-9B" "${QWEN9B}") || { echo "SKIP Qwen3.5-9B: download or path failed."; QWEN9B=""; }
GEMMA3_12B=$(ensure_model "google/gemma-3-12b-it" "${GEMMA3_12B}") || { echo "SKIP Gemma-3-12B-IT: download or path failed."; GEMMA3_12B=""; }

run_model() {
  local name="$1"
  local model_path="$2"
  local out_subdir="$3"
  if [[ -z "${model_path}" ]] || [[ ! -d "${model_path}" ]] || [[ ! -f "${model_path}/config.json" ]]; then
    echo "SKIP ${name}: no valid path."
    return 0
  fi
  echo "========== ${name} =========="
  export LOCAL_MODEL="${model_path}"
  mkdir -p "${P2_OUT_ROOT}/${out_subdir}"
  $PYTHON_CMD src/baselines/generate_p2_baseline_outputs.py \
    --input-jsonl "${P2_INPUT}" \
    --output-dir "${P2_OUT_ROOT}/${out_subdir}" \
    --use-local-model \
    --local-model "${model_path}"
  echo "Exit ${name}: $?"
  echo ""
}

# Run all four models sequentially (each uses 4 GPUs via vLLM TP)
run_model "Llama-3.1-8B"   "${LLAMA8B}"   "p2_llama8b${P2_OUT_SUFFIX}"
run_model "Mistral-7B"     "${MISTRAL7B}" "p2_mistral7b${P2_OUT_SUFFIX}"
run_model "Qwen3.5-9B"     "${QWEN9B}"   "p2_qwen9b${P2_OUT_SUFFIX}"
run_model "Gemma-3-12B-IT" "${GEMMA3_12B}" "p2_gemma3_12b${P2_OUT_SUFFIX}"

echo "Done at: $(date)"
