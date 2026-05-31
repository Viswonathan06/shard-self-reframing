#!/usr/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=p2_llama8b_dna_ctrl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export USE_LOCAL_MODEL=true
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_TRUST_REMOTE_CODE=1

if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

resolve_model_path() {
  local path="$1"
  if [[ -d "${path}" ]] && [[ -f "${path}/config.json" ]]; then
    echo "${path}"; return 0
  fi
  if [[ -d "${path}/snapshots" ]]; then
    local snap
    snap=$(find "${path}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then
      echo "${snap}"; return 0
    fi
  fi
  echo ""
}

# Try shared hub first (read-only), then download to user-owned dir
SHARED_HUB="$HF_HOME"
USER_HUB="/playpen/${USER}/huggingface/hub"
LLAMA8B_RESOLVED=$(resolve_model_path "${SHARED_HUB}/models--meta-llama--Llama-3.1-8B-Instruct")

if [[ -z "${LLAMA8B_RESOLVED}" ]]; then
  echo "Llama-3.1-8B not in shared hub — downloading to ${USER_HUB}..."
  mkdir -p "${USER_HUB}"
  LLAMA8B_RESOLVED=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    cache_dir=sys.argv[1],
    token=os.getenv('HF_TOKEN')
)
print(path)
" "${USER_HUB}")
  if [[ $? -ne 0 ]] || [[ -z "${LLAMA8B_RESOLVED}" ]]; then
    echo "ERROR: Failed to download Llama-3.1-8B-Instruct"
    exit 1
  fi
  echo "Downloaded to: ${LLAMA8B_RESOLVED}"
fi

INPUT_JSONL="${ROOT_DIR}/output/p1_p2_dna_llama70b_gpt5cat.jsonl"
OUTPUT_DIR="${ROOT_DIR}/output/p2_controlled/donotanswer/p2_llama8b_dna_ctrl_gpt5cat"
SYSTEM_PROMPT="${ROOT_DIR}/src/prompts/p2_response_system_prompt_controlled.txt"

echo "=== Llama-3.1-8B controlled output (DoNotAnswer / gpt5cat) ==="
echo "Model:  ${LLAMA8B_RESOLVED}"
echo "Input:  ${INPUT_JSONL}"
echo "Output: ${OUTPUT_DIR}"
echo "System prompt: ${SYSTEM_PROMPT}"
nvidia-smi 2>/dev/null || true
echo ""

mkdir -p "${OUTPUT_DIR}"

$PYTHON_CMD src/baselines/generate_p2_controlled_outputs.py \
  --input-jsonl "${INPUT_JSONL}" \
  --output-dir  "${OUTPUT_DIR}" \
  --system-prompt-file "${SYSTEM_PROMPT}" \
  --use-local-model \
  --local-model "${LLAMA8B_RESOLVED}" \
  --temperature 0.3 \
  --max-tokens 2000

EXIT_CODE=$?
echo "Done at: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
