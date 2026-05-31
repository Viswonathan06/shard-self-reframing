#!/usr/bin/bash
# Benign intent + guidelines experiment (DoNotAnswer): Llama-3.1-8B, Qwen3.5-9B, Qwen3.5-27B,
# Llama-3.3-70B, Mistral-7B, Mistral-Small-3.2-24B — consecutive inference across all models.
# Each model responds directly to P1 using a "assume benign intent + guidelines" system prompt.
# No P2, no O1, no baseline join — inference only.
# WandB: runs grouped under WANDB_RUN_GROUP=dna_benign_intent_guidelines_multimodel.

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --job-name=dna_benign_intent_guidelines
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$PROJECT_ROOT"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export ROOT_DIR
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE=1

if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

# WandB (set LINGUASAFE_EXPERIMENTS_ROOT for LinguaSafe output paths + tags)
LINGUASAFE_EXPERIMENTS_ROOT="${LINGUASAFE_EXPERIMENTS_ROOT:-}"
if [[ -n "${LINGUASAFE_EXPERIMENTS_ROOT}" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-linguasafe_benign_intent_guidelines_multimodel}"
  EXPERIMENT_DIR="${EXPERIMENT_DIR:-${LINGUASAFE_EXPERIMENTS_ROOT}/benign_intent_guidelines_multimodel}"
  export WANDB_TAGS="${EXPERIMENT_NAME},benign_intent,guidelines,linguasafe"
else
  EXPERIMENT_NAME="dna_benign_intent_guidelines_multimodel"
  EXPERIMENT_DIR="${ROOT_DIR}/output/dna_benign_intent_guidelines_multimodel"
  export WANDB_TAGS="${EXPERIMENT_NAME},benign_intent,guidelines,donotanswer"
fi
export WANDB_PROJECT="multi-lingual-alignment-llms"
export WANDB_RUN_GROUP="${EXPERIMENT_NAME}"

if [[ -z "${WANDB_API_KEY}" ]] && [[ -f "${ROOT_DIR}/.wandb_key" ]]; then
  export WANDB_API_KEY=$(cat "${ROOT_DIR}/.wandb_key")
fi
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WARNING: WANDB_API_KEY not set and .wandb_key not found — WandB logging will be disabled."
  export WANDB_MODE=disabled
fi

# HF cache
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d /playpen/huggingface ]] && [[ -w /playpen/huggingface ]]; then
    export HF_HOME=/playpen/huggingface
  else
    PLAYPEN_USER_HF="/playpen/${USER}/huggingface"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" 2>/dev/null && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
    else
      export HF_HOME="${HOME}/.cache/huggingface"
    fi
  fi
fi
HF_HUB="${HF_HOME}/hub"
mkdir -p "${HF_HUB}"

has_model_weights() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 1
  if [[ -f "${dir}/model.safetensors" ]] || [[ -f "${dir}/model.safetensors.index.json" ]] || [[ -f "${dir}/pytorch_model.bin" ]]; then
    return 0
  fi
  # Accept sharded safetensors naming (e.g., model-00001-of-00004.safetensors)
  compgen -G "${dir}/model-*.safetensors" >/dev/null
}

purge_model_cache() {
  local path="$1"
  # Safety: only delete per-model caches under this job's HF hub.
  if [[ "${path}" == "${HF_HUB}/models--"* ]] && [[ -d "${path}" ]]; then
    echo "Purging possibly corrupt cache at ${path}" 1>&2
    rm -rf "${path}" 2>/dev/null || true
  fi
}

resolve_model_path() {
  local path="$1"
  if [[ -d "${path}" ]] && [[ -f "${path}/config.json" ]] && has_model_weights "${path}"; then
    echo "${path}"; return 0
  fi
  if [[ -d "${path}/snapshots" ]]; then
    local snap
    snap=$(find "${path}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]] && has_model_weights "${snap}"; then
      echo "${snap}"; return 0
    fi
  fi
  echo ""
}

ensure_model() {
  local repo_id="$1"
  local path="$2"
  local resolved
  resolved=$(resolve_model_path "${path}")
  if [[ -n "${resolved}" ]]; then echo "${resolved}"; return 0; fi
  # If a model dir exists but does not resolve, it is likely partial/corrupt.
  # Purge before download so we do not reuse broken shards.
  purge_model_cache "${path}"
  echo "Downloading ${repo_id} to ${HF_HUB}..." 1>&2
  local downloaded
  local dl_err
  dl_err=$(mktemp)
  downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2],
                         resume_download=True, force_download=True, token=os.getenv('HF_TOKEN'))
print(path)
" "${repo_id}" "${HF_HUB}" 2>"${dl_err}")
  if [[ $? -ne 0 ]] || [[ -z "${downloaded}" ]]; then
    if grep -q "No space left on device" "${dl_err}"; then
      echo "HF cache full while downloading ${repo_id}; purging only this model and retrying once..." 1>&2
      purge_model_cache "${path}"
      downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2],
                         resume_download=True, force_download=True, token=os.getenv('HF_TOKEN'))
print(path)
" "${repo_id}" "${HF_HUB}" 2>"${dl_err}")
      if [[ $? -ne 0 ]] || [[ -z "${downloaded}" ]]; then
        rm -f "${dl_err}"
        echo "" 1>&2
        return 1
      fi
    else
      rm -f "${dl_err}"
      echo "" 1>&2
      return 1
    fi
  fi
  if ! has_model_weights "${downloaded}"; then
    echo "Downloaded snapshot for ${repo_id} is incomplete (missing model weights)." 1>&2
    purge_model_cache "${path}"
    rm -f "${dl_err}"
    echo "" 1>&2
    return 1
  fi
  rm -f "${dl_err}"
  echo "${downloaded}"
}

# ============================================================
# Paths
# ============================================================
P1P2_JSONL="${ROOT_DIR}/output/p1_p2_dna_llama70b_gpt5cat.jsonl"
LINGUASAFE_P2S_DIR="${LINGUASAFE_P2S_DIR:-}"
SYSTEM_PROMPT="${ROOT_DIR}/src/prompts/p1_benign_intent_guidelines_system_prompt.txt"

if [[ -z "${LINGUASAFE_P2S_DIR}" ]]; then
  if [[ ! -f "${P1P2_JSONL}" ]]; then
    echo "ERROR: input file not found: ${P1P2_JSONL}"; exit 1
  fi
fi

# ============================================================
# run_model: P1 → model (benign intent + guidelines) → output
# Args: $1=short_name  $2=resolved_model_path
# ============================================================
run_model() {
  local name="$1"
  local model_path="$2"
  local out_dir="${EXPERIMENT_DIR}/${name}"
  local input_jsonl="${P1P2_JSONL}"
  if [[ -n "${LINGUASAFE_P2S_DIR}" ]]; then
    input_jsonl="${LINGUASAFE_P2S_DIR%/}/${name}_p2.jsonl"
  fi

  echo ""
  echo "=========================================="
  echo "=== Model: ${name}"
  echo "=== Path:  ${model_path}"
  echo "=== Output: ${out_dir}"
  echo "=== Input:  ${input_jsonl}"
  echo "=========================================="
  nvidia-smi 2>/dev/null || true
  echo ""

  if [[ ! -f "${input_jsonl}" ]]; then
    echo "ERROR: P1/P2 jsonl not found for ${name}: ${input_jsonl}"
    return 1
  fi

  mkdir -p "${out_dir}"

  export VLLM_TENSOR_PARALLEL_SIZE=4
  export VLLM_GPU_MEMORY_UTILIZATION=0.85
  unset VLLM_ONLY
  unset VLLM_MAX_MODEL_LEN
  export WANDB_JOB_TYPE="inference"
  export WANDB_RUN_NAME="benign_intent_guidelines_${name}"

  $PYTHON_CMD src/baselines/generate_p1_baseline_outputs.py \
    --input-jsonl        "${input_jsonl}" \
    --output-dir         "${out_dir}" \
    --system-prompt-file "${SYSTEM_PROMPT}" \
    --use-local-model \
    --local-model        "${model_path}" \
    --temperature 0.3 \
    --max-tokens 2000 \
    --exit-on-null \
    --wandb

  local exit_code=$?
  echo "Done: ${name} | Exit: ${exit_code} | $(date)"
  return ${exit_code}
}

run_model_repo() {
  local name="$1"
  local repo_id="$2"
  local cache_path="$3"
  local model_path

  model_path=$(ensure_model "${repo_id}" "${cache_path}") \
    || { echo "SKIP ${name}: model resolve/download failed"; return 1; }

  run_model "${name}" "${model_path}"
}

# ============================================================
# Run consecutively (resolve/download each model right before use)
# ============================================================
# Force a clean llama8b download each run (avoids stale/corrupt shard reuse).
purge_model_cache "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct"
run_model_repo "llama8b" "meta-llama/Llama-3.1-8B-Instruct" \
  "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct" || true
# Free HF hub space for subsequent model downloads.
purge_model_cache "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct"

run_model_repo "qwen35_9b" "Qwen/Qwen3.5-9B" \
  "${HF_HUB}/models--Qwen--Qwen3.5-9B" || true
purge_model_cache "${HF_HUB}/models--Qwen--Qwen3.5-9B"

run_model_repo "qwen35_27b" "Qwen/Qwen3.5-27B" \
  "${HF_HUB}/models--Qwen--Qwen3.5-27B" || true
purge_model_cache "${HF_HUB}/models--Qwen--Qwen3.5-27B"

run_model_repo "llama33_70b" "meta-llama/Llama-3.3-70B-Instruct" \
  "${LOCAL_MODEL_SNAPSHOT:-$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}" || true

run_model_repo "mistral_7b" "mistralai/Mistral-7B-Instruct-v0.3" \
  "${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3" || true
purge_model_cache "${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3"

run_model_repo "mistral_24b" "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
  "${HF_HUB}/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506" || true
purge_model_cache "${HF_HUB}/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506"

echo ""
echo "All models done at: $(date)"
