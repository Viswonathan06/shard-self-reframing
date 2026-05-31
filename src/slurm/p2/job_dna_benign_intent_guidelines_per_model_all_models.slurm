#!/usr/bin/bash
# DoNotAnswer benign_intent + guidelines using per-model p2s.
# Each model responds to its own self-generated reframed prompts (from DNA p2s dir).
# Mirrors the LinguaSafe benign_intent_guidelines run.
#
# Prerequisite: job_dna_p1_to_p2_per_model_all_models.slurm must have completed.
#   Per-model p2s must exist at: output/DNA_New_Experiments/p2s/<model>_p2.jsonl
#
# Output goes to a NEW directory to avoid picking up checkpoints from the old
# dna_benign_intent_guidelines_multimodel run (which used the shared llama70b p2).
#   output/DNA_New_Experiments/dna_benign_intent_guidelines_per_model_multimodel/<model>/
#
# Submit:
#   sbatch src/slurm/p2/job_dna_benign_intent_guidelines_per_model_all_models.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=dna_benign_intent_guidelines_per_model
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
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

# Verify model_client is intact
if ! $PYTHON_CMD -c "from src.utils.model_client import call_model, ContentFilteredError" 2>/dev/null; then
  echo "ERROR: model_client.py import check failed — aborting"
  exit 1
fi
echo "model_client import OK"

if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
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

DNA_P2S_DIR="${ROOT_DIR}/output/DNA_New_Experiments/p2s"
EXPERIMENT_DIR="${ROOT_DIR}/output/DNA_New_Experiments/dna_benign_intent_guidelines_per_model_multimodel"
SYSTEM_PROMPT="${ROOT_DIR}/src/prompts/p1_benign_intent_guidelines_system_prompt.txt"

if [[ ! -f "${SYSTEM_PROMPT}" ]]; then
  echo "ERROR: System prompt not found: ${SYSTEM_PROMPT}"; exit 1
fi

has_model_weights() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 1
  if [[ -f "${dir}/model.safetensors" ]] || [[ -f "${dir}/model.safetensors.index.json" ]] || [[ -f "${dir}/pytorch_model.bin" ]]; then
    return 0
  fi
  compgen -G "${dir}/model-*.safetensors" >/dev/null
}

purge_model_cache() {
  local path="$1"
  if [[ "${path}" == "${HF_HUB}/models--"* ]] && [[ -d "${path}" ]]; then
    echo "Purging cache: ${path}"
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
  purge_model_cache "${path}"
  echo "Downloading ${repo_id}..." 1>&2
  local downloaded
  downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2],
                         resume_download=True, force_download=True, token=os.getenv('HF_TOKEN'))
print(path)
" "${repo_id}" "${HF_HUB}" 2>/dev/null)
  if [[ $? -ne 0 ]] || [[ -z "${downloaded}" ]] || ! has_model_weights "${downloaded}"; then
    purge_model_cache "${path}"
    echo "" 1>&2; return 1
  fi
  echo "${downloaded}"
}

run_model() {
  local name="$1"
  local model_path="$2"
  local out_dir="${EXPERIMENT_DIR}/${name}"
  local input_jsonl="${DNA_P2S_DIR}/${name}_p2.jsonl"

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
    echo "ERROR: per-model p2 not found: ${input_jsonl}"
    echo "       Run job_dna_p1_to_p2_per_model_all_models.slurm first."
    return 1
  fi

  mkdir -p "${out_dir}"

  export VLLM_TENSOR_PARALLEL_SIZE=4
  export VLLM_GPU_MEMORY_UTILIZATION=0.85

  $PYTHON_CMD src/baselines/generate_p1_baseline_outputs.py \
    --input-jsonl        "${input_jsonl}" \
    --output-dir         "${out_dir}" \
    --system-prompt-file "${SYSTEM_PROMPT}" \
    --use-local-model \
    --local-model        "${model_path}" \
    --temperature 0.3 \
    --max-tokens 2000 \
    --exit-on-null

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

echo "DNA benign_intent_guidelines (per-model p2s) at: $(date)"
echo "Input p2s dir: ${DNA_P2S_DIR}"
echo "Output dir:    ${EXPERIMENT_DIR}"

purge_model_cache "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct"
run_model_repo "llama8b" "meta-llama/Llama-3.1-8B-Instruct" \
  "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct" || true
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
