#!/usr/bin/bash
# DoNotAnswer P1→P2: each model generates its own reframed prompts from the DNA dataset.
# Mirrors job_linguasafe_p1_to_p2_per_model_all_models.slurm but uses the DNA JSONL as input.
# Writes per-model JSONL to: output/DNA_New_Experiments/p2s/<short_name>_p2.jsonl
# (intermediate checkpoints live under output/DNA_New_Experiments/p2s/_work_<short_name>/).
#
# Submit:
#   sbatch src/slurm/p2/job_dna_p1_to_p2_per_model_all_models.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --job-name=dna_p1_to_p2_per_model
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
mkdir -p "${ROOT_DIR}/logs"

export ROOT_DIR
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

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

declare -a MODEL_HUB_ROOTS=()
_add_hub_root() {
  local d="$1"
  [[ -z "${d}" ]] || [[ ! -d "${d}" ]] && return 0
  local x
  for x in "${MODEL_HUB_ROOTS[@]}"; do
    [[ "${x}" == "${d}" ]] && return 0
  done
  MODEL_HUB_ROOTS+=("${d}")
}
_add_hub_root "${HF_HUB}"
_add_hub_root "$HF_HOME"
_add_hub_root "${HOME:-}/.cache/huggingface/hub"
echo "Model hub search roots: ${MODEL_HUB_ROOTS[*]}"

export USE_LOCAL_MODEL=true
export VLLM_ONLY="${VLLM_ONLY:-1}"
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-3072}"

DNA_JSONL="${ROOT_DIR}/dataset/donotanswer_no_outputs.jsonl"
GUIDELINES_DIR="${ROOT_DIR}/output/guidelines/guidelines_openai_one"
UNIVERSAL_GUIDELINES="${ROOT_DIR}/src/prompts/guidelines.txt"
P2S_OUT_DIR="${ROOT_DIR}/output/DNA_New_Experiments/p2s"

if [[ ! -f "${DNA_JSONL}" ]]; then
  echo "ERROR: DNA dataset not found: ${DNA_JSONL}"; exit 1
fi

has_model_weights() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 1
  if [[ -f "${dir}/model.safetensors" ]] || [[ -f "${dir}/model.safetensors.index.json" ]] || [[ -f "${dir}/pytorch_model.bin" ]]; then
    return 0
  fi
  compgen -G "${dir}/model-*.safetensors" >/dev/null
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

repo_id_to_hub_dir() {
  local rid="$1"
  echo "models--${rid//\//--}"
}

resolve_model_existing() {
  local repo_id="$1"
  local preferred="${2:-}"
  local slug r
  slug=$(repo_id_to_hub_dir "${repo_id}")
  if [[ -n "${preferred}" ]]; then
    r=$(resolve_model_path "${preferred}")
    if [[ -n "${r}" ]]; then echo "Using existing model (hint): ${r}" 1>&2; echo "${r}"; return 0; fi
  fi
  local root
  for root in "${MODEL_HUB_ROOTS[@]}"; do
    r=$(resolve_model_path "${root}/${slug}")
    if [[ -n "${r}" ]]; then echo "Using existing model: ${r}" 1>&2; echo "${r}"; return 0; fi
  done
  echo ""; return 1
}

ensure_model() {
  local repo_id="$1"
  local preferred="${2:-}"
  local resolved
  resolved=$(resolve_model_existing "${repo_id}" "${preferred}") && echo "${resolved}" && return 0 || true
  echo "No local snapshot for ${repo_id}; downloading to ${HF_HUB} ..." 1>&2
  local downloaded
  downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2],
                         resume_download=True, token=os.getenv('HF_TOKEN'))
print(path)
" "${repo_id}" "${HF_HUB}") || { echo "" 1>&2; return 1; }
  echo "${downloaded}"
}

run_p1_to_p2_for_model() {
  local short_name="$1"
  local model_path="$2"
  local work_dir="${P2S_OUT_DIR}/_work_${short_name}"
  local final_jsonl="${P2S_OUT_DIR}/${short_name}_p2.jsonl"

  if [[ -f "${final_jsonl}" ]]; then
    local n
    n=$(wc -l < "${final_jsonl}" 2>/dev/null || echo 0)
    if [[ "${n}" -ge 939 ]]; then
      echo "SKIP ${short_name}: final JSONL already has ${n} rows (>= 939)"
      return 0
    fi
    echo "Partial ${short_name}: ${n}/939 rows — resuming from checkpoint"
  fi

  mkdir -p "${work_dir}"
  echo ""
  echo "=========================================="
  echo "P1→P2 model: ${short_name}"
  echo "Local model: ${model_path}"
  echo "Work dir:    ${work_dir}"
  echo "Final file:  ${final_jsonl}"
  echo "=========================================="

  $PYTHON_CMD "${ROOT_DIR}/src/prompt_generation/run_p1_to_p2_guidelines.py" \
    --jsonl-path "${DNA_JSONL}" \
    --language "en" \
    --guidelines-mode full \
    --guidelines-dir "${GUIDELINES_DIR}" \
    --universal-guidelines "${UNIVERSAL_GUIDELINES}" \
    --output-dir "${work_dir}" \
    --use-local-model \
    --local-model "${model_path}" \
    --temperature 0.3 \
    --max-tokens 512

  local produced=""
  for cand in \
    "${work_dir}/p1_p2_outputs_openai_one.jsonl" \
    "${work_dir}/p1_p2_outputs.jsonl" \
    "${work_dir}/p1_p2_outputs_v4.jsonl" \
    "${work_dir}/p1_p2_outputs_v3.jsonl" \
    "${work_dir}/p1_p2_outputs_v2.jsonl"; do
    if [[ -f "${cand}" ]]; then
      produced="${cand}"
      break
    fi
  done
  if [[ -z "${produced}" ]]; then
    echo "ERROR: No p1_p2_outputs*.jsonl under ${work_dir} for ${short_name}"
    return 1
  fi
  cp -f "${produced}" "${final_jsonl}"
  echo "Wrote ${final_jsonl} (from ${produced})"
}

run_p1_to_p2_repo() {
  local name="$1"
  local repo_id="$2"
  local preferred="${3:-}"
  local mp
  mp=$(ensure_model "${repo_id}" "${preferred}") \
    || { echo "SKIP ${name}: model resolve/download failed"; return 1; }
  run_p1_to_p2_for_model "${name}" "${mp}"
}

echo "DNA P1→P2 per model | $(date) | HOST=$(hostname)"
echo "DNA input: ${DNA_JSONL}"
echo "P2S output dir: ${P2S_OUT_DIR}"
mkdir -p "${P2S_OUT_DIR}"

purge_model_cache() {
  local path="$1"
  if [[ "${path}" == "${HF_HUB}/models--"* ]] && [[ -d "${path}" ]]; then
    echo "Purging cache: ${path}"
    rm -rf "${path}" 2>/dev/null || true
  fi
}

purge_model_cache "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct"
run_p1_to_p2_repo "llama8b" "meta-llama/Llama-3.1-8B-Instruct" \
  "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct" || true
purge_model_cache "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct"

run_p1_to_p2_repo "qwen35_9b" "Qwen/Qwen3.5-9B" \
  "${HF_HUB}/models--Qwen--Qwen3.5-9B" || true
purge_model_cache "${HF_HUB}/models--Qwen--Qwen3.5-9B"

run_p1_to_p2_repo "qwen35_27b" "Qwen/Qwen3.5-27B" \
  "${HF_HUB}/models--Qwen--Qwen3.5-27B" || true
purge_model_cache "${HF_HUB}/models--Qwen--Qwen3.5-27B"

run_p1_to_p2_repo "llama33_70b" "meta-llama/Llama-3.3-70B-Instruct" \
  "${LOCAL_MODEL_SNAPSHOT:-$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}" || true

run_p1_to_p2_repo "mistral_7b" "mistralai/Mistral-7B-Instruct-v0.3" \
  "${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3" || true
purge_model_cache "${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3"

run_p1_to_p2_repo "mistral_24b" "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
  "${HF_HUB}/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506" || true
purge_model_cache "${HF_HUB}/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506"

echo ""
echo "All DNA P1→P2 passes finished at: $(date)"
