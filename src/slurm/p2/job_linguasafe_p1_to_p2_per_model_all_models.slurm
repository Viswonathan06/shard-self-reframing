#!/bin/bash
# LinguaSafe P1→P2: run the same rewriter pipeline as job_p1_to_p2_guidelines_qwen35_27b_gpu.slurm,
# once per inference model (each model generates its own P2 from CSV P1s).
# Writes canonical JSONL for downstream DNA jobs:
#   output/Linguasafe_Experiments/p2s/<short_name>_p2.jsonl
# (intermediate checkpoints live under output/Linguasafe_Experiments/p2s/_work_<short_name>/).
#
# Submit alone:
#   sbatch src/slurm/p2/job_linguasafe_p1_to_p2_per_model_all_models.slurm
# Optional:
#   LANGUAGE=zh GUIDELINES_DIR=... sbatch ...
#   EXTRA_MODEL_HUB_DIRS=/path/to/hub:/other/hub  — extra dirs to search for models--* (before download).
#   USE_EXISTING_MODELS_ONLY=1 — never call snapshot_download; skip models not found locally.

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=120:00:00
#SBATCH --job-name=linguasafe_p1p2_all_models
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

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

if [[ -z "${HF_HOME:-}" ]]; then
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
mkdir -p "${HF_HOME}/hub"
HF_HUB="${HF_HOME}/hub"

# Search these hub dirs for models--* before any download (avoids filling a full HF_HOME).
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
if [[ -n "${EXTRA_MODEL_HUB_DIRS:-}" ]]; then
  IFS=':' read -ra _extra_hub <<< "${EXTRA_MODEL_HUB_DIRS}"
  for _eh in "${_extra_hub[@]}"; do
    [[ -n "${_eh}" ]] && _add_hub_root "${_eh}"
  done
fi
echo "Model hub search roots: ${MODEL_HUB_ROOTS[*]}"

export USE_LOCAL_MODEL=true
export VLLM_ONLY="${VLLM_ONLY:-1}"
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-3072}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

LANGUAGE="${LANGUAGE:-en}"
CSV_PATH="${CSV_PATH:-${ROOT_DIR}/dataset/linguasafe.csv}"
GUIDELINES_DIR="${GUIDELINES_DIR:-${ROOT_DIR}/output/guidelines/guidelines_openai_one}"
UNIVERSAL_GUIDELINES="${UNIVERSAL_GUIDELINES:-${ROOT_DIR}/src/prompts/guidelines.txt}"
GUIDELINES_MODE="${GUIDELINES_MODE:-full}"
P2S_OUT_DIR="${P2S_OUT_DIR:-${ROOT_DIR}/output/Linguasafe_Experiments/p2s}"

WANDB_ARG=""
[[ -n "${WANDB:-}" ]] && WANDB_ARG="--wandb"

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
    echo "${path}"
    return 0
  fi
  if [[ -d "${path}/snapshots" ]]; then
    local snap
    snap=$(find "${path}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]] && has_model_weights "${snap}"; then
      echo "${snap}"
      return 0
    fi
  fi
  echo ""
}

# HF repo id -> hub folder name under each hub root (e.g. meta-llama/Llama-3.1-8B-Instruct -> models--meta-llama--Llama-3.1-8B-Instruct)
repo_id_to_hub_dir() {
  local rid="$1"
  echo "models--${rid//\//--}"
}

# Prefer existing installs: optional explicit path (e.g. llama70 snapshot), then every MODEL_HUB_ROOTS entry.
resolve_model_existing() {
  local repo_id="$1"
  local preferred="${2:-}"
  local slug r
  slug=$(repo_id_to_hub_dir "${repo_id}")

  if [[ -n "${preferred}" ]]; then
    r=$(resolve_model_path "${preferred}")
    if [[ -n "${r}" ]]; then
      echo "Using existing model (hint): ${r}" 1>&2
      echo "${r}"
      return 0
    fi
  fi

  local root
  for root in "${MODEL_HUB_ROOTS[@]}"; do
    r=$(resolve_model_path "${root}/${slug}")
    if [[ -n "${r}" ]]; then
      echo "Using existing model: ${r}" 1>&2
      echo "${r}"
      return 0
    fi
  done
  echo ""
  return 1
}

ensure_model() {
  local repo_id="$1"
  local preferred="${2:-}"
  local resolved
  resolved=$(resolve_model_existing "${repo_id}" "${preferred}")
  if [[ -n "${resolved}" ]]; then
    echo "${resolved}"
    return 0
  fi
  if [[ "${USE_EXISTING_MODELS_ONLY:-0}" == "1" ]] || [[ "${HF_SKIP_MODEL_DOWNLOAD:-0}" == "1" ]]; then
    echo "SKIP ${repo_id}: no complete local weights under MODEL_HUB_ROOTS (download disabled)." 1>&2
    echo "" 1>&2
    return 1
  fi
  echo "No local snapshot for ${repo_id}; downloading to ${HF_HUB} ..." 1>&2
  local downloaded
  downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
repo_id = sys.argv[1]
cache_dir = sys.argv[2]
token = os.getenv('HF_TOKEN')
path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, resume_download=True, token=token)
print(path)
" "${repo_id}" "${HF_HUB}") || { echo "" 1>&2; return 1; }
  echo "${downloaded}"
}

run_p1_to_p2_for_model() {
  local short_name="$1"
  local model_path="$2"
  local work_dir="${P2S_OUT_DIR}/_work_${short_name}"
  local final_jsonl="${P2S_OUT_DIR}/${short_name}_p2.jsonl"

  mkdir -p "${work_dir}"

  echo ""
  echo "=========================================="
  echo "P1→P2 model: ${short_name}"
  echo "Local model: ${model_path}"
  echo "Work dir:    ${work_dir}"
  echo "Final file:  ${final_jsonl}"
  echo "=========================================="

  $PYTHON_CMD "${ROOT_DIR}/src/prompt_generation/run_p1_to_p2_guidelines.py" \
    --csv-path "${CSV_PATH}" \
    --language "${LANGUAGE}" \
    --guidelines-mode "${GUIDELINES_MODE}" \
    --guidelines-dir "${GUIDELINES_DIR}" \
    --universal-guidelines "${UNIVERSAL_GUIDELINES}" \
    --output-dir "${work_dir}" \
    --use-local-model \
    --local-model "${model_path}" \
    --temperature "${TEMPERATURE:-0.3}" \
    --max-tokens "${MAX_TOKENS:-512}" \
    ${WANDB_ARG}

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

echo "LinguaSafe P1→P2 per model | $(date) | HOST=$(hostname)"
echo "CSV=${CSV_PATH} LANGUAGE=${LANGUAGE}"
echo "P2S_OUT_DIR=${P2S_OUT_DIR}"

mkdir -p "${P2S_OUT_DIR}"

# Same model list / IDs as DNA multimodel jobs
mp=""
if mp=$(ensure_model "meta-llama/Llama-3.1-8B-Instruct" "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct"); then
  run_p1_to_p2_for_model "llama8b" "${mp}" || true
else echo "SKIP llama8b: model resolve/download failed"; fi

if mp=$(ensure_model "Qwen/Qwen3.5-9B" "${HF_HUB}/models--Qwen--Qwen3.5-9B"); then
  run_p1_to_p2_for_model "qwen35_9b" "${mp}" || true
else echo "SKIP qwen35_9b: model resolve/download failed"; fi

if mp=$(ensure_model "Qwen/Qwen3.5-27B" "${HF_HUB}/models--Qwen--Qwen3.5-27B"); then
  run_p1_to_p2_for_model "qwen35_27b" "${mp}" || true
else echo "SKIP qwen35_27b: model resolve/download failed"; fi

if mp=$(ensure_model "meta-llama/Llama-3.3-70B-Instruct" \
  "${LOCAL_MODEL_SNAPSHOT:-$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}"); then
  run_p1_to_p2_for_model "llama33_70b" "${mp}" || true
else echo "SKIP llama33_70b: model resolve/download failed"; fi

if mp=$(ensure_model "mistralai/Mistral-7B-Instruct-v0.3" "${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3"); then
  run_p1_to_p2_for_model "mistral_7b" "${mp}" || true
else echo "SKIP mistral_7b: model resolve/download failed"; fi

if mp=$(ensure_model "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
  "${HF_HUB}/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506"); then
  run_p1_to_p2_for_model "mistral_24b" "${mp}" || true
else echo "SKIP mistral_24b: model resolve/download failed"; fi

echo ""
echo "All P1→P2 passes finished at: $(date)"
