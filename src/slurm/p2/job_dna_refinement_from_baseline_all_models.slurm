#!/usr/bin/bash
# Refinement-from-baseline experiment (DoNotAnswer): all 6 per-model p2 models.
# Uses existing O1 baseline outputs from helpful_assistant_multimodel.
# Skips the baseline phase — O1 already exists. Does join + refinement only.
#
# When called from job_dna_per_model_experiments_sequential.slurm (or directly
# with LINGUASAFE_EXPERIMENTS_ROOT set), outputs go to:
#   ${LINGUASAFE_EXPERIMENTS_ROOT}/refinement_from_baseline_multimodel/{model}/
#
# Standalone (no env vars), outputs to:
#   output/DNA_Per_Model_Experiments/refinement_from_baseline_multimodel/{model}/
#
# SYSTEM_PROMPT env var may be overridden by the caller (e.g. guidelines variant).

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --job-name=dna_refine_from_baseline_all_models
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

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

# WandB
LINGUASAFE_EXPERIMENTS_ROOT="${LINGUASAFE_EXPERIMENTS_ROOT:-}"
if [[ -n "${LINGUASAFE_EXPERIMENTS_ROOT}" ]]; then
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-refinement_from_baseline_multimodel}"
  EXPERIMENT_DIR="${EXPERIMENT_DIR:-${LINGUASAFE_EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}}"
  export WANDB_TAGS="${EXPERIMENT_NAME},refinement_from_baseline,per_model"
else
  EXPERIMENT_NAME="${EXPERIMENT_NAME:-refinement_from_baseline_multimodel}"
  EXPERIMENT_DIR="${EXPERIMENT_DIR:-${ROOT_DIR}/output/DNA_Per_Model_Experiments/${EXPERIMENT_NAME}}"
  export WANDB_TAGS="${EXPERIMENT_NAME},refinement_from_baseline,donotanswer"
fi
export WANDB_PROJECT="multi-lingual-alignment-llms"
export WANDB_RUN_GROUP="${EXPERIMENT_NAME}"

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ -f "${ROOT_DIR}/.wandb_key" ]]; then
  export WANDB_API_KEY=$(cat "${ROOT_DIR}/.wandb_key")
fi
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: WANDB_API_KEY not set — WandB disabled."
  export WANDB_MODE=disabled
fi

# HF cache
if [[ -z "${HF_HOME:-}" ]]; then
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
  compgen -G "${dir}/model-*.safetensors" >/dev/null
}

purge_model_cache() {
  local path="$1"
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
    cat "${dl_err}" 1>&2; rm -f "${dl_err}"; return 1
  fi
  if ! has_model_weights "${downloaded}"; then
    echo "Downloaded snapshot for ${repo_id} is incomplete." 1>&2
    purge_model_cache "${path}"; rm -f "${dl_err}"; return 1
  fi
  rm -f "${dl_err}"
  echo "${downloaded}"
}

# ============================================================
# Paths
# ============================================================
# Per-model p2 files (produced by job_dna_p1_to_p2_per_model_all_models.slurm)
P2S_DIR="${LINGUASAFE_P2S_DIR:-${ROOT_DIR}/output/DNA_New_Experiments/p2s}"

# O1 source: helpful_assistant baseline for DNA per-model experiments
O1_ROOT="${LINGUASAFE_EXPERIMENTS_ROOT:-${ROOT_DIR}/output/DNA_Per_Model_Experiments}/helpful_assistant_multimodel"

# Refinement system prompt (may be overridden by guidelines caller)
SYSTEM_PROMPT="${SYSTEM_PROMPT:-${ROOT_DIR}/src/prompts/p2_response_system_prompt_refinement.txt}"

echo "======================================================"
echo "DNA Refinement-from-Baseline — All Models"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "Experiment:    ${EXPERIMENT_NAME}"
echo "Output root:   ${EXPERIMENT_DIR}"
echo "O1 source:     ${O1_ROOT}"
echo "P2s dir:       ${P2S_DIR}"
echo "System prompt: ${SYSTEM_PROMPT}"
echo "======================================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || true

# ============================================================
# run_refinement_from_baseline
# Join existing O1 with per-model p2, then run refinement.
# Args: $1=short_name  $2=resolved_model_path
# ============================================================
run_refinement_from_baseline() {
  local name="$1"
  local model_path="$2"
  local out_dir="${EXPERIMENT_DIR}/${name}"
  local refinement_input="${out_dir}/refinement_input.jsonl"
  local refined_dir="${out_dir}"  # baseline_outputs.jsonl goes directly in out_dir
  local o1_file="${O1_ROOT}/${name}/p1_baseline_outputs.jsonl"
  local p2_file="${P2S_DIR}/${name}_p2.jsonl"

  echo ""
  echo "=========================================="
  echo "=== Model: ${name}  | $(date)"
  echo "=== Path:  ${model_path}"
  echo "=== O1:    ${o1_file}"
  echo "=== P2:    ${p2_file}"
  echo "=== Out:   ${out_dir}"
  echo "=========================================="

  if [[ ! -f "${o1_file}" ]]; then
    echo "ERROR: O1 not found: ${o1_file}"; return 1
  fi
  if [[ ! -f "${p2_file}" ]]; then
    echo "ERROR: per-model p2 not found: ${p2_file}"; return 1
  fi

  mkdir -p "${out_dir}"

  # ----------------------------------------------------------
  # Phase 1: Join O1 with per-model p2 records → refinement_input.jsonl
  # (skip if already done and non-empty)
  # ----------------------------------------------------------
  local n_p2
  n_p2=$(wc -l < "${p2_file}")
  local n_ri=0
  [[ -f "${refinement_input}" ]] && n_ri=$(wc -l < "${refinement_input}")

  if [[ "${n_ri}" -ge "${n_p2}" ]]; then
    echo "Phase 1 (join): already done (${n_ri} rows). Skipping."
  else
    echo "Phase 1: Joining O1 with p2 records (${n_p2} rows)..."
    O1_FILE="${o1_file}" P2_FILE="${p2_file}" OUT_FILE="${refinement_input}" \
    $PYTHON_CMD - <<'PYEOF'
import json, os

o1_file  = os.environ["O1_FILE"]
p2_file  = os.environ["P2_FILE"]
out_file = os.environ["OUT_FILE"]

def norm(s):
    return " ".join((s or "").split()).strip()

# Index O1 by p1_id and by normalised p1 text
o1_by_id = {}
o1_by_p1 = {}
with open(o1_file) as f:
    for line in f:
        r = json.loads(line)
        pid = r.get("p1_id")
        p1k = norm(r.get("p1", ""))
        out = r.get("output", "")
        if pid is not None:
            o1_by_id[str(pid)] = out
        if p1k:
            o1_by_p1[p1k] = out

written = skipped = 0
with open(p2_file) as fin, open(out_file, "w") as fout:
    for line in fin:
        r = json.loads(line)
        pid = r.get("p1_id")
        p1k = norm(r.get("p1", ""))
        baseline = None
        if pid is not None and str(pid) in o1_by_id:
            baseline = o1_by_id[str(pid)]
        elif p1k and p1k in o1_by_p1:
            baseline = o1_by_p1[p1k]
        if baseline is None:
            skipped += 1
            continue
        r["baseline_output"] = baseline
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        written += 1

print(f"Joined {written} records -> {out_file}  (skipped {skipped})")
PYEOF
    if [[ $? -ne 0 ]]; then
      echo "ERROR: Join step failed for ${name}"; return 1
    fi
  fi

  # ----------------------------------------------------------
  # Phase 2: Refinement — P1 + P2 + O1 → O2
  # (skip if already done and non-empty)
  # ----------------------------------------------------------
  local refined_file="${refined_dir}/baseline_outputs.jsonl"
  local n_ref=0
  [[ -f "${refined_file}" ]] && n_ref=$(wc -l < "${refined_file}")
  local n_ri_now
  n_ri_now=$(wc -l < "${refinement_input}")

  if [[ "${n_ref}" -ge "${n_ri_now}" ]]; then
    echo "Phase 2 (refinement): already done (${n_ref} rows). Skipping."
    return 0
  fi

  echo "Phase 2: Refinement inference (${n_ri_now} rows)..."
  export VLLM_TENSOR_PARALLEL_SIZE=4
  export VLLM_GPU_MEMORY_UTILIZATION=0.85
  unset VLLM_ONLY  2>/dev/null || true
  unset VLLM_MAX_MODEL_LEN 2>/dev/null || true
  export WANDB_JOB_TYPE="refinement"
  export WANDB_RUN_NAME="refinement_from_baseline_${name}"

  $PYTHON_CMD src/baselines/generate_p2_controlled_outputs.py \
    --input-jsonl        "${refinement_input}" \
    --output-dir         "${refined_dir}" \
    --system-prompt-file "${SYSTEM_PROMPT}" \
    --use-local-model \
    --local-model        "${model_path}" \
    --temperature 0.3 \
    --max-tokens 2000 \
    --refinement-mode \
    --exit-on-null \
    --wandb

  local exit_code=$?
  echo "Refinement done: ${name} | Exit: ${exit_code} | $(date)"
  return ${exit_code}
}

run_model_repo() {
  local name="$1"
  local repo_id="$2"
  local cache_path="$3"
  local model_path
  model_path=$(ensure_model "${repo_id}" "${cache_path}") \
    || { echo "SKIP ${name}: model resolve/download failed"; return 1; }
  run_refinement_from_baseline "${name}" "${model_path}"
}

# ============================================================
# Run all 6 models consecutively
# ============================================================
SUCCESSFUL=(); FAILED=()

run_one() {
  local name="$1"; local repo="$2"; local cache="$3"
  if run_model_repo "${name}" "${repo}" "${cache}"; then
    SUCCESSFUL+=("${name}")
  else
    echo "FAILED: ${name}"; FAILED+=("${name}")
  fi
  # Free cache after each model so next download fits
  [[ "${cache}" == "${HF_HUB}/"* ]] && purge_model_cache "${cache}" || true
}

run_one "llama8b" "meta-llama/Llama-3.1-8B-Instruct" \
  "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct"

run_one "qwen35_9b" "Qwen/Qwen3.5-9B" \
  "${HF_HUB}/models--Qwen--Qwen3.5-9B"

run_one "qwen35_27b" "Qwen/Qwen3.5-27B" \
  "${HF_HUB}/models--Qwen--Qwen3.5-27B"

run_one "llama33_70b" "meta-llama/Llama-3.3-70B-Instruct" \
  "${HF_HUB}/models--meta-llama--Llama-3.3-70B-Instruct"

run_one "mistral_7b" "mistralai/Mistral-7B-Instruct-v0.3" \
  "${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3"

run_one "mistral_24b" "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
  "${HF_HUB}/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506"

echo ""
echo "======================================================"
echo "Done: ${#SUCCESSFUL[@]}/6 models | $(date)"
[[ ${#SUCCESSFUL[@]} -gt 0 ]] && printf '  OK:   %s\n' "${SUCCESSFUL[@]}"
[[ ${#FAILED[@]} -gt 0 ]]    && printf '  FAIL: %s\n' "${FAILED[@]}"
echo "======================================================"

[[ ${#FAILED[@]} -gt 0 ]] && exit 1 || exit 0
