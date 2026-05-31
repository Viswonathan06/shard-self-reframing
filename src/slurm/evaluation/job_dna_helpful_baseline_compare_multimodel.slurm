#!/usr/bin/bash
# Compare helpful-assistant baseline vs target DNA outputs per model.
# Default target: benign intent outputs.
# Override for guidelines:
#   EVAL_NAME=benign_intent_guidelines \
#   VARIANT_ROOT=$PROJECT_ROOT/output/dna_benign_intent_guidelines_multimodel \
#   sbatch src/slurm/evaluation/job_dna_helpful_baseline_compare_multimodel.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --job-name=dna_eval_helpful_vs_target
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$PROJECT_ROOT"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export ROOT_DIR
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

# WandB
EXPERIMENT_NAME="dna_evaluation_multimodel"
export WANDB_PROJECT="multi-lingual-alignment-llms"
export WANDB_RUN_GROUP="${EXPERIMENT_NAME}"
export WANDB_TAGS="${EXPERIMENT_NAME},judge,gemma4-31b,donotanswer"

if [[ -z "${WANDB_API_KEY}" ]] && [[ -f "${ROOT_DIR}/.wandb_key" ]]; then
  export WANDB_API_KEY=$(cat "${ROOT_DIR}/.wandb_key")
fi
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WARNING: WANDB_API_KEY not set and .wandb_key not found — WandB logging disabled."
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

ensure_model() {
  local repo_id="$1"
  local path="$2"
  local resolved
  resolved=$(resolve_model_path "${path}")
  if [[ -n "${resolved}" ]]; then echo "${resolved}"; return 0; fi
  echo "Downloading missing judge model: ${repo_id} -> ${HF_HUB}" 1>&2
  local downloaded
  downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2],
                         resume_download=True, token=os.getenv('HF_TOKEN'))
print(path)
" "${repo_id}" "${HF_HUB}")
  if [[ $? -ne 0 ]] || [[ -z "${downloaded}" ]]; then
    return 1
  fi
  echo "${downloaded}"
}

JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-google/gemma-4-31B-it}"
JUDGE_HINT="${JUDGE_MODEL_PATH:-${HF_HUB}/models--google--gemma-4-31B-it}"
JUDGE_RESOLVED=$(ensure_model "${JUDGE_MODEL_ID}" "${JUDGE_HINT}") || {
  echo "ERROR: Could not resolve judge model ${JUDGE_MODEL_ID}"
  exit 1
}
echo "Judge model: ${JUDGE_RESOLVED}"

BASELINE_ROOT="${BASELINE_ROOT:-${ROOT_DIR}/output/dna_helpful_assistant_multimodel}"
VARIANT_ROOT="${VARIANT_ROOT:-${ROOT_DIR}/output/dna_benign_intent_multimodel}"
EVAL_NAME="${EVAL_NAME:-benign_intent}"
EVAL_ROOT="${ROOT_DIR}/output/dna_evaluation/${EVAL_NAME}"

MODELS=(
  "llama8b"
  "qwen35_9b"
  "qwen35_27b"
  "llama33_70b"
  "mistral_7b"
  "mistral_24b"
)

mkdir -p "${EVAL_ROOT}"
echo "Baseline root: ${BASELINE_ROOT}"
echo "Variant root:  ${VARIANT_ROOT}"
echo "Eval root:     ${EVAL_ROOT}"

for model in "${MODELS[@]}"; do
  O1_JSONL="${BASELINE_ROOT}/${model}/p1_baseline_outputs.jsonl"
  O2_JSONL="${VARIANT_ROOT}/${model}/p1_baseline_outputs.jsonl"
  OUT_DIR="${EVAL_ROOT}/${model}"

  if [[ ! -f "${O1_JSONL}" ]]; then
    echo "SKIP ${model}: missing baseline file ${O1_JSONL}"
    continue
  fi
  if [[ ! -f "${O2_JSONL}" ]]; then
    echo "SKIP ${model}: missing variant file ${O2_JSONL}"
    continue
  fi

  mkdir -p "${OUT_DIR}"
  echo ""
  echo "=========================================="
  echo "Evaluating model: ${model}"
  echo "O1 (baseline): ${O1_JSONL}"
  echo "O2 (variant):  ${O2_JSONL}"
  echo "Output dir:    ${OUT_DIR}"
  echo "=========================================="

  export WANDB_JOB_TYPE="evaluation"
  export WANDB_RUN_NAME="dna_eval_${EVAL_NAME}_${model}"

  $PYTHON_CMD src/evaluation_ranking/evaluate_safety_helpfulness.py \
    --o1-jsonl "${O1_JSONL}" \
    --o2-jsonl "${O2_JSONL}" \
    --output-dir "${OUT_DIR}" \
    --evals helpfulness,harmlessness \
    --harmfulness-mode \
    --use-local-model \
    --judge-model "${JUDGE_RESOLVED}" \
    --temperature 0.1 \
    --max-tokens 512 \
    --wandb

  rc=$?
  echo "Done eval: ${model} | Exit: ${rc} | $(date)"
done

echo ""
echo "All evaluations done at: $(date)"
