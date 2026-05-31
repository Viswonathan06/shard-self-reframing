#!/usr/bin/bash
# Evaluation: Llama 8B P1 baseline vs SFT model outputs — Gemma 4 31B judge
# Uses the same script and judge prompts as DNA experiments:
#   src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py
#
# O1: dna_baseline_multimodel/llama8b/p1_baseline_outputs.jsonl  (field: output)
# O2: SFT completions derived from output/sft_inference_results.jsonl  (field: output)
#
# Usage:
#   sbatch src/slurm/evaluation/job_eval_sft_vs_baseline_gemma4.slurm
#
# Overrides:
#   BASELINE_JSONL=/path/to/p1_baseline_outputs.jsonl sbatch ...
#   SFT_INFERENCE_JSONL=/path/to/sft_inference_results.jsonl sbatch ...
#   EVAL_OUT_DIR=/path/to/dir sbatch ...
#   MODEL=llama8b (only affects baseline path if BASELINE_JSONL unset)
#   EVALS=all|helpfulness,relevance,harmlessness
#   MAX_RECORDS=N

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#SBATCH --job-name=eval_baseline_vs_sft_gemma4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-}"
if [[ -z "$ROOT_DIR" ]] && [[ -d "${SLURM_SUBMIT_DIR:-}/src/evaluation_ranking" ]]; then
  ROOT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
fi
if [[ -z "$ROOT_DIR" ]]; then
  _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  _ROOT_CAND="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
  if [[ -d "${_ROOT_CAND}/src/evaluation_ranking" ]]; then
    ROOT_DIR="${_ROOT_CAND}"
  fi
fi
if [[ -z "$ROOT_DIR" ]] || [[ ! -d "$ROOT_DIR/src/evaluation_ranking" ]]; then
  echo "ERROR: Could not resolve repo root."
  exit 1
fi

cd "$ROOT_DIR" || exit 1
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

MODEL="${MODEL:-llama8b}"
EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

SFT_INFERENCE_FILE="${SFT_INFERENCE_JSONL:-${ROOT_DIR}/output/sft_inference_results.jsonl}"
BASELINE_JSONL="${BASELINE_JSONL:-${ROOT_DIR}/output/DNA_New_Experiments/dna_baseline_multimodel/${MODEL}/p1_baseline_outputs.jsonl}"

O2_FOR_EVAL="${O2_PREP_JSONL:-${ROOT_DIR}/output/sft_evaluation/sft_${MODEL}_outputs_for_gemma_pairwise.jsonl}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT_DIR}/output/sft_evaluation/baseline_vs_sft_${MODEL}_gemma4_judge}"

echo "=== Baseline vs SFT (Gemma 4 31B) — DNA-style pairwise eval ==="
echo "Job: ${SLURM_JOB_ID:-local}  Host: $(hostname)  ROOT: ${ROOT_DIR}"
echo "MODEL: ${MODEL}"
echo "  Baseline O1 JSONL:  ${BASELINE_JSONL}"
echo "  SFT inference:      ${SFT_INFERENCE_FILE}"
echo "  Derived O2 JSONL:   ${O2_FOR_EVAL}"
echo "  Eval output dir:    ${EVAL_OUT_DIR}"

if [[ ! -f "${BASELINE_JSONL}" ]]; then
  echo "ERROR: Baseline file not found: ${BASELINE_JSONL}"
  exit 1
fi
if [[ ! -f "${SFT_INFERENCE_FILE}" ]]; then
  echo "ERROR: SFT inference file not found: ${SFT_INFERENCE_FILE}"
  exit 1
fi

# HF / Gemma
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d /playpen/huggingface/hub ]] && [[ -w /playpen/huggingface/hub ]]; then
    export HF_HOME=/playpen/huggingface
  elif [[ -d "/playpen/${USER}/hf_cache" ]]; then
    export HF_HOME="/playpen/${USER}/hf_cache"
  else
    export HF_HOME="${HOME}/.cache/huggingface"
  fi
fi

export USE_LOCAL_MODEL=true
export JUDGE_MODEL="google/gemma-4-31B-it"
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_MAX_MODEL_LEN=4096
export VLLM_ENFORCE_EAGER=1

GEMMA4_SHARED_PATH="/playpen/huggingface/hub/models--google--gemma-4-31B-it"
GEMMA4_PLAYPEN="${PLAYPEN:-/playpen}/guam/hf_cache/hub/models--google--gemma-4-31B-it"
GEMMA4_USER_PATH="${HF_HOME}/hub/models--google--gemma-4-31B-it"

resolve_gemma4_path() {
  local p snap
  for p in "${GEMMA4_SHARED_PATH}/snapshots" "${GEMMA4_PLAYPEN}/snapshots" "${GEMMA4_USER_PATH}/snapshots"; do
    if [[ -d "$p" ]]; then
      snap=$(find "$p" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
      if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then
        echo "${snap}"
        return 0
      fi
    fi
  done
  echo ""
  return 1
}

GEMMA4_MODEL_PATH=$(resolve_gemma4_path)
if [[ -n "${GEMMA4_MODEL_PATH}" ]]; then
  export LOCAL_MODEL="${GEMMA4_MODEL_PATH}"
  echo "✓ Gemma snapshot: ${GEMMA4_MODEL_PATH}"
else
  export LOCAL_MODEL="google/gemma-4-31B-it"
  echo "⚠ No local Gemma snapshot; LOCAL_MODEL=${LOCAL_MODEL}"
fi

if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

mkdir -p "$(dirname "${O2_FOR_EVAL}")" "${EVAL_OUT_DIR}"

echo ""
echo "=== Building O2 JSONL from SFT inference ==="
"${PYTHON_CMD}" scripts/prepare_sft_eval_o2_jsonl.py \
  --input-jsonl "${SFT_INFERENCE_FILE}" \
  --output-jsonl "${O2_FOR_EVAL}"

CMD_ARGS=(
  "--o1-jsonl" "${BASELINE_JSONL}"
  "--o2-jsonl" "${O2_FOR_EVAL}"
  "--output-dir" "${EVAL_OUT_DIR}"
  "--evals" "${EVALS}"
  "--temperature" "${TEMPERATURE}"
  "--max-tokens" "${MAX_TOKENS}"
  "--use-local-model"
  "--judge-model" "${LOCAL_MODEL}"
)
if [[ -n "${MAX_RECORDS}" ]]; then
  CMD_ARGS+=("--max-records" "${MAX_RECORDS}")
fi

echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""
echo "=== Running evaluate_safety_helpfulness_single_random.py ==="
echo "${PYTHON_CMD} src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py ${CMD_ARGS[*]}"
"${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "  ${EVAL_OUT_DIR}/eval_results.jsonl"
echo "  ${EVAL_OUT_DIR}/eval_summary.json"

exit $?
