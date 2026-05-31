#!/usr/bin/bash
# Pairwise eval: Llama 8B answering P2 (safe rewrite) vs SFT answering P1 — Gemma 4 31B judge
#
# Experimental framing:
#   O1: Llama 8B baseline given the explicit *safe rewrite* (P2) — best-case baseline
#   O2: SFT Llama 8B given the *original* P1 — tests whether fine-tuning closes the gap
#
# Inputs (must exist before running this job):
#   output/linguasafe_evaluation/o1_llama8b_p2_baseline.jsonl   (built by prepare_linguasafe_unseen.py)
#   output/linguasafe_evaluation/sft_llama8b_inference.jsonl    (built by job_sft_inference_linguasafe.slurm)
#
# Usage:
#   sbatch src/slurm/evaluation/job_eval_linguasafe_sft_vs_p2baseline.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=36:00:00
#SBATCH --job-name=eval_linguasafe_sft_gemma4
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

EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

O1_JSONL="${O1_JSONL:-${ROOT_DIR}/output/linguasafe_evaluation/o1_llama8b_p2_baseline.jsonl}"
SFT_INFERENCE="${SFT_INFERENCE:-${ROOT_DIR}/output/linguasafe_evaluation/sft_llama8b_inference.jsonl}"
O2_FOR_EVAL="${O2_PREP_JSONL:-${ROOT_DIR}/output/linguasafe_evaluation/sft_llama8b_outputs_for_eval.jsonl}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT_DIR}/output/linguasafe_evaluation/sft_vs_p2baseline_gemma4_judge}"

echo "=== LinguaSafe: SFT(P1) vs Baseline(P2) — Gemma 4 31B pairwise eval ==="
echo "Job: ${SLURM_JOB_ID:-local}  Host: $(hostname)  ROOT: ${ROOT_DIR}"
echo "  O1 (baseline on P2): ${O1_JSONL}"
echo "  SFT inference:       ${SFT_INFERENCE}"
echo "  Derived O2 JSONL:    ${O2_FOR_EVAL}"
echo "  Eval output dir:     ${EVAL_OUT_DIR}"

if [[ ! -f "${O1_JSONL}" ]]; then
  echo "ERROR: O1 baseline not found: ${O1_JSONL}"
  echo "  Run: python scripts/prepare_linguasafe_unseen.py"
  exit 1
fi
if [[ ! -f "${SFT_INFERENCE}" ]]; then
  echo "ERROR: SFT inference not found: ${SFT_INFERENCE}"
  echo "  Run: sbatch src/slurm/inference/job_sft_inference_linguasafe.slurm"
  exit 1
fi

# HF / Gemma
if [[ -z "${HF_HOME:-}" ]]; then
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
  --input-jsonl  "${SFT_INFERENCE}" \
  --output-jsonl "${O2_FOR_EVAL}" \
  --text-field   sft_final_response

CMD_ARGS=(
  "--o1-jsonl" "${O1_JSONL}"
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
"${PYTHON_CMD}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "  ${EVAL_OUT_DIR}/eval_results.jsonl"
echo "  ${EVAL_OUT_DIR}/eval_summary.json"

exit $?
