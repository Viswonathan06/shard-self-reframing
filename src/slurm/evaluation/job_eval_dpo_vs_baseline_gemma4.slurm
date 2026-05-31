#!/usr/bin/bash
# Evaluation: Llama 8B P1 baseline vs DPO model outputs — Gemma 4 31B judge
#
# O1: dna_baseline_multimodel/llama8b/p1_baseline_outputs.jsonl  (field: output)
# O2: DPO completions from output/dpo_evaluation/dpo_llama8b_inference.jsonl (field: dpo_generated_response)
#
# Usage:
#   sbatch src/slurm/evaluation/job_eval_dpo_vs_baseline_gemma4.slurm
#
# Overrides:
#   DPO_INFERENCE_JSONL=/path sbatch ...
#   EVAL_OUT_DIR=/path sbatch ...
#   MAX_RECORDS=N sbatch ...

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --job-name=eval_baseline_vs_dpo_gemma4
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
  [[ -d "${_ROOT_CAND}/src/evaluation_ranking" ]] && ROOT_DIR="${_ROOT_CAND}"
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

DPO_INFERENCE_FILE="${DPO_INFERENCE_JSONL:-${ROOT_DIR}/output/dpo_evaluation/dpo_llama8b_inference.jsonl}"
BASELINE_JSONL="${BASELINE_JSONL:-${ROOT_DIR}/output/DNA_New_Experiments/dna_baseline_multimodel/llama8b/p1_baseline_outputs.jsonl}"

O2_FOR_EVAL="${ROOT_DIR}/output/dpo_evaluation/dpo_llama8b_outputs_for_gemma_pairwise.jsonl"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT_DIR}/output/dpo_evaluation/baseline_vs_dpo_llama8b_gemma4_judge}"

echo "=== Baseline vs DPO (Gemma 4 31B) — pairwise eval ==="
echo "Job: ${SLURM_JOB_ID:-local}  Host: $(hostname)  ROOT: ${ROOT_DIR}"
echo "  Baseline O1:    ${BASELINE_JSONL}"
echo "  DPO inference:  ${DPO_INFERENCE_FILE}"
echo "  Derived O2:     ${O2_FOR_EVAL}"
echo "  Eval output:    ${EVAL_OUT_DIR}"

[[ -f "${BASELINE_JSONL}" ]]      || { echo "ERROR: Baseline not found: ${BASELINE_JSONL}"; exit 1; }
[[ -f "${DPO_INFERENCE_FILE}" ]] || { echo "ERROR: DPO inference not found: ${DPO_INFERENCE_FILE}"; exit 1; }

# ── HF / Gemma ──────────────────────────────────────────────────────────────
if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
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

resolve_gemma4_path() {
  local p snap
  for p in \
    "$HF_HOME/models--google--gemma-4-31B-it/snapshots" \
    "$HF_HOME/hub/models--google--gemma-4-31B-it/snapshots" \
    "${HF_HOME}/hub/models--google--gemma-4-31B-it/snapshots"; do
    if [[ -d "$p" ]]; then
      snap=$(find "$p" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1)
      if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then
        echo "${snap}"; return 0
      fi
    fi
  done
  echo ""; return 1
}

GEMMA4_MODEL_PATH=$(resolve_gemma4_path)
if [[ -n "${GEMMA4_MODEL_PATH}" ]]; then
  export LOCAL_MODEL="${GEMMA4_MODEL_PATH}"
  echo "✓ Gemma snapshot: ${GEMMA4_MODEL_PATH}"
else
  export LOCAL_MODEL="google/gemma-4-31B-it"
  echo "⚠ No local Gemma snapshot; will attempt download"
fi

if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
  source "${ROOT_DIR}/.venv/bin/activate"; PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

mkdir -p "$(dirname "${O2_FOR_EVAL}")" "${EVAL_OUT_DIR}"

echo ""
echo "=== Building O2 JSONL from DPO inference ==="
"${PYTHON_CMD}" scripts/prepare_sft_eval_o2_jsonl.py \
  --input-jsonl  "${DPO_INFERENCE_FILE}" \
  --output-jsonl "${O2_FOR_EVAL}" \
  --text-field   "dpo_generated_response"

CMD_ARGS=(
  "--o1-jsonl"     "${BASELINE_JSONL}"
  "--o2-jsonl"     "${O2_FOR_EVAL}"
  "--output-dir"   "${EVAL_OUT_DIR}"
  "--evals"        "${EVALS}"
  "--temperature"  "${TEMPERATURE}"
  "--max-tokens"   "${MAX_TOKENS}"
  "--use-local-model"
  "--judge-model"  "${LOCAL_MODEL}"
)
[[ -n "${MAX_RECORDS}" ]] && CMD_ARGS+=("--max-records" "${MAX_RECORDS}")

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
