#!/usr/bin/bash
# LinguaSafe: Baseline (O1) vs Refinement (O2) — all 6 models, Gemma4 judge.
#
# O1 source: output/Linguasafe_Experiments/helpful_assistant_multimodel/{model}/p1_baseline_outputs.jsonl
# O2 source: output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_multimodel/{model}/baseline_outputs.jsonl
#
# Uses --restart-eval to clear stale null checkpoints.
# Output: output/Linguasafe_Experiments/evaluation/baseline_vs_refinement_{model}_gemma4_judge/
#
# Usage:
#   sbatch src/slurm/evaluation/job_eval_linguasafe_baseline_vs_refinement_gemma4_fixed.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=eval_linguasafe_baseline_vs_refinement_gemma4
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
echo "======================================================"
echo "LinguaSafe: Baseline vs Refinement — All Models"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "Python: ${PYTHON}"
echo "======================================================"
"${PYTHON}" --version

MODELS_STR="${MODELS:-llama8b llama33_70b qwen35_9b qwen35_27b mistral_7b mistral_24b}"
read -ra MODELS <<< "${MODELS_STR}"
EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

echo "Models: ${MODELS[*]}"
echo ""

# --- Gemma4 setup (same as current working judge jobs) ---
if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
else
    export HF_HOME="${HOME}/.cache/huggingface"
fi
export HF_HUB_CACHE="${HF_HOME}/hub"

export USE_LOCAL_MODEL=true
export JUDGE_MODEL="google/gemma-4-31B-it"
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=4096
export VLLM_ENFORCE_EAGER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

LOCAL_MODEL=""
for snap in "${HF_HOME}/hub/models--google--gemma-4-31B-it/snapshots/"/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && LOCAL_MODEL="${snap%/}" && break
done
[[ -n "$LOCAL_MODEL" ]] || LOCAL_MODEL="google/gemma-4-31B-it"
export LOCAL_MODEL
echo "Judge: ${LOCAL_MODEL}"
echo ""

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

LINGUASAFE_ROOT="${ROOT_DIR}/output/Linguasafe_Experiments"
EVAL_ROOT="${LINGUASAFE_ROOT}/evaluation"
O1_ROOT="${LINGUASAFE_ROOT}/helpful_assistant_multimodel"
O2_ROOT="${LINGUASAFE_ROOT}/linguasafe_refinement_from_baseline_multimodel"

SUCCESSFUL_MODELS=()
FAILED_MODELS=()
TOTAL=${#MODELS[@]}; N=0

for model in "${MODELS[@]}"; do
    N=$((N+1))
    echo ""
    echo "=== Model $N/$TOTAL: $model | $(date) ==="

    O1_FILE="${O1_ROOT}/${model}/p1_baseline_outputs.jsonl"
    O2_FILE="${O2_ROOT}/${model}/baseline_outputs.jsonl"
    EVAL_OUT_DIR="${EVAL_ROOT}/baseline_vs_refinement_${model}_gemma4_judge"

    if [[ ! -f "${O1_FILE}" ]]; then
        echo "ERROR: O1 not found: ${O1_FILE}"; FAILED_MODELS+=("${model}"); continue
    fi
    if [[ ! -f "${O2_FILE}" ]]; then
        echo "ERROR: O2 not found: ${O2_FILE}"; FAILED_MODELS+=("${model}"); continue
    fi

    echo "  O1: ${O1_FILE}  ($(wc -l < "${O1_FILE}") rows)"
    echo "  O2: ${O2_FILE}  ($(wc -l < "${O2_FILE}") rows)"
    echo "  Out: ${EVAL_OUT_DIR}"
    mkdir -p "${EVAL_OUT_DIR}"

    CMD_ARGS=(
        "--o1-jsonl"    "${O1_FILE}"
        "--o2-jsonl"    "${O2_FILE}"
        "--output-dir"  "${EVAL_OUT_DIR}"
        "--evals"       "${EVALS}"
        "--temperature" "${TEMPERATURE}"
        "--max-tokens"  "${MAX_TOKENS}"
        "--use-local-model"
        "--judge-model" "${LOCAL_MODEL}"
        "--restart-eval"
    )
    [[ -n "${MAX_RECORDS}" ]] && CMD_ARGS+=("--max-records" "${MAX_RECORDS}")

    if "${PYTHON}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"; then
        echo "SUCCESS: ${model}"; SUCCESSFUL_MODELS+=("${model}")
        [[ -f "${EVAL_OUT_DIR}/eval_summary.json" ]] && \
            python3 -c "
import json
d = json.load(open('${EVAL_OUT_DIR}/eval_summary.json'))
for k in ('helpfulness','harmlessness'):
    v = d.get(k,{})
    print(f'  {k}: o1_wins={v.get(\"o1_wins\",\"?\")}, o2_wins={v.get(\"o2_wins\",\"?\")}, ties={v.get(\"ties\",\"?\")}')
" 2>/dev/null || true
    else
        echo "FAILED: ${model}"; FAILED_MODELS+=("${model}")
    fi
done

echo ""
echo "======================================================"
echo "Done: ${#SUCCESSFUL_MODELS[@]}/${TOTAL} models | $(date)"
[[ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]] && printf '  OK:   %s\n' "${SUCCESSFUL_MODELS[@]}"
[[ ${#FAILED_MODELS[@]} -gt 0 ]]    && printf '  FAIL: %s\n' "${FAILED_MODELS[@]}"
echo "======================================================"

[[ ${#FAILED_MODELS[@]} -gt 0 ]] && exit 1 || exit 0
