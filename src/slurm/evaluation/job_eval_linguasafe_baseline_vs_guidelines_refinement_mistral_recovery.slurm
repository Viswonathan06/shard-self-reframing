#!/usr/bin/bash
# Recovery eval: LinguaSafe baseline vs guidelines-refinement for mistral_7b and mistral_24b.
# mistral_7b: resume from 2261/3267 (eval_checkpoint.jsonl exists).
# mistral_24b: start fresh (no eval output yet).
# Gemma4-31B judge at 4-way TP on gpu3 ($HF_HOME already has the model).
#
# Submit: sbatch src/slurm/evaluation/job_eval_linguasafe_baseline_vs_guidelines_refinement_mistral_recovery.slurm

#SBATCH -n 1
#SBATCH --nodelist=gpu3
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=eval_linguasafe_guidelines_refinement_mistral
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
echo "=== LinguaSafe: Baseline vs Guidelines Refinement Eval (Mistral recovery) ==="
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
"${PYTHON}" --version

# Gemma4-31B judge via node-local /playpen
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_MAX_MODEL_LEN=8192
export VLLM_ENFORCE_EAGER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

PLAYPEN_HUB="$HF_HOME"
GEMMA4_PATH=""
for snap in "${PLAYPEN_HUB}/models--google--gemma-4-31B-it/snapshots/"/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && GEMMA4_PATH="${snap%/}" && break
done
if [[ -z "${GEMMA4_PATH}" ]]; then
    echo "ERROR: Gemma4-31B not found in ${PLAYPEN_HUB}"; exit 1
fi
echo "Judge: ${GEMMA4_PATH}"

export HF_HUB_OFFLINE=1
export USE_LOCAL_MODEL=true

nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

REFINE_GUIDELINES_DIR="${ROOT_DIR}/output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_guidelines_multimodel"
EVAL_BASE="${ROOT_DIR}/output/Linguasafe_Experiments/evaluation"

MODELS=(mistral_7b mistral_24b)
TOTAL=${#MODELS[@]}; N=0
SUCCESSFUL_MODELS=()
FAILED_MODELS=()

for model in "${MODELS[@]}"; do
    N=$((N+1))
    echo ""
    echo "=== Model $N/$TOTAL: $model | $(date) ==="

    REFINEMENT_FILE="${REFINE_GUIDELINES_DIR}/${model}/checkpoint.jsonl"
    EVAL_OUT_DIR="${EVAL_BASE}/baseline_vs_guidelines_refinement_${model}_gemma4_judge"

    if [[ ! -f "${REFINEMENT_FILE}" ]]; then
        echo "ERROR: refinement checkpoint not found: ${REFINEMENT_FILE}"
        FAILED_MODELS+=("${model}"); continue
    fi
    echo "Input: ${REFINEMENT_FILE} ($(wc -l < "${REFINEMENT_FILE}") lines)"
    mkdir -p "${EVAL_OUT_DIR}"

    N_DONE=0
    [[ -f "${EVAL_OUT_DIR}/eval_results.jsonl" ]] && N_DONE=$(wc -l < "${EVAL_OUT_DIR}/eval_results.jsonl")
    N_TOTAL=$(wc -l < "${REFINEMENT_FILE}")

    if [[ "${N_DONE}" -ge "${N_TOTAL}" ]] && [[ "${N_DONE}" -gt 0 ]]; then
        echo "${model}: already complete (${N_DONE} rows). Skipping."
        SUCCESSFUL_MODELS+=("${model}"); continue
    fi
    echo "${model}: resuming from ${N_DONE}/${N_TOTAL}..."

    if "${PYTHON}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
        --refinement-jsonl "${REFINEMENT_FILE}" \
        --output-dir       "${EVAL_OUT_DIR}" \
        --evals            all \
        --temperature      0.1 \
        --max-tokens       512 \
        --use-local-model \
        --judge-model      "${GEMMA4_PATH}"; then
        echo "SUCCESS: ${model} ($(date))"
        SUCCESSFUL_MODELS+=("${model}")
        [[ -f "${EVAL_OUT_DIR}/eval_results.jsonl" ]] && echo "  eval_results: $(wc -l < "${EVAL_OUT_DIR}/eval_results.jsonl") rows"
        [[ -f "${EVAL_OUT_DIR}/eval_summary.json" ]] && grep -E "(helpfulness|harmlessness|win|total)" "${EVAL_OUT_DIR}/eval_summary.json" | head -6 || true
    else
        echo "FAILED: ${model} ($(date))"
        FAILED_MODELS+=("${model}")
    fi
done

echo ""
echo "======================================================"
echo "Done: ${#SUCCESSFUL_MODELS[@]}/${TOTAL} models | $(date)"
[[ ${#FAILED_MODELS[@]} -gt 0 ]] && printf 'Failed: %s\n' "${FAILED_MODELS[@]}" && exit 1
echo "======================================================"
exit 0
