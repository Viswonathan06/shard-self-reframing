#!/usr/bin/bash
# LinguaSafe: Baseline vs Refinement judge for a single model.
# Submit 6 of these in parallel (one per model) for maximum throughput.
#
# Required env var:
#   MODEL=llama8b  (one of: llama8b llama33_70b qwen35_9b qwen35_27b mistral_7b mistral_24b)
#
# Usage:
#   for m in llama8b llama33_70b qwen35_9b qwen35_27b mistral_7b mistral_24b; do
#     MODEL=$m sbatch src/slurm/evaluation/job_eval_baseline_vs_refinement_single_model.slurm
#   done

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_bvr_%x
#SBATCH --output=$PROJECT_ROOT/logs/eval_bvr_${MODEL}_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/eval_bvr_${MODEL}_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true

MODEL="${MODEL:?MODEL env var required (e.g. MODEL=llama8b)}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
echo "======================================================"
echo "Baseline vs Refinement — ${MODEL}"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version

EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
RESTART_EVAL="${RESTART_EVAL:-}"   # set to 1 to wipe checkpoint; default = resume

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
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLLM_MAX_MODEL_LEN=8192
export FLASHINFER_DISABLE_VERSION_CHECK=1
# No VLLM_ENFORCE_EAGER — allows CUDA graphs for ~2-3x throughput improvement

LOCAL_MODEL=""
for snap in "${HF_HOME}/hub/models--google--gemma-4-31B-it/snapshots/"/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && LOCAL_MODEL="${snap%/}" && break
done
[[ -n "$LOCAL_MODEL" ]] || LOCAL_MODEL="google/gemma-4-31B-it"
export LOCAL_MODEL
echo "Judge: ${LOCAL_MODEL}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

LINGUASAFE_ROOT="${ROOT_DIR}/output/Linguasafe_Experiments"
EVAL_ROOT="${LINGUASAFE_ROOT}/evaluation"
O1_FILE="${LINGUASAFE_ROOT}/helpful_assistant_multimodel/${MODEL}/p1_baseline_outputs.jsonl"
O2_FILE="${LINGUASAFE_ROOT}/linguasafe_refinement_from_baseline_multimodel/${MODEL}/baseline_outputs.jsonl"
EVAL_OUT_DIR="${EVAL_ROOT}/baseline_vs_refinement_${MODEL}_gemma4_judge"

if [[ ! -f "${O1_FILE}" ]]; then
    echo "ERROR: O1 not found: ${O1_FILE}"; exit 1
fi
if [[ ! -f "${O2_FILE}" ]]; then
    echo "ERROR: O2 not found: ${O2_FILE}"; exit 1
fi

echo "O1: ${O1_FILE}  ($(wc -l < "${O1_FILE}") rows)"
echo "O2: ${O2_FILE}  ($(wc -l < "${O2_FILE}") rows)"
echo "Out: ${EVAL_OUT_DIR}"
mkdir -p "${EVAL_OUT_DIR}"

CKPT="${EVAL_OUT_DIR}/eval_checkpoint.jsonl"
if [[ -n "${RESTART_EVAL}" ]]; then
    echo "RESTART_EVAL set — wiping checkpoint"
elif [[ -f "${CKPT}" ]]; then
    echo "Resuming from checkpoint: $(wc -l < "${CKPT}") rows done"
fi

CMD_ARGS=(
    "--o1-jsonl"    "${O1_FILE}"
    "--o2-jsonl"    "${O2_FILE}"
    "--output-dir"  "${EVAL_OUT_DIR}"
    "--evals"       "${EVALS}"
    "--temperature" "${TEMPERATURE}"
    "--max-tokens"  "${MAX_TOKENS}"
    "--use-local-model"
    "--judge-model" "${LOCAL_MODEL}"
)
[[ -n "${RESTART_EVAL}" ]]  && CMD_ARGS+=("--restart-eval")
[[ -n "${MAX_RECORDS}" ]]   && CMD_ARGS+=("--max-records" "${MAX_RECORDS}")

"${PYTHON}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py "${CMD_ARGS[@]}"
EXIT=$?

echo ""
echo "======================================================"
echo "${MODEL}: exit=${EXIT} | $(date)"
[[ -f "${EVAL_OUT_DIR}/eval_summary.json" ]] && \
    "${PYTHON}" -c "
import json
d = json.load(open('${EVAL_OUT_DIR}/eval_summary.json'))
for k in ('helpfulness','harmlessness'):
    v = d.get(k,{})
    print(f'  {k}: o1_wins={v.get(\"o1_wins\",\"?\")}, o2_wins={v.get(\"o2_wins\",\"?\")}, ties={v.get(\"ties\",\"?\")}')
" 2>/dev/null || true
echo "======================================================"

exit ${EXIT}
