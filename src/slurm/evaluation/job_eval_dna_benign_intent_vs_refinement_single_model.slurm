#!/usr/bin/bash
# DNA: Benign-intent vs Refinement judge for a single model.
#
# Required env var:
#   MODEL=llama8b  (one of: llama8b llama33_70b qwen35_9b qwen35_27b mistral_7b mistral_24b)
#
# Usage:
#   for m in llama8b llama33_70b qwen35_9b qwen35_27b mistral_7b mistral_24b; do
#     MODEL=$m sbatch src/slurm/evaluation/job_eval_dna_benign_intent_vs_refinement_single_model.slurm
#   done

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_dna_bi_vs_ref_%x
#SBATCH --output=$PROJECT_ROOT/logs/eval_dna_bi_vs_ref_${MODEL}_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/eval_dna_bi_vs_ref_${MODEL}_%j.err

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
echo "DNA Benign-Intent vs Refinement — ${MODEL}"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version

EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RESTART_EVAL="${RESTART_EVAL:-}"

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

LOCAL_MODEL=""
for snap in "${HF_HOME}/hub/models--google--gemma-4-31B-it/snapshots/"/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && LOCAL_MODEL="${snap%/}" && break
done
[[ -n "$LOCAL_MODEL" ]] || LOCAL_MODEL="google/gemma-4-31B-it"
export LOCAL_MODEL
echo "Judge: ${LOCAL_MODEL}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

DNA_ROOT="${ROOT_DIR}/output/DNA_Per_Model_Experiments"
O1_FILE="${DNA_ROOT}/benign_intent_multimodel/${MODEL}/p1_baseline_outputs.jsonl"
O2_FILE="${DNA_ROOT}/refinement_from_baseline_multimodel/${MODEL}/baseline_outputs.jsonl"
EVAL_OUT_DIR="${DNA_ROOT}/evaluation/benign_intent_vs_refinement_${MODEL}_gemma4_judge"

if [[ ! -f "${O1_FILE}" ]]; then
    echo "ERROR: O1 not found: ${O1_FILE}"; exit 1
fi
if [[ ! -f "${O2_FILE}" ]]; then
    echo "ERROR: O2 not found: ${O2_FILE}"; exit 1
fi

echo "O1 (benign intent): ${O1_FILE}  ($(wc -l < "${O1_FILE}") rows)"
echo "O2 (refinement):    ${O2_FILE}  ($(wc -l < "${O2_FILE}") rows)"
echo "Out: ${EVAL_OUT_DIR}"
mkdir -p "${EVAL_OUT_DIR}"

CKPT="${EVAL_OUT_DIR}/eval_checkpoint.jsonl"
if [[ -n "${RESTART_EVAL}" ]]; then
    echo "RESTART_EVAL set — wiping checkpoint"
    rm -f "${CKPT}"
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
    "--batch-size"  "${BATCH_SIZE}"
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
