#!/usr/bin/bash
# LinguaSafe: Baseline vs Refinement — two models in parallel on one 4-GPU node.
# Each model gets 2 GPUs (tp=2), running concurrently to double throughput.
#
# Required env vars:
#   MODEL_A=llama8b   MODEL_B=qwen35_9b
#   (any of: llama8b llama33_70b qwen35_9b qwen35_27b mistral_7b mistral_24b)
#
# Optional:
#   RESTART_EVAL=1   — wipe checkpoints and start fresh (default: resume)
#   MODEL_B=""       — run only MODEL_A (solo mode, still uses tp=2 on GPUs 0,1)
#
# To kill one sub-process without stopping the other:
#   kill $(cat logs/eval_bvr_<model>_<jobid>.pid)
#
# Usage:
#   MODEL_A=qwen35_9b  MODEL_B=mistral_7b   sbatch src/slurm/evaluation/job_eval_bvr_paired.slurm
#   MODEL_A=qwen35_27b MODEL_B=mistral_24b  sbatch src/slurm/evaluation/job_eval_bvr_paired.slurm
#   MODEL_A=llama33_70b MODEL_B=""          sbatch src/slurm/evaluation/job_eval_bvr_paired.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_bvr_paired
#SBATCH --output=$PROJECT_ROOT/logs/eval_bvr_paired_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/eval_bvr_paired_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true

MODEL_A="${MODEL_A:?MODEL_A env var required}"
MODEL_B="${MODEL_B:-}"
# RESTART_EVAL=1 restarts both; MODEL_A_RESTART=1 / MODEL_B_RESTART=1 for per-model control
RESTART_EVAL="${RESTART_EVAL:-}"
MODEL_A_RESTART="${MODEL_A_RESTART:-${RESTART_EVAL}}"
MODEL_B_RESTART="${MODEL_B_RESTART:-${RESTART_EVAL}}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
JOB_ID="${SLURM_JOB_ID:-local}"
LOG_DIR="${ROOT_DIR}/logs"

echo "======================================================"
echo "Paired Eval — A=${MODEL_A}  B=${MODEL_B:-solo}  restart=${RESTART_EVAL:-no}"
echo "Job: ${JOB_ID}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version

EVALS="${EVALS:-all}"
MAX_RECORDS="${MAX_RECORDS:-}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-512}"

if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
else
    export HF_HOME="${HOME}/.cache/huggingface"
fi
export HF_HUB_CACHE="${HF_HOME}/hub"

LOCAL_MODEL=""
for snap in "${HF_HOME}/hub/models--google--gemma-4-31B-it/snapshots/"/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && LOCAL_MODEL="${snap%/}" && break
done
[[ -n "$LOCAL_MODEL" ]] || LOCAL_MODEL="google/gemma-4-31B-it"
echo "Judge: ${LOCAL_MODEL}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

LINGUASAFE_ROOT="${ROOT_DIR}/output/Linguasafe_Experiments"
EVAL_ROOT="${LINGUASAFE_ROOT}/evaluation"

# Split SLURM-allocated GPUs into two pairs instead of hardcoding 0,1 / 2,3.
# SLURM sets CUDA_VISIBLE_DEVICES to the physical GPUs assigned to this job.
IFS=',' read -ra SLURM_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
GPUS_A="${SLURM_GPUS[0]},${SLURM_GPUS[1]}"
GPUS_B="${SLURM_GPUS[2]},${SLURM_GPUS[3]}"
echo "SLURM GPUs: ${CUDA_VISIBLE_DEVICES:-0,1,2,3}  →  A=${GPUS_A}  B=${GPUS_B}"

# Cleanup PIDs on exit/signal so stale .pid files don't linger
CHILD_PIDS=()
cleanup() {
    echo "Caught signal — stopping child processes..."
    for pid in "${CHILD_PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
}
trap cleanup SIGTERM SIGINT

# Run one model eval as a background process.
# Logs go to a per-model file so you can tail them independently.
# Writes a .pid file so you can kill it without touching the other.
run_eval() {
    local model="$1"
    local cuda_devices="$2"
    local restart="${3:-}"
    local log="${LOG_DIR}/eval_bvr_${model}_${JOB_ID}.out"
    local pid_file="${LOG_DIR}/eval_bvr_${model}_${JOB_ID}.pid"

    local o1="${LINGUASAFE_ROOT}/helpful_assistant_multimodel/${model}/p1_baseline_outputs.jsonl"
    local o2="${LINGUASAFE_ROOT}/linguasafe_refinement_from_baseline_multimodel/${model}/baseline_outputs.jsonl"
    local out_dir="${EVAL_ROOT}/baseline_vs_refinement_${model}_gemma4_judge"
    local ckpt="${out_dir}/eval_checkpoint.jsonl"

    if [[ ! -f "${o1}" ]]; then echo "ERROR [${model}]: O1 not found: ${o1}" >&2; return 1; fi
    if [[ ! -f "${o2}" ]]; then echo "ERROR [${model}]: O2 not found: ${o2}" >&2; return 1; fi
    mkdir -p "${out_dir}"

    # Resume by default; wipe only when per-model restart flag is set
    local restart_flag=""
    if [[ -n "${restart}" ]]; then
        restart_flag="--restart-eval"
        echo "[${model}] RESTART set — wiping checkpoint" >&2
    elif [[ -f "${ckpt}" ]]; then
        local done_count
        done_count=$(wc -l < "${ckpt}")
        local total_count
        total_count=$(wc -l < "${o2}")
        echo "[${model}] Resuming from checkpoint: ${done_count}/${total_count} done" >&2
    else
        echo "[${model}] No checkpoint found — starting fresh" >&2
    fi

    echo "[${model}] GPUs=${cuda_devices}  log=${log}  pid_file=${pid_file}" >&2

    local cmd_args=(
        "--o1-jsonl"    "${o1}"
        "--o2-jsonl"    "${o2}"
        "--output-dir"  "${out_dir}"
        "--evals"       "${EVALS}"
        "--temperature" "${TEMPERATURE}"
        "--max-tokens"  "${MAX_TOKENS}"
        "--use-local-model"
        "--judge-model" "${LOCAL_MODEL}"
    )
    [[ -n "${restart_flag}" ]]    && cmd_args+=("${restart_flag}")
    [[ -n "${MAX_RECORDS:-}" ]]   && cmd_args+=("--max-records" "${MAX_RECORDS}")

    # Launch directly in current shell (not a subshell) so the parent can wait on it.
    # Use `env` to scope the GPU/vLLM vars without a subshell wrapper.
    env \
        CUDA_VISIBLE_DEVICES="${cuda_devices}" \
        VLLM_TENSOR_PARALLEL_SIZE=2 \
        VLLM_GPU_MEMORY_UTILIZATION=0.85 \
        VLLM_MAX_MODEL_LEN=4096 \
        VLLM_TRUST_REMOTE_CODE=1 \
        VLLM_ONLY=1 \
        USE_LOCAL_MODEL=true \
        JUDGE_MODEL="google/gemma-4-31B-it" \
        FLASHINFER_DISABLE_VERSION_CHECK=1 \
        "${PYTHON}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
            "${cmd_args[@]}" >"${log}" 2>&1 &
    local pid=$!
    echo "${pid}" > "${pid_file}"
    echo "[${model}] PID=${pid}  (kill: kill \$(cat ${pid_file}))" >&2
}

# Launch evals — call directly (not via $()) so PIDs are children of this shell
run_eval "${MODEL_A}" "${GPUS_A}" "${MODEL_A_RESTART}"
PID_A=$(cat "${LOG_DIR}/eval_bvr_${MODEL_A}_${JOB_ID}.pid")
CHILD_PIDS+=("${PID_A}")

PID_B=""
if [[ -n "${MODEL_B}" ]]; then
    # Small stagger so torch.compile cache writes don't collide
    sleep 30
    run_eval "${MODEL_B}" "${GPUS_B}" "${MODEL_B_RESTART}"
    PID_B=$(cat "${LOG_DIR}/eval_bvr_${MODEL_B}_${JOB_ID}.pid")
    CHILD_PIDS+=("${PID_B}")
fi

echo ""
echo "Both processes launched. Tail logs with:"
echo "  tail -f ${LOG_DIR}/eval_bvr_${MODEL_A}_${JOB_ID}.out"
[[ -n "${MODEL_B}" ]] && echo "  tail -f ${LOG_DIR}/eval_bvr_${MODEL_B}_${JOB_ID}.out"
echo ""

# Wait for both and collect exit codes
EXIT=0

wait "${PID_A}" && echo "[${MODEL_A}] SUCCESS" || { echo "[${MODEL_A}] FAILED"; EXIT=1; }

if [[ -n "${PID_B}" ]]; then
    wait "${PID_B}" && echo "[${MODEL_B}] SUCCESS" || { echo "[${MODEL_B}] FAILED"; EXIT=1; }
fi

# Clean up pid files
rm -f "${LOG_DIR}/eval_bvr_${MODEL_A}_${JOB_ID}.pid"
[[ -n "${MODEL_B}" ]] && rm -f "${LOG_DIR}/eval_bvr_${MODEL_B}_${JOB_ID}.pid"

echo ""
echo "======================================================"
echo "Paired eval done | exit=${EXIT} | $(date)"
echo "======================================================"
exit ${EXIT}
