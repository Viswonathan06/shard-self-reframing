#!/usr/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=p1_to_p2_instructions_only
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || {
    echo "ERROR: Cannot cd to $ROOT_DIR"
    exit 1
}

source ~/.bashrc 2>/dev/null || true

# Archive existing logs before starting
LOGS_DIR="${ROOT_DIR}/logs"
ARCHIVE_DIR="${LOGS_DIR}/archive"
mkdir -p "$ARCHIVE_DIR"

if [ -d "$LOGS_DIR" ]; then
    # Create archive subdirectory with timestamp
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    ARCHIVE_SUBDIR="${ARCHIVE_DIR}/${TIMESTAMP}"
    mkdir -p "$ARCHIVE_SUBDIR"
    
    # Move all existing log files (but not archive directory itself)
    if find "$LOGS_DIR" -maxdepth 1 -type f \( -name "*.out" -o -name "*.err" \) | grep -q .; then
        echo "Archiving existing logs to ${ARCHIVE_SUBDIR}..."
        find "$LOGS_DIR" -maxdepth 1 -type f \( -name "*.out" -o -name "*.err" \) -exec mv {} "$ARCHIVE_SUBDIR/" \;
        echo "✓ Archived existing logs"
    fi
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="/playpen/huggingface"
export USE_LOCAL_MODEL=true
export VLLM_TENSOR_PARALLEL_SIZE=4
# 0.85 gives vLLM room for 70B weights + KV cache on 4 GPUs; lower if node is shared
export VLLM_GPU_MEMORY_UTILIZATION=0.85
# If CUDA device-side assert recurs, uncomment below to get the exact failing line:
# export CUDA_LAUNCH_BLOCKING=1

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not found in environment"
    echo "Make sure HF_TOKEN is set in ~/.bashrc"
    exit 1
fi

# Prefer project venv (has numpy<2.4 for vLLM/numba). Else Anaconda/system Python.
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
    source "${ROOT_DIR}/.venv/bin/activate"
    PYTHON_CMD="python"
    echo "Using project .venv (numpy<2.4 for vLLM)"
elif command -v python &> /dev/null && [[ $(which python) == *"anaconda3"* ]]; then
    PYTHON_CMD="python"
elif [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    PYTHON_CMD="python3"
fi
# vLLM worker subprocesses need numba, which requires numpy<2.4. Use project venv or: pip install 'numpy>=2.2,<2.4'

# Diagnostics: what is using the GPUs (see if another job or your abandoned run is holding memory)
echo "=== GPU diagnostics (current user: $USER, job id: ${SLURM_JOB_ID:-N/A}) ==="
nvidia-smi 2>/dev/null || true
echo "=== GPU processes (pid, process_name, used_memory) ==="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
echo "=== Process details (user, pid, command) ==="
my_pids=""
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ -z "$pid" ] && continue
    line=$(ps -o user=,pid=,args= -p "$pid" 2>/dev/null)
    [ -n "$line" ] && echo "$line"
    proc_user=$(ps -o user= -p "$pid" 2>/dev/null)
    [ "$proc_user" = "$USER" ] && my_pids="${my_pids} $pid"
done
[ -n "$my_pids" ] && echo ">>> GPU process(es) owned by you (${USER}):${my_pids} — if from an abandoned run, free GPUs with: kill${my_pids}"
echo "=== end GPU diagnostics ==="

# Use existing LLaMA 3.3 70B Instruct (override with LOCAL_MODEL_SNAPSHOT if needed)
MODEL_SNAPSHOT="${LOCAL_MODEL_SNAPSHOT:-$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}"
if [ ! -d "${MODEL_SNAPSHOT}" ]; then
    echo "ERROR: Model snapshot not found: ${MODEL_SNAPSHOT}"
    echo "Set LOCAL_MODEL_SNAPSHOT to a valid path, or ensure the model exists on this node."
    exit 1
fi
echo "Using model: ${MODEL_SNAPSHOT}"

cleanup() { true; }
trap cleanup EXIT ERR
LANGUAGE=${LANGUAGE:-en}

# Instructions-only mode: no universal or category-specific guidelines. Own output dir to avoid overwriting.
P1_P2_OUT="${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines_instructions_only"
[ -d "${P1_P2_OUT}" ] || mkdir -p "${P1_P2_OUT}"
echo "Guidelines mode: instructions_only"
echo "P1→P2 output dir: ${P1_P2_OUT}"
# Enable wandb logging: WANDB=1 sbatch ... (requires wandb login or WANDB_API_KEY)
WANDB_ARG=""
[ -n "${WANDB}" ] && WANDB_ARG="--wandb"

$PYTHON_CMD src/prompt_generation/run_p1_to_p2_guidelines.py \
    --csv-path "${ROOT_DIR}/dataset/linguasafe.csv" \
    --language "${LANGUAGE}" \
    --guidelines-mode instructions_only \
    --output-dir "${P1_P2_OUT}" \
    --use-local-model \
    --local-model "${MODEL_SNAPSHOT}" \
    --temperature 0.3 \
    --max-tokens 512 \
    ${WANDB_ARG}

EXIT_CODE=$?
cleanup
exit $EXIT_CODE
