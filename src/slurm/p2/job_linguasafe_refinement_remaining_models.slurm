#!/bin/bash
#SBATCH --job-name=linguasafe_refinement_remaining
#SBATCH --output=$PROJECT_ROOT/logs/linguasafe_refinement_remaining_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/linguasafe_refinement_remaining_%j.err
#SBATCH --partition=a6000
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

cd $PROJECT_ROOT

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export USE_LOCAL_MODEL=true
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME/hub"
export CUDA_VISIBLE_DEVICES=0,1,2,3
# vLLM V1 multiprocess workers: spawn avoids fork+CUDA deadlocks on some stacks (torch/vLLM).
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_TENSOR_PARALLEL_SIZE=4
# Default vLLM budget (27B / 24B / 7B): parent CUDA leaves ~35 GiB free; 0.88×47 GiB was too high for 70B workers.
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
export VLLM_MAX_MODEL_LEN=4096
export VLLM_NO_GPU_UTIL_CLAMP=1
# VLLM_USE_V1=0 only works on older vLLM; 0.15+ is V1-only — keep spawn/NCCL tweaks above instead.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_DISABLE_CHUNKED_PREFILL="${VLLM_DISABLE_CHUNKED_PREFILL:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_ENFORCE_EAGER=1
# Default: vLLM (do not set this). Only for emergencies if vLLM cannot start: REFINEMENT_USE_HF=1 sbatch ...
REFINEMENT_USE_HF="${REFINEMENT_USE_HF:-0}"

# source $CONDA_PREFIX/../etc/profile.d/conda.sh
conda activate base || true

REFINE_DIR="$(pwd)/output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_guidelines_multimodel"
# llama8b refinement_input.jsonl is used as template (has all prompt metadata)
TEMPLATE_JSONL="${REFINE_DIR}/llama8b/refinement_input.jsonl"

# HF hub layout: --cache-dir X → X/hub/models--Org--Repo/snapshots/<hash>/
resolve_hf_snapshot() {
    local repo_id="$1"
    local slug
    slug=$(echo "$repo_id" | sed 's|/|--|g')
    local hub_root="${HF_HUB_CACHE:-${HF_HOME}/hub}"
    local repo_dir="${hub_root}/models--${slug}"
    if [[ ! -d "${repo_dir}/snapshots" ]]; then
        echo ""
        return 1
    fi
    local best=""
    for d in "${repo_dir}/snapshots"/*/; do
        [[ -f "${d}config.json" ]] || continue
        best="${d%/}"
    done
    echo "${best}"
}

# Build refinement_input.jsonl for a model using its own helpful_assistant baseline outputs
prepare_refinement_input() {
    local model_name="$1"
    local out_file="${REFINE_DIR}/${model_name}/refinement_input.jsonl"
    local baseline_ckpt="$(pwd)/output/Linguasafe_Experiments/helpful_assistant_multimodel/${model_name}/checkpoint.jsonl"

    if [[ -f "$out_file" ]]; then
        echo "refinement_input.jsonl already exists for $model_name, skipping."
        return 0
    fi

    if [[ ! -f "$baseline_ckpt" ]]; then
        echo "ERROR: baseline checkpoint not found: $baseline_ckpt"
        return 1
    fi

    echo "Preparing refinement_input.jsonl for $model_name ..."
    python3 - <<PYEOF
import json

template_file = "${TEMPLATE_JSONL}"
baseline_file = "${baseline_ckpt}"
out_file      = "${out_file}"

# Load baseline outputs keyed by p1_id
baseline = {}
with open(baseline_file) as f:
    for line in f:
        r = json.loads(line)
        pid = str(r.get("p1_id", ""))
        baseline[pid] = r.get("output", r.get("response", r.get("baseline_output", "")))

# Merge: use template for all prompt metadata, replace baseline_output with this model's output
written = 0
skipped = 0
with open(template_file) as fin, open(out_file, "w") as fout:
    for line in fin:
        r = json.loads(line)
        pid = str(r.get("p1_id", ""))
        if pid in baseline:
            r["baseline_output"] = baseline[pid]
            fout.write(json.dumps(r) + "\n")
            written += 1
        else:
            skipped += 1

print(f"Done: {written} rows written, {skipped} skipped (no baseline match) for ${model_name}")
PYEOF
    echo "refinement_input.jsonl prepared for $model_name ($(wc -l < "$out_file") lines)"
}

run_refinement_model() {
    local model_name="$1"
    local local_path="$2"

    echo ""
    echo "========================================"
    echo "Starting: $model_name at $(date)"
    echo "========================================"

    mkdir -p "${REFINE_DIR}/${model_name}"

    # Skip if already complete (expected rows = template refinement_input line count)
    local ckpt="${REFINE_DIR}/${model_name}/checkpoint.jsonl"
    local expected
    expected=$(wc -l < "${TEMPLATE_JSONL}")
    if [[ -f "$ckpt" ]] && [[ $(wc -l < "$ckpt") -ge "${expected}" ]]; then
        echo "$model_name already completed (${expected} rows), skipping."
        return 0
    fi

    # Prepare refinement input from this model's own baseline outputs
    prepare_refinement_input "$model_name" || return 1

    if [[ -z "${local_path}" ]] || [[ ! -f "${local_path}/config.json" ]]; then
        echo "ERROR: Invalid or empty model path for ${model_name}: '${local_path}'"
        return 1
    fi
    echo "Using model: $local_path"
    export LOCAL_MODEL="$local_path"

    echo "Running refinement for $model_name ..."
    EXTRA_ARGS=()
    if [[ "${REFINEMENT_USE_HF}" == "1" ]] || [[ "${REFINEMENT_USE_HF}" == "true" ]]; then
        EXTRA_ARGS+=(--use-hf)
        echo "REFINEMENT_USE_HF: using HuggingFace backend (skip vLLM)."
    fi
    python3 src/baselines/generate_p2_controlled_outputs.py \
        --input-jsonl "${REFINE_DIR}/${model_name}/refinement_input.jsonl" \
        --output-dir  "${REFINE_DIR}/${model_name}" \
        --system-prompt-file "src/prompts/p2_response_system_prompt_refinement_with_guidelines.txt" \
        --use-local-model \
        --local-model "$local_path" \
        --temperature 0.3 \
        --max-tokens 2000 \
        --refinement-mode \
        --exit-on-null \
        "${EXTRA_ARGS[@]}"

    echo "Completed: $model_name at $(date)"
}

echo "Starting LinguaSafe guidelines refinement at $(date)"

# qwen35_27b: download into HF hub layout (--cache-dir HF_HOME → HF_HOME/hub/...)
echo "=== Qwen/Qwen3.5-27B ==="
QWEN27B_PATH="$(resolve_hf_snapshot "Qwen/Qwen3.5-27B")"
if [[ -z "${QWEN27B_PATH}" ]]; then
    echo "Downloading Qwen/Qwen3.5-27B to ${HF_HOME} (srun single task) ..."
    srun -N1 -n1 -c8 --mem=64G hf download Qwen/Qwen3.5-27B \
        --cache-dir "${HF_HOME}" || {
        echo "srun hf download failed; retrying without srun ..."
        hf download Qwen/Qwen3.5-27B \
            --cache-dir "${HF_HOME}"
    }
    QWEN27B_PATH="$(resolve_hf_snapshot "Qwen/Qwen3.5-27B")"
fi
echo "Qwen3.5-27B snapshot: ${QWEN27B_PATH}"
run_refinement_model "qwen35_27b" "$QWEN27B_PATH" || true

# llama33_70b: use 0.73 util so vLLM V1 workers pass the free-memory check.
# max_model_len raised to 4096 so prompts longer than 2048 tokens don't get 0 output slots.
export VLLM_GPU_MEMORY_UTILIZATION=0.73
run_refinement_model "llama33_70b" \
    "$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b" || true

# After llama, GPUs may have ~12 GiB occupied by other processes.
# 35/47 = 0.74 free → use 0.72 so vLLM V1 startup check passes.
export VLLM_GPU_MEMORY_UTILIZATION=0.72

# mistral_7b: already cached
run_refinement_model "mistral_7b" \
    "$HF_HOME/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71" || true

# mistral_24b: in shared cache
run_refinement_model "mistral_24b" \
    "$HF_HOME/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/95a6d26c4bfb886c58daf9d3f7332c857cb27b43" || true

export VLLM_GPU_MEMORY_UTILIZATION=0.88

echo "All remaining LinguaSafe refinement completed at $(date)"
