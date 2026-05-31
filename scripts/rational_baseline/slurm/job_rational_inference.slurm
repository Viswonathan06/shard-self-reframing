#!/usr/bin/bash
# Rational baseline inference (arxiv 2503.05021)
# Runs Self-Check Reasoning inference for a single model on DNA and/or LinguaSafe.
#
# Usage:
#   MODEL=llama8b    sbatch scripts/rational_baseline/slurm/job_rational_inference.slurm
#   MODEL=mistral7b  sbatch scripts/rational_baseline/slurm/job_rational_inference.slurm
#   MODEL=mistral24b sbatch scripts/rational_baseline/slurm/job_rational_inference.slurm
#   MODEL=phi4       sbatch scripts/rational_baseline/slurm/job_rational_inference.slurm
#   MODEL=qwen35_9b  sbatch scripts/rational_baseline/slurm/job_rational_inference.slurm
#   MODEL=qwen35_27b sbatch scripts/rational_baseline/slurm/job_rational_inference.slurm
#   MODEL=llama70b   sbatch scripts/rational_baseline/slurm/job_rational_inference.slurm
#
# Optional overrides:
#   DATASET=dna|linguasafe|both  (default: both — runs DNA then LinguaSafe sequentially)
#   MODE=rejection               (default: compliance)
#   MODEL_TAG=llama8b_custom     (overrides output dir name)
#   PROMPT_AS_USER=1             (prepend rational prompt into user message instead of system role)

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=08:00:00
#SBATCH --job-name=rational_%x
#SBATCH --output=$PROJECT_ROOT/logs/rational_%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/rational_%x_%j.err

set -euo pipefail

MODEL="${MODEL:?Set MODEL=llama8b|mistral7b|mistral24b|phi4|qwen35_9b|qwen35_27b|llama70b}"
DATASET="${DATASET:-both}"
MODE="${MODE:-compliance}"
MODEL_TAG="${MODEL_TAG:-${MODEL}}"
PROMPT_AS_USER="${PROMPT_AS_USER:-0}"

ROOT="$PROJECT_ROOT"
HOME_ROOT="$HOME"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0

# ── Dataset input paths ────────────────────────────────────────────────────────
# Using llama8b baseline outputs as the canonical prompt list for each dataset.
# (p1_id and p1 fields are dataset-wide — model outputs are ignored.)

DNA_INPUT="$HOME_ROOT/output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/llama8b/p1_baseline_outputs.jsonl"
LINGUASAFE_INPUT="$HOME_ROOT/output/Linguasafe_Experiments/helpful_assistant_multimodel/llama8b/p1_baseline_outputs.jsonl"

[[ "$DATASET" != "linguasafe" ]] && [[ ! -f "$DNA_INPUT" ]] && { echo "ERROR: DNA input not found: $DNA_INPUT" >&2; exit 1; }
[[ "$DATASET" != "dna"       ]] && [[ ! -f "$LINGUASAFE_INPUT" ]] && { echo "ERROR: LinguaSafe input not found: $LINGUASAFE_INPUT" >&2; exit 1; }

# ── Model paths (local snapshots where pre-cached, HF download otherwise) ─────

# Models with known /playpen snapshot paths (fast when node has them)
_USER_HF="$HF_HOME"
declare -A CACHED_MODELS=(
    [llama8b]="$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    [qwen35_9b]="${_USER_HF}/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    [qwen35_27b]="${_USER_HF}/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    [mistral7b]="${_USER_HF}/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
    [phi4]="${_USER_HF}/models--microsoft--phi-4/snapshots/932b33c0ec9ca189badeb22480721a8de9d0e006"
)

# HF repo IDs — used when cached path is absent or for models never pre-cached
declare -A HF_IDS=(
    [llama8b]="meta-llama/Llama-3.1-8B-Instruct"
    [qwen35_9b]="Qwen/Qwen3.5-9B"
    [qwen35_27b]="Qwen/Qwen3.5-27B"
    [mistral7b]="mistralai/Mistral-7B-Instruct-v0.3"
    [phi4]="microsoft/phi-4"
    [mistral24b]="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    [llama70b]="meta-llama/Llama-3.3-70B-Instruct"
)

_hf_download() {
    local hf_id="$1"
    echo "Downloading $hf_id from HF Hub..."
    BASE_MODEL=$("$PYTHON" -c "
from huggingface_hub import snapshot_download
import os
path = snapshot_download(repo_id='$hf_id', cache_dir=os.environ['HF_HOME'] + '/hub')
print(path)
" 2>/dev/null | tail -1)
    echo "Model at: $BASE_MODEL"
}

if [[ ! -v HF_IDS[$MODEL] ]]; then
    echo "ERROR: Unknown MODEL=$MODEL" >&2; exit 1
fi

if [[ -v CACHED_MODELS[$MODEL] ]] && [[ -f "${CACHED_MODELS[$MODEL]}/config.json" ]]; then
    BASE_MODEL="${CACHED_MODELS[$MODEL]}"
    echo "Using cached model: $BASE_MODEL"
else
    # Try to find any snapshot for this model under _USER_HF (handles stale hash)
    _hf_repo_dir="${_USER_HF}/models--$(echo "${HF_IDS[$MODEL]}" | tr '/' '--')/snapshots"
    _found=$(find "$_hf_repo_dir" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
    if [[ -n "$_found" ]] && [[ -f "$_found/config.json" ]]; then
        BASE_MODEL="$_found"
        echo "Using found snapshot: $BASE_MODEL"
    else
        _hf_download "${HF_IDS[$MODEL]}"
    fi
fi

# ── Model-specific settings ────────────────────────────────────────────────────

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

# phi4 has 10 KV heads — TP > 1 is invalid
INFER_TP=1
if [[ "$MODEL" != "phi4" ]]; then
    for _tp in 2 4 8; do (( _tp <= NUM_GPUS )) && INFER_TP=$_tp; done
fi

# Mistral models do not use a system prompt (tokenizer doesn't support it cleanly)
NO_SYSTEM_PROMPT_FLAG=""
[[ "$MODEL" == mistral* ]] && NO_SYSTEM_PROMPT_FLAG="--no-system-prompt"

PROMPT_AS_USER_FLAG=""
[[ "$PROMPT_AS_USER" == "1" ]] && PROMPT_AS_USER_FLAG="--prompt-as-user"

# Qwen3 thinking mode
THINKING_FLAG=""
[[ "$MODEL" == qwen35* ]] && THINKING_FLAG="--enable-thinking"

# Larger models need longer context window for reasoning chains
MAX_NEW_TOKENS=4096
MAX_MODEL_LEN=8192
[[ "$MODEL" == "llama70b" || "$MODEL" == "qwen35_27b" ]] && MAX_MODEL_LEN=4096

# ── Run helper ─────────────────────────────────────────────────────────────────

run_dataset() {
    local dataset_name="$1"
    local input_jsonl="$2"

    local output_dir="$ROOT/output/rational_baseline/${MODEL_TAG}/${dataset_name}"
    local output_jsonl="$output_dir/inference_${MODE}.jsonl"
    mkdir -p "$output_dir"

    echo "========================================================"
    echo "Rational baseline inference  [$dataset_name]"
    echo "Model:     $MODEL_TAG  ($BASE_MODEL)"
    echo "Mode:      $MODE"
    echo "Input:     $input_jsonl"
    echo "Output:    $output_jsonl"
    echo "GPUs:      $NUM_GPUS  (TP=$INFER_TP)"
    echo "========================================================"

    "$PYTHON" scripts/rational_baseline/rational_inference_vllm.py \
        --model                  "$BASE_MODEL" \
        --input                  "$input_jsonl" \
        --output                 "$output_jsonl" \
        --model-tag              "$MODEL_TAG" \
        --mode                   "$MODE" \
        --tensor-parallel-size   "$INFER_TP" \
        --gpu-memory-utilization 0.85 \
        --max-model-len          "$MAX_MODEL_LEN" \
        --max-new-tokens         "$MAX_NEW_TOKENS" \
        --temperature            0.3 \
        --batch-size             128 \
        $NO_SYSTEM_PROMPT_FLAG \
        $THINKING_FLAG \
        $PROMPT_AS_USER_FLAG

    echo "Done: $output_jsonl"
    echo "Rows written: $(wc -l < "$output_jsonl")"
    echo ""
}

# ── Execute ────────────────────────────────────────────────────────────────────

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

if [[ "$DATASET" == "dna" || "$DATASET" == "both" ]]; then
    run_dataset "dna" "$DNA_INPUT"
fi

if [[ "$DATASET" == "linguasafe" || "$DATASET" == "both" ]]; then
    run_dataset "linguasafe" "$LINGUASAFE_INPUT"
fi

echo "All datasets complete for model: $MODEL_TAG"
