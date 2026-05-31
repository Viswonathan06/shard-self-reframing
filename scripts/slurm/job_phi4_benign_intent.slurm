#!/usr/bin/bash
# Benign-intent inference + judge for Phi-4 on DNA and LinguaSafe.
#
# Stage 1: Benign-intent inference (Phi-4, TP=1) — DNA and LinguaSafe
# Stage 2: Judge (Gemma-4 31B, TP=4) — baseline vs benign-intent, DNA and LinguaSafe
#
# Prerequisites:
#   output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/phi4/p1_baseline_outputs.jsonl
#   output/Linguasafe_Experiments/helpful_assistant_multimodel/phi4/p1_baseline_outputs.jsonl
#   output/DNA_New_Experiments/p2s/phi4_p2.jsonl
#   output/Linguasafe_Experiments/p2s/phi4_p2.jsonl
#
# Usage:
#   sbatch scripts/slurm/job_phi4_benign_intent.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=phi4_benign_intent
#SBATCH --output=$PROJECT_ROOT/logs/phi4_benign_intent_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/phi4_benign_intent_%j.err

set -euo pipefail

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

source ~/.bashrc 2>/dev/null || true
export ROOT_DIR="$ROOT"
export PYTHONPATH="${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export HF_HUB="${HF_HOME}/hub"
export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"

echo "========================================================"
echo "Phi-4 benign-intent pipeline"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "========================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ── Resolve Phi-4 path ────────────────────────────────────────────────────────
PHI4_PATH=$(find "${HF_HUB}/models--microsoft--phi-4/snapshots" \
    -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
if [[ ! -f "${PHI4_PATH}/config.json" ]]; then
    echo "Phi-4 not found in /playpen — downloading..."
    PHI4_PATH=$("$PYTHON" -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='microsoft/phi-4', cache_dir='${HF_HUB}'))
")
fi
[[ -f "${PHI4_PATH}/config.json" ]] || { echo "ERROR: Could not resolve Phi-4 path"; exit 1; }
echo "Phi-4: ${PHI4_PATH}"

# ── Paths ─────────────────────────────────────────────────────────────────────
BENIGN_SYSTEM="${ROOT}/src/prompts/p1_benign_intent_system_prompt.txt"

DNA_P2_JSONL="${ROOT}/output/DNA_New_Experiments/p2s/phi4_p2.jsonl"
DNA_O1="${ROOT}/output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/phi4/p1_baseline_outputs.jsonl"
DNA_BI_DIR="${ROOT}/output/DNA_Per_Model_Experiments/benign_intent_multimodel/phi4"
DNA_JUDGE_DIR="${ROOT}/output/DNA_Per_Model_Experiments/evaluation/benign_intent_multimodel_phi4_gemma4_judge"

LS_P2_JSONL="${ROOT}/output/Linguasafe_Experiments/p2s/phi4_p2.jsonl"
LS_O1="${ROOT}/output/Linguasafe_Experiments/helpful_assistant_multimodel/phi4/p1_baseline_outputs.jsonl"
LS_BI_DIR="${ROOT}/output/Linguasafe_Experiments/benign_intent_multimodel/phi4"
LS_JUDGE_DIR="${ROOT}/output/Linguasafe_Experiments/evaluation/benign_intent_multimodel_phi4_gemma4_judge"

mkdir -p "$DNA_BI_DIR" "$DNA_JUDGE_DIR" "$LS_BI_DIR" "$LS_JUDGE_DIR"

# Validate prerequisites
for f in "$BENIGN_SYSTEM" "$DNA_P2_JSONL" "$DNA_O1" "$LS_P2_JSONL" "$LS_O1"; do
    [[ -f "$f" ]] || { echo "ERROR: required file missing: $f"; exit 1; }
done

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Benign-intent inference — Phi-4, TP=1
# ══════════════════════════════════════════════════════════════════════════════
# phi4 has 10 KV heads — max valid tensor-parallel size is 1.
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=4096

echo ""
echo ">>> STAGE 1a: DNA benign-intent inference ($(wc -l < "$DNA_P2_JSONL") rows)"
if [[ -f "${DNA_BI_DIR}/p1_baseline_outputs.jsonl" ]] && \
   [[ $(wc -l < "${DNA_BI_DIR}/p1_baseline_outputs.jsonl") -ge 939 ]]; then
    echo "  SKIP: already complete ($(wc -l < "${DNA_BI_DIR}/p1_baseline_outputs.jsonl") rows)"
else
    "$PYTHON" src/baselines/generate_p1_baseline_outputs.py \
        --input-jsonl         "$DNA_P2_JSONL" \
        --output-dir          "$DNA_BI_DIR" \
        --system-prompt-file  "$BENIGN_SYSTEM" \
        --use-local-model \
        --local-model         "$PHI4_PATH" \
        --temperature         0.3 \
        --max-tokens          2000 \
        --exit-on-null
    echo "  Done: ${DNA_BI_DIR}/p1_baseline_outputs.jsonl"
fi

echo ""
echo ">>> STAGE 1b: LinguaSafe benign-intent inference ($(wc -l < "$LS_P2_JSONL") rows)"
if [[ -f "${LS_BI_DIR}/p1_baseline_outputs.jsonl" ]] && \
   [[ $(wc -l < "${LS_BI_DIR}/p1_baseline_outputs.jsonl") -ge 3267 ]]; then
    echo "  SKIP: already complete ($(wc -l < "${LS_BI_DIR}/p1_baseline_outputs.jsonl") rows)"
else
    "$PYTHON" src/baselines/generate_p1_baseline_outputs.py \
        --input-jsonl         "$LS_P2_JSONL" \
        --output-dir          "$LS_BI_DIR" \
        --system-prompt-file  "$BENIGN_SYSTEM" \
        --use-local-model \
        --local-model         "$PHI4_PATH" \
        --temperature         0.3 \
        --max-tokens          2000 \
        --exit-on-null
    echo "  Done: ${LS_BI_DIR}/p1_baseline_outputs.jsonl"
fi

echo ""
echo "Stage 1 complete at: $(date)"

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Judge — Gemma-4 31B, TP=4
# ══════════════════════════════════════════════════════════════════════════════
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_MAX_MODEL_LEN=4096
export VLLM_ENFORCE_EAGER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
unset HF_HOME HF_HUB_CACHE 2>/dev/null || true

# Resolve Gemma-4 path (checks common/playpen caches)
_gemma_ok() {
    local d="${1%/}"
    [[ -d "$d" ]] && [[ -r "${d}/config.json" ]] || return 1
    local f
    shopt -s nullglob
    for f in "${d}"/*.safetensors; do [[ -r "$f" ]] && { shopt -u nullglob; return 0; }; done
    shopt -u nullglob; return 1
}

GEMMA4_PATH=""
for _root in \
    "/common/guam/hf_cache/models--google--gemma-4-31b-it/snapshots" \
    "$HF_HOME/models--google--gemma-4-31b-it/snapshots" \
    "$HF_HOME/models--google--gemma-4-31B-it/snapshots"; do
    [[ -d "$_root" ]] || continue
    _cand=$(find "$_root" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
    _gemma_ok "$_cand" && { GEMMA4_PATH="$_cand"; break; }
done

[[ -n "$GEMMA4_PATH" ]] || { echo "ERROR: Gemma-4 judge model not found"; exit 1; }
echo "Gemma-4: ${GEMMA4_PATH}"

export LOCAL_MODEL="${GEMMA4_PATH}"
export HF_HOME="$(dirname "$(dirname "$(dirname "${GEMMA4_PATH}")")")"
[[ -d "${HF_HOME}/hub" ]] && export HF_HUB_CACHE="${HF_HOME}/hub" || export HF_HUB_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1

echo ""
echo ">>> STAGE 2a: Judge DNA (O1 vs benign-intent)"
"$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
    --o1-jsonl    "$DNA_O1" \
    --o2-jsonl    "${DNA_BI_DIR}/p1_baseline_outputs.jsonl" \
    --output-dir  "$DNA_JUDGE_DIR" \
    --evals       all \
    --temperature 0.1 \
    --max-tokens  512 \
    --use-local-model \
    --judge-model "$GEMMA4_PATH"
echo "  Results: $DNA_JUDGE_DIR"

echo ""
echo ">>> STAGE 2b: Judge LinguaSafe (O1 vs benign-intent)"
"$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
    --o1-jsonl    "$LS_O1" \
    --o2-jsonl    "${LS_BI_DIR}/p1_baseline_outputs.jsonl" \
    --output-dir  "$LS_JUDGE_DIR" \
    --evals       all \
    --temperature 0.1 \
    --max-tokens  512 \
    --use-local-model \
    --judge-model "$GEMMA4_PATH"
echo "  Results: $LS_JUDGE_DIR"

echo ""
echo "========================================================"
echo "Phi-4 benign-intent pipeline complete at: $(date)"
echo "DNA judge:        $DNA_JUDGE_DIR"
echo "LinguaSafe judge: $LS_JUDGE_DIR"
echo "========================================================"
