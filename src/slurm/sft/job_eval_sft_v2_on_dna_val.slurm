#!/bin/bash
#SBATCH --job-name=eval_sft_v2_dna_val
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err
#SBATCH --partition=a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=gpu3

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/common/guam/Safeshift}"
cd "$ROOT_DIR"

echo "=========================================="
echo "Eval: baseline vs SFT v2 on DNA v2 val set"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "=========================================="

source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# vLLM env. HF_HOME on /playpen is read-only for other users; point modules cache
# (which vLLM writes to during init) at a writable path. HF_HUB_CACHE still points
# at /playpen so we read snapshots from there.
export HF_HUB_CACHE=$HF_HOME
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/${USER}_hf_modules}"
mkdir -p "$HF_MODULES_CACHE"
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ENFORCE_EAGER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

PHASE1_PY="python"     # transformers 4.57 (matches sft_v2)
PHASE2_PY="/common/guam/anaconda3/bin/python3"                  # has vllm 0.19 + Gemma judge support

INPUT_FILE="${INPUT_FILE:-output/sft_evaluation/dna_val_v2_baseline_vs_sft_v2.jsonl}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-output/sft_evaluation/baseline_vs_sft_v2_dna_val_gemma4_judge}"

mkdir -p "$(dirname "$INPUT_FILE")" "$EVAL_OUT_DIR"

###############################################################################
# Phase 1: generate baseline + SFT v2 responses on the DNA v2 val set (1 GPU is
# enough; we asked for 4 just so phase 2 has them).
###############################################################################
echo ""
echo "--- Phase 1: inference (baseline + SFT v2) ---"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

# Auto-resolve a Llama-3.1-8B-Instruct snapshot that actually has all 4 shards.
# /playpen is per-node and several nodes have partial snapshots; the user's own
# $HF_HOME copy is reliably complete.
BASE_MODEL=""
for cand in \
  $HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  /playpen/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  $HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/
do
  cand="${cand%/}"
  [[ -d "$cand" ]] || continue
  shards=$(ls "$cand"/model-*-of-*.safetensors 2>/dev/null | wc -l)
  if (( shards >= 4 )); then
    BASE_MODEL="$cand"; break
  fi
done
[[ -n "$BASE_MODEL" ]] || { echo "ERROR: could not locate a complete Llama-3.1-8B-Instruct snapshot"; exit 1; }
echo "Resolved BASE_MODEL: $BASE_MODEL  (shards: $(ls "$BASE_MODEL"/model-*-of-*.safetensors 2>/dev/null | wc -l))"

# Skip phase 1 if a prior run already produced the inference file (idempotent rerun).
if [[ -s "$INPUT_FILE" ]]; then
  echo "Phase 1 skipped: $INPUT_FILE already exists ($(wc -l < "$INPUT_FILE") rows)"
else
  # Restrict phase-1 vLLM to one GPU; phase 2 will use all four for the Gemma judge.
  CUDA_VISIBLE_DEVICES=0 "$PHASE1_PY" scripts/inference_baseline_and_sft_v2_on_dna_val.py \
    --base-model "$BASE_MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --output "$INPUT_FILE"
fi

echo ""
echo "Inference output: $INPUT_FILE  ($(wc -l < "$INPUT_FILE") rows)"

###############################################################################
# Phase 2: judge with Gemma-4 across all 4 GPUs.
###############################################################################
echo ""
echo "--- Phase 2: Gemma-4 judge ---"
# Phase 2 needs HF_HOME for the judge config helpers; set it now (phase 1 doesn't need it).
export HF_HOME=/playpen/huggingface
export USE_LOCAL_MODEL=true
export JUDGE_MODEL="google/gemma-4-31B-it"
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_MAX_MODEL_LEN=4096

# Resolve a Gemma-4 snapshot that is actually usable (has tokenizer.json AND
# all model shards). Per-node /playpen caches are inconsistent; many nodes have
# stub snapshots (config.json only). Fall back to the model name so vLLM/HF will
# re-download to a writable cache.
GEMMA4_HUB="$HF_HOME/models--google--gemma-4-31B-it"
GEMMA4_PATH=""
if [[ -d "${GEMMA4_HUB}/snapshots" ]]; then
  for snap in "${GEMMA4_HUB}"/snapshots/*/; do
    cand="${snap%/}"
    [[ -d "$cand" ]] || continue
    [[ -f "${cand}/tokenizer.json" ]] || continue
    shards=$(ls "${cand}"/model-*-of-*.safetensors 2>/dev/null | wc -l)
    (( shards >= 1 )) || continue
    GEMMA4_PATH="$cand"; break
  done
fi
if [[ -n "$GEMMA4_PATH" ]]; then
  export LOCAL_MODEL="$GEMMA4_PATH"
  echo "Judge (local snapshot): $LOCAL_MODEL"
else
  # No usable local snapshot — let HF resolve it. Needs a writable HF_HUB_CACHE.
  export HF_HUB_CACHE="${HF_HUB_CACHE_WRITABLE:-$HF_HOME_hub}"
  mkdir -p "$HF_HUB_CACHE"
  export LOCAL_MODEL="google/gemma-4-31B-it"
  echo "Judge (will resolve via HF, cache=$HF_HUB_CACHE): $LOCAL_MODEL"
fi

"$PHASE2_PY" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
  --refinement-jsonl "$INPUT_FILE" \
  --output-dir       "$EVAL_OUT_DIR" \
  --evals            all \
  --temperature      0.1 \
  --max-tokens       512 \
  --use-local-model \
  --judge-model      "$LOCAL_MODEL" \
  --exit-on-null \
  --restart-eval

echo ""
echo "=========================================="
echo "Eval summary:"
[[ -f "$EVAL_OUT_DIR/eval_summary.json" ]] && cat "$EVAL_OUT_DIR/eval_summary.json"
echo ""
echo "End: $(date)"
echo "=========================================="
