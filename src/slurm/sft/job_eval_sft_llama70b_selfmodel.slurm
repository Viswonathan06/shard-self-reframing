#!/usr/bin/bash
# Gemma-4 judge: llama70b SFT (self_model) vs baseline — helpfulness + safety.
# Runs after job_sft_inference_llama70b_selfmodel.slurm.
#
#   sbatch --dependency=afterany:<infer_job> \
#          src/slurm/sft/job_eval_sft_llama70b_selfmodel.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --nodelist=gpu4,gpu5
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_llama70b_sft
#SBATCH --output=/common/guam/Safeshift/logs/%x_%j.out
#SBATCH --error=/common/guam/Safeshift/logs/%x_%j.err

set -euo pipefail
ROOT="/common/guam/Safeshift"; cd "$ROOT"; mkdir -p logs
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/${USER}_eval70b_sft_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID:-$$}"; mkdir -p "$TRITON_CACHE_DIR"

PYTHON="/common/guam/anaconda3/bin/python3"

SFT_RAW="${ROOT}/output/SFT/self_model/llama70b/inference/llama70b.jsonl"
[[ -f "$SFT_RAW" ]] || { echo "ERROR: SFT inference output not found: $SFT_RAW"; exit 1; }
echo "SFT output: $SFT_RAW ($(wc -l < $SFT_RAW) rows)"

# build_dpo_vs_ref_eval_input.py expects dpo_generated_response; remap from response
SFT_OUT="${ROOT}/output/SFT/self_model/llama70b/inference/llama70b_remapped.jsonl"
"$PYTHON" - "$SFT_RAW" "$SFT_OUT" <<'PYEOF'
import sys, json
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f, open(dst, "w") as out:
    for line in f:
        r = json.loads(line)
        r.setdefault("dpo_generated_response", r.get("response", ""))
        out.write(json.dumps(r) + "\n")
print(f"Remapped {src} -> {dst}")
PYEOF

echo "=========================================="
echo "Eval: llama70b SFT (self_model) vs baseline + vs o2"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || true

# Resolve Gemma-4
GEMMA4_PATH=""
for hub in /playpen/huggingface/hub/models--google--gemma-4-31B-it /playpen/huggingface/hub/models--google--gemma-4-31b-it; do
  [[ -d "${hub}/snapshots" ]] || continue
  for snap in "${hub}"/snapshots/*/; do
    cand="${snap%/}"
    [[ -f "${cand}/tokenizer.json" && -f "${cand}/config.json" ]] || continue
    (( $(ls "${cand}"/model-*-of-*.safetensors 2>/dev/null | wc -l) >= 1 )) || continue
    GEMMA4_PATH="$cand" && break
  done
  [[ -n "$GEMMA4_PATH" ]] && break
done
[[ -n "$GEMMA4_PATH" ]] || { echo "ERROR: Gemma-4 not found on $(hostname)"; exit 1; }
echo "Gemma-4: $GEMMA4_PATH"

export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_MAX_MODEL_LEN=16384
export VLLM_ENFORCE_EAGER=1
export VLLM_ONLY=1
export VLLM_NO_GPU_UTIL_CLAMP=1
export LOCAL_MODEL="$GEMMA4_PATH"

run_judge() {
  local label="$1" input="$2" outdir="$3"
  echo ""; echo "── ${label} ──"
  echo "Input: $input ($(wc -l < $input) rows) -> $outdir"
  mkdir -p "$outdir"
  "$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
    --refinement-jsonl "$input" \
    --output-dir       "$outdir" \
    --evals            all \
    --temperature      0.1 \
    --max-tokens       512 \
    --use-local-model \
    --judge-model      "$GEMMA4_PATH"
  echo "Done: $outdir"
}

# ── SFT vs baseline ───────────────────────────────────────────────────────────
BASELINE_REF="${ROOT}/output/SFT/self_model/llama70b/modelwise_test_baselines/llama70b.jsonl"
VS_BASELINE_INPUT="${ROOT}/output/SFT/self_model/llama70b/inference/vs_baseline_eval_input.jsonl"

echo "Building SFT vs baseline eval input..."
"$PYTHON" scripts/build_dpo_vs_ref_eval_input.py \
  --dpo-jsonl  "$SFT_OUT" \
  --ref-jsonl  "$BASELINE_REF" \
  --ref-field  baseline_output \
  --output     "$VS_BASELINE_INPUT"

run_judge "SFT vs baseline" \
  "$VS_BASELINE_INPUT" \
  "${ROOT}/output/SFT/self_model/llama70b/inference/vs_baseline_gemma4_judge"

# ── SFT vs DPO ────────────────────────────────────────────────────────────────
DPO_REF="${ROOT}/output/dpo_evaluation/dpo_llama70b_hhrlhf_on_selfmodel_test.jsonl"
VS_DPO_INPUT="${ROOT}/output/SFT/self_model/llama70b/inference/vs_dpo_eval_input.jsonl"

echo "Building SFT vs DPO eval input..."
"$PYTHON" scripts/build_dpo_vs_ref_eval_input.py \
  --dpo-jsonl  "$SFT_OUT" \
  --ref-jsonl  "$DPO_REF" \
  --ref-field  dpo_generated_response \
  --output     "$VS_DPO_INPUT"

run_judge "SFT vs DPO" \
  "$VS_DPO_INPUT" \
  "${ROOT}/output/SFT/self_model/llama70b/inference/vs_dpo_gemma4_judge"

echo ""
echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
