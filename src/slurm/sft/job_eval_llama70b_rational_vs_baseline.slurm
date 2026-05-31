#!/usr/bin/bash
# Gemma-4 judge: llama70b rational SFT vs llama70b baseline
# on the rational_sft test set (612 rows).
# Runs after rational SFT inference (14558) and baseline inference complete.
#
#   sbatch --dependency=afterany:<rational_job>:<baseline_job> \
#          src/slurm/sft/job_eval_llama70b_rational_vs_baseline.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --nodelist=gpu4,gpu5
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_llama70b_rat_vs_base
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
export TMPDIR="/tmp/${USER}_eval70b_rat_vs_base_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID:-$$}"; mkdir -p "$TRITON_CACHE_DIR"

PYTHON="/common/guam/anaconda3/bin/python3"

INFER_DIR="${ROOT}/output/SFT/rational_sft/llama70b/inference"
RATIONAL_RAW="${INFER_DIR}/llama70b_rational.jsonl"
BASELINE_RAW="${INFER_DIR}/llama70b_baseline.jsonl"

[[ -f "$RATIONAL_RAW" ]] || { echo "ERROR: rational inference not found: $RATIONAL_RAW"; exit 1; }
[[ -f "$BASELINE_RAW" ]] || { echo "ERROR: baseline inference not found: $BASELINE_RAW"; exit 1; }
echo "Rational SFT: $RATIONAL_RAW ($(wc -l < $RATIONAL_RAW) rows)"
echo "Baseline:     $BASELINE_RAW ($(wc -l < $BASELINE_RAW) rows)"

# Remap rational SFT: response -> dpo_generated_response (fallback to sft_raw_output)
RATIONAL_REMAP="${INFER_DIR}/llama70b_rational_remapped.jsonl"
"$PYTHON" - "$RATIONAL_RAW" "$RATIONAL_REMAP" <<'PYEOF'
import sys, json
src, dst = sys.argv[1], sys.argv[2]
empty = 0
with open(src) as f, open(dst, "w") as out:
    for line in f:
        r = json.loads(line)
        resp = r.get("response") or r.get("sft_raw_output") or ""
        if not r.get("response"):
            empty += 1
        r["dpo_generated_response"] = resp
        out.write(json.dumps(r) + "\n")
print(f"Remapped {src} -> {dst}  (fell back to sft_raw_output for {empty} rows)")
PYEOF

# Remap baseline: output -> baseline_output (for ref-field lookup)
BASELINE_REMAP="${INFER_DIR}/llama70b_baseline_remapped.jsonl"
"$PYTHON" - "$BASELINE_RAW" "$BASELINE_REMAP" <<'PYEOF'
import sys, json
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f, open(dst, "w") as out:
    for line in f:
        r = json.loads(line)
        r.setdefault("baseline_output", r.get("output", ""))
        out.write(json.dumps(r) + "\n")
print(f"Remapped {src} -> {dst}")
PYEOF

echo "=========================================="
echo "Eval: llama70b rational SFT vs baseline"
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

# rational SFT (O2) vs baseline (O1)
VS_INPUT="${INFER_DIR}/rational_vs_baseline_eval_input.jsonl"
echo "Building rational SFT vs baseline eval input..."
"$PYTHON" scripts/build_dpo_vs_ref_eval_input.py \
  --dpo-jsonl  "$RATIONAL_REMAP" \
  --ref-jsonl  "$BASELINE_REMAP" \
  --ref-field  baseline_output \
  --output     "$VS_INPUT"

echo ""; echo "── rational SFT vs baseline ──"
echo "Input: $VS_INPUT ($(wc -l < $VS_INPUT) rows)"
OUT_DIR="${INFER_DIR}/rational_vs_baseline_gemma4_judge"
mkdir -p "$OUT_DIR"
"$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
  --refinement-jsonl "$VS_INPUT" \
  --output-dir       "$OUT_DIR" \
  --evals            all \
  --temperature      0.1 \
  --max-tokens       512 \
  --use-local-model \
  --judge-model      "$GEMMA4_PATH"

echo ""
echo "=========================================="
echo "All done: $(date)  ->  $OUT_DIR"
echo "=========================================="
