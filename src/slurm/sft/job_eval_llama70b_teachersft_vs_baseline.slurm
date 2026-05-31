#!/usr/bin/bash
# Gemma-4 judge: llama70b teacher SFT (combined_dna_linguasafe) vs llama70b baseline
# on the combined_dna_linguasafe test set.
#
#   sbatch src/slurm/sft/job_eval_llama70b_teachersft_vs_baseline.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --nodelist=gpu4,gpu5
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_llama70b_teachersft_vs_base
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
export TMPDIR="/tmp/${USER}_eval70b_teachersft_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID:-$$}"; mkdir -p "$TRITON_CACHE_DIR"

PYTHON="/common/guam/anaconda3/bin/python3"

INFER_DIR="${ROOT}/output/SFT/qwen35_122b_teacher/combined_dna_linguasafe/inference"
SFT_RAW="${INFER_DIR}/llama70b.jsonl"
BASELINE_REF="${ROOT}/output/SFT/qwen35_122b_teacher/combined_dna_linguasafe/modelwise_test_baselines/llama33_70b.jsonl"

[[ -f "$SFT_RAW" ]]      || { echo "ERROR: SFT inference not found: $SFT_RAW"; exit 1; }
[[ -f "$BASELINE_REF" ]] || { echo "ERROR: baseline not found: $BASELINE_REF"; exit 1; }
echo "Teacher SFT: $SFT_RAW ($(wc -l < $SFT_RAW) rows)"
echo "Baseline:    $BASELINE_REF ($(wc -l < $BASELINE_REF) rows)"

# Remap response -> dpo_generated_response for the pairing script
SFT_REMAP="${INFER_DIR}/llama70b_remapped.jsonl"
"$PYTHON" - "$SFT_RAW" "$SFT_REMAP" <<'PYEOF'
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
echo "Eval: llama70b teacher SFT vs baseline"
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

VS_INPUT="${INFER_DIR}/teachersft_vs_baseline_eval_input.jsonl"
echo "Building teacher SFT vs baseline eval input..."
"$PYTHON" scripts/build_dpo_vs_ref_eval_input.py \
  --dpo-jsonl  "$SFT_REMAP" \
  --ref-jsonl  "$BASELINE_REF" \
  --ref-field  baseline_output \
  --output     "$VS_INPUT"

echo ""; echo "── teacher SFT vs baseline ──"
echo "Input: $VS_INPUT ($(wc -l < $VS_INPUT) rows)"
OUT_DIR="${INFER_DIR}/teachersft_vs_baseline_gemma4_judge"
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
echo "Done: $(date)  ->  $OUT_DIR"
echo "=========================================="
