#!/usr/bin/bash
# Inference: Llama-3.3-70B rational SFT with CoT prompt template.
# Wraps p1 as:
#   "Query: <p1>\n\nFirst reason through the query step by step. Then write your final answer after '### Response:'."
# Post-processes output to extract text after '### Response:' as the final response.
# Output: output/SFT/rational_sft/llama70b/inference/llama70b_rational_cot.jsonl
#
#   sbatch src/slurm/sft/job_sft_inference_llama70b_rational_cot.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --nodelist=gpu5
#SBATCH --time=6:00:00
#SBATCH --job-name=infer_llama70b_rat_cot
#SBATCH --output=/common/guam/Safeshift/logs/%x_%j.out
#SBATCH --error=/common/guam/Safeshift/logs/%x_%j.err

set -euo pipefail
ROOT="/common/guam/Safeshift"; cd "$ROOT"; mkdir -p logs
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME=$HF_HOME
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ENFORCE_EAGER=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_NO_GPU_UTIL_CLAMP=1
export TMPDIR="/tmp/${USER}_infer70b_rat_cot_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID:-$$}"; mkdir -p "$TRITON_CACHE_DIR"

PYTHON="/common/guam/anaconda3/bin/python3"
NGPU="${SLURM_GPUS_ON_NODE:-4}"

echo "======================================================"
echo "Inference: Llama-3.3-70B rational SFT (CoT prompt)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  TP=${NGPU}  Start: $(date)"
echo "======================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null || true

BASE_MODEL="$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
[[ -f "${BASE_MODEL}/config.json" ]] || { echo "ERROR: base model not found: ${BASE_MODEL}"; exit 1; }

SFT_ADAPTER="${ROOT}/models/llama70b_sft_rational"
RAW_INPUT="${ROOT}/output/SFT/rational_sft/llama70b/test.jsonl"
OUTPUT_DIR="${ROOT}/output/SFT/rational_sft/llama70b/inference"
PREP_INPUT="${TMPDIR}/rational_cot_input.jsonl"
RAW_OUTPUT="${TMPDIR}/rational_cot_raw.jsonl"
FINAL_OUTPUT="${OUTPUT_DIR}/llama70b_rational_cot.jsonl"

[[ -f "${SFT_ADAPTER}/adapter_config.json" ]] || { echo "ERROR: adapter not found: ${SFT_ADAPTER}"; exit 1; }
[[ -f "${RAW_INPUT}" ]] || { echo "ERROR: test set not found: ${RAW_INPUT}"; exit 1; }
mkdir -p "$OUTPUT_DIR"

echo "Base:    ${BASE_MODEL}"
echo "Adapter: ${SFT_ADAPTER}"
echo "Input:   ${RAW_INPUT} ($(wc -l < "${RAW_INPUT}") rows)"

# ── Step 1: copy input as-is (p1 unchanged) ──────────────────────────────────
cp "$RAW_INPUT" "$PREP_INPUT"
echo "Input: $(wc -l < $PREP_INPUT) rows"

# ── Step 2: run inference ─────────────────────────────────────────────────────
SYS_PROMPT="You are a helpful and safe assistant. When a user asks about a sensitive topic, respond helpfully while avoiding harm. Always address the user's underlying need even if you cannot address the exact phrasing of their request. First reason through the query step by step. Then write your final answer after '### Response:'."

"${PYTHON}" scripts/inference_sft_qwen_vllm.py \
    --base-model              "${BASE_MODEL}" \
    --sft-adapter             "${SFT_ADAPTER}" \
    --input-data              "${PREP_INPUT}" \
    --output                  "${RAW_OUTPUT}" \
    --system-prompt           "${SYS_PROMPT}" \
    --tensor-parallel-size    "${NGPU}" \
    --gpu-memory-utilization  0.85 \
    --max-model-len           8192 \
    --max-new-tokens          2048 \
    --temperature             0.3 \
    --lora-rank               16

echo "Raw inference done: $(wc -l < $RAW_OUTPUT) rows"

# ── Step 3: extract text after '### Response:' ────────────────────────────────
echo "Extracting final responses after '### Response:'..."
"$PYTHON" - "$RAW_OUTPUT" "$FINAL_OUTPUT" <<'PYEOF'
import sys, json, re

src, dst = sys.argv[1], sys.argv[2]
extracted = fallback = 0

with open(src) as f, open(dst, "w") as out:
    for line in f:
        r = json.loads(line)
        raw = r.get("sft_raw_output") or r.get("response") or ""
        # Extract after '### Response:'
        marker = "### Response:"
        if marker in raw:
            final = raw.split(marker, 1)[1].strip()
            extracted += 1
        else:
            # Fallback: use full raw output (strips <think> if present)
            final = r.get("response") or raw
            fallback += 1
        r["response"] = final
        r["dpo_generated_response"] = final
        out.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Extracted via marker: {extracted}  |  fallback: {fallback}")
PYEOF

echo ""
echo "======================================================"
echo "Done: $(date)  ->  ${FINAL_OUTPUT}"
wc -l "${FINAL_OUTPUT}" || true
# Quick quality check
"$PYTHON" - "$FINAL_OUTPUT" <<'PYEOF'
import json
with open(sys.argv[1] if len(sys.argv)>1 else "/dev/stdin") as f:
    rows = [json.loads(l) for l in f]
import sys
lens = [len(r.get("response","")) for r in rows]
empty = sum(1 for l in lens if l==0)
print(f"Non-empty: {len(rows)-empty}/{len(rows)}  avg_len={sum(lens)/len(lens):.0f} chars")
PYEOF
echo "======================================================"
