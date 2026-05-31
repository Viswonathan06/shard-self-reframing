#!/usr/bin/bash
# SFT: Llama-3.3-70B-Instruct on qwen35_122b_teacher/combined_dna_linguasafe data.
#
#   sbatch src/slurm/sft/job_sft_llama70b_combined_dna_linguasafe.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=160G
#SBATCH --nodelist=gpu5
#SBATCH --time=24:00:00
#SBATCH --job-name=sft_llama70b_combined
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
export TMPDIR="/tmp/${USER}_sft70b_comb_${SLURM_JOB_ID:-$$}"; mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="/tmp/${USER}_triton_${SLURM_JOB_ID:-$$}"; mkdir -p "$TRITON_CACHE_DIR"

PYTHON="/common/guam/anaconda3/bin/python3"
NGPU="${SLURM_GPUS_ON_NODE:-4}"

echo "======================================================"
echo "SFT: Llama-3.3-70B-Instruct on combined_dna_linguasafe"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  GPUs: ${NGPU}  Start: $(date)"
echo "======================================================"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null || true

BASE_MODEL="$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b"
[[ -f "${BASE_MODEL}/config.json" ]] || { echo "ERROR: base model not found: ${BASE_MODEL}"; exit 1; }
echo "Model: ${BASE_MODEL}"

DATA_DIR="${ROOT}/output/SFT/qwen35_122b_teacher/combined_dna_linguasafe"

# ── Convert p1/o2 → messages format ──────────────────────────────────────────
SYS_PROMPT="You are a helpful and safe assistant. When a user asks about a sensitive topic, respond helpfully while avoiding harm. Always address the user's underlying need even if you cannot address the exact phrasing of their request."

PREP_DIR="${TMPDIR}/sft_prep"; mkdir -p "$PREP_DIR"
echo "Converting data to messages format..."

convert_jsonl() {
  local src="$1" dst="$2"
  "$PYTHON" - "$src" "$dst" "$SYS_PROMPT" <<'PYEOF'
import sys, json
src, dst, sys_prompt = sys.argv[1], sys.argv[2], sys.argv[3]
count = 0
with open(src) as f, open(dst, "w") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        p1 = (r.get("p1") or "").strip()
        o2 = (r.get("o2") or "").strip()
        if not p1 or not o2:
            continue
        out.write(json.dumps({
            "messages": [
                {"role": "system",    "content": sys_prompt},
                {"role": "user",      "content": p1},
                {"role": "assistant", "content": o2},
            ],
            "p1_id": r.get("p1_id", ""),
        }, ensure_ascii=False) + "\n")
        count += 1
print(f"  {src} -> {dst}: {count} rows")
PYEOF
}

convert_jsonl "${DATA_DIR}/train.jsonl" "${PREP_DIR}/train.jsonl"
convert_jsonl "${DATA_DIR}/val.jsonl"   "${PREP_DIR}/val.jsonl"
echo "Train: $(wc -l < ${PREP_DIR}/train.jsonl) rows | Val: $(wc -l < ${PREP_DIR}/val.jsonl) rows"

# ── Training ──────────────────────────────────────────────────────────────────
OUT_DIR="${ROOT}/models/llama70b_sft_combined_dna_linguasafe"
mkdir -p "$OUT_DIR"

echo ""
echo "Training | GPUs: ${NGPU} | out: ${OUT_DIR}"
# effective batch = NGPU * batch_size * grad_accum = 4 * 1 * 16 = 64

accelerate launch \
  --config_file src/slurm/dpo/fsdp_llama70b.yaml \
  --num_processes "${NGPU}" \
  scripts/train_sft_linguasafe.py \
    --base-model              "${BASE_MODEL}" \
    --train-jsonl             "${PREP_DIR}/train.jsonl" \
    --val-jsonl               "${PREP_DIR}/val.jsonl" \
    --output-dir              "${OUT_DIR}" \
    --max-length              1024 \
    --epochs                  3 \
    --batch-size              1 \
    --grad-accum              16 \
    --lr                      2e-5 \
    --warmup-ratio            0.05 \
    --lora-r                  16 \
    --lora-alpha              32 \
    --lora-dropout            0.05 \
    --logging-steps           10 \
    --save-steps              50 \
    --report-to               none \
    --run-name                "sft_llama70b_combined_dna_linguasafe" \
    --fsdp                    "full_shard auto_wrap" \
    --fsdp-transformer-cls    "LlamaDecoderLayer"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Adapter: ${OUT_DIR}"
echo "======================================================"
