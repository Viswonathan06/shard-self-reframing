#!/usr/bin/bash
# SFT: Llama-3.3-70B-Instruct on qwen35_122b teacher data (combined DNA + LinguaSafe).
# Input:  output/SFT/qwen35_122b_teacher/combined_dna_linguasafe/{train,val}.jsonl
# Output: output/SFT/qwen35_122b_teacher/combined_dna_linguasafe/models/llama70b/
# Uses QLoRA (4-bit) + DDP on 4x48GB GPUs; max-length 512 keeps peak GPU mem ~40GB.
#
# Usage:
#   sbatch scripts/slurm/job_sft_llama70b_teacher_combined.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --job-name=sft_llama70b_teacher_combined
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="$PROJECT_ROOT"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true

export PYTHONPATH="${PYTHONPATH:-}:${ROOT_DIR}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"

NUM_GPUS=4
# Effective batch = 4 GPUs x bs=1 x grad_accum=16 = 64
GRAD_ACCUM=16

echo "======================================================"
echo "SFT: Llama-3.3-70B-Instruct (combined teacher data)"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Locate base model ──────────────────────────────────────────────────────────
MODEL_PATH=""
for search_root in \
    "$HF_HOME" \
    "$HF_HOME" \
    "$HF_HOME"; do
    for snap in "${search_root}/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/"/*/; do
        snap="${snap%/}"
        [[ -d "$snap" ]] && \
        [[ -f "${snap}/model.safetensors.index.json" ]] && \
        MODEL_PATH="${snap}" && break 2
    done
done
[[ -z "$MODEL_PATH" ]] && { echo "ERROR: Llama-3.3-70B-Instruct not found in any cache"; exit 1; }
echo "Model: ${MODEL_PATH}"
echo ""

DATA_DIR="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/combined_dna_linguasafe"
PREP_DIR="${DATA_DIR}/prepared"
OUT_DIR="${DATA_DIR}/models/llama70b"
mkdir -p "$PREP_DIR" "$OUT_DIR"

# ── Stage 0: add messages field (p1 -> user, o2 -> assistant) ─────────────────
echo ">>> Stage 0: Preparing data with messages field"
"$PYTHON" - << 'PYEOF'
import json
from pathlib import Path

ROOT = Path("$PROJECT_ROOT")
DATA_DIR = ROOT / "output/SFT/qwen35_122b_teacher/combined_dna_linguasafe"
PREP_DIR = DATA_DIR / "prepared"

SYSTEM_PROMPT = (
    "You are a helpful and safe assistant. When a user asks about a sensitive topic, "
    "respond helpfully while avoiding harm. Always address the user's underlying need "
    "even if you cannot address the exact phrasing of their request."
)

def convert(in_path, out_path):
    rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    converted = []
    skipped = 0
    for r in rows:
        o2 = (r.get("o2") or "").strip()
        p1 = (r.get("p1") or "").strip()
        if not o2 or not p1:
            skipped += 1
            continue
        r["messages"] = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": p1},
            {"role": "assistant", "content": o2},
        ]
        converted.append(r)
    with open(out_path, "w") as f:
        for r in converted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {in_path.name}: {len(converted)} rows written ({skipped} skipped) -> {out_path}")

convert(DATA_DIR / "train.jsonl", PREP_DIR / "train.jsonl")
convert(DATA_DIR / "val.jsonl",   PREP_DIR / "val.jsonl")
PYEOF
echo ">>> Stage 0 complete"
echo ""

# ── Stage 1: QLoRA fine-tuning ─────────────────────────────────────────────────
echo ">>> Stage 1: Training"
MASTER_PORT=$(( 29500 + (${SLURM_JOB_ID:-0} % 1000) ))
"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port="${MASTER_PORT}" \
    scripts/train_sft_linguasafe.py \
        --base-model             "${MODEL_PATH}" \
        --train-jsonl            "${PREP_DIR}/train.jsonl" \
        --val-jsonl              "${PREP_DIR}/val.jsonl" \
        --output-dir             "${OUT_DIR}" \
        --max-length             512 \
        --epochs                 3 \
        --batch-size             1 \
        --grad-accum             "${GRAD_ACCUM}" \
        --lr                     2e-5 \
        --warmup-ratio           0.05 \
        --lora-r                 16 \
        --lora-alpha             32 \
        --lora-dropout           0.05 \
        --logging-steps          10 \
        --save-steps             100 \
        --load-in-4bit \
        --run-name               "sft_llama70b_teacher_combined" \
        --wandb-project          "linguasafe-sft"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Model saved to: ${OUT_DIR}"
echo "======================================================"
