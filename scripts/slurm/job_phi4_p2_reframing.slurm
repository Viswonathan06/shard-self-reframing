#!/usr/bin/bash
# Stage 1 only: P1→P2 safety reframing for Phi-4, DNA + LinguaSafe.
# Uses TP=1 (phi-4 14B fits on 1×49GB GPU). Run this first; then
# job_phi4_full_pipeline.slurm will skip Stage 1 and proceed to refinement.
#
# Usage:
#   sbatch --nodelist=gpu1 scripts/slurm/job_phi4_p2_reframing.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=phi4_p2_reframe
#SBATCH --output=$PROJECT_ROOT/logs/phi4_p2_reframe_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/phi4_p2_reframe_%j.err

set -euo pipefail

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export HF_HUB="${HF_HOME}/hub"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"

# ── Phi-4 model path ──────────────────────────────────────────────────────────
PHI4_PATH=$(find "${HF_HUB}/models--microsoft--phi-4/snapshots" -maxdepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
if [[ ! -f "${PHI4_PATH}/config.json" ]]; then
    echo "Phi-4 not found — downloading..."
    PHI4_PATH=$("$PYTHON" -c "
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id='microsoft/phi-4', cache_dir='${HF_HUB}', resume_download=True))
")
fi
echo "Phi-4: ${PHI4_PATH}"

# ── Paths ─────────────────────────────────────────────────────────────────────
DNA_JSONL="dataset/donotanswer_no_outputs.jsonl"
LS_CSV="dataset/linguasafe.csv"
GUIDELINES_DIR="output/guidelines/guidelines_openai_one"
UNIVERSAL_GUIDELINES="src/prompts/guidelines.txt"

DNA_P2_JSONL="output/DNA_New_Experiments/p2s/phi4_p2.jsonl"
LS_P2_JSONL="output/Linguasafe_Experiments/p2s/phi4_p2.jsonl"

mkdir -p "$(dirname "$DNA_P2_JSONL")" "$(dirname "$LS_P2_JSONL")"

echo "========================================================"
echo "Phi-4 P1→P2 reframing"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── DNA ───────────────────────────────────────────────────────────────────────
echo ">>> DNA P1→P2 (939 rows)"
if [[ -f "$DNA_P2_JSONL" ]]; then
    DONE=$(python3 -c "
import json
rows = [json.loads(l) for l in open('$DNA_P2_JSONL') if l.strip()]
print(sum(1 for r in rows if r.get('p2')))
" 2>/dev/null || echo 0)
    echo "  Found $DONE valid rows already"
else
    DONE=0
fi
if [[ "$DONE" -ge 939 ]]; then
    echo "  SKIP: already complete"
else
    "$PYTHON" scripts/run_p1_to_p2_vllm.py \
        --model                   "$PHI4_PATH" \
        --jsonl-path              "$DNA_JSONL" \
        --language                "en" \
        --guidelines-dir          "$GUIDELINES_DIR" \
        --universal-guidelines    "$UNIVERSAL_GUIDELINES" \
        --output                  "$DNA_P2_JSONL" \
        --tensor-parallel-size    1 \
        --gpu-memory-utilization  0.85 \
        --max-model-len           4096 \
        --temperature             0.3 \
        --max-tokens              512 \
        --batch-size              128
    echo "  Done: $DNA_P2_JSONL"
fi

echo ""

# ── LinguaSafe ────────────────────────────────────────────────────────────────
echo ">>> LinguaSafe P1→P2 (3267 rows)"
if [[ -f "$LS_P2_JSONL" ]]; then
    DONE=$(python3 -c "
import json
rows = [json.loads(l) for l in open('$LS_P2_JSONL') if l.strip()]
print(sum(1 for r in rows if r.get('p2')))
" 2>/dev/null || echo 0)
    echo "  Found $DONE valid rows already"
else
    DONE=0
fi
if [[ "$DONE" -ge 3267 ]]; then
    echo "  SKIP: already complete"
else
    "$PYTHON" scripts/run_p1_to_p2_vllm.py \
        --model                   "$PHI4_PATH" \
        --csv-path                "$LS_CSV" \
        --language                "en" \
        --guidelines-dir          "$GUIDELINES_DIR" \
        --universal-guidelines    "$UNIVERSAL_GUIDELINES" \
        --output                  "$LS_P2_JSONL" \
        --tensor-parallel-size    1 \
        --gpu-memory-utilization  0.85 \
        --max-model-len           4096 \
        --temperature             0.3 \
        --max-tokens              512 \
        --batch-size              128
    echo "  Done: $LS_P2_JSONL"
fi

echo ""
echo "========================================================"
echo "P1→P2 reframing complete"
echo "DNA:        $DNA_P2_JSONL"
echo "LinguaSafe: $LS_P2_JSONL"
echo "========================================================"
