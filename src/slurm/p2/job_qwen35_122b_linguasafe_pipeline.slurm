#!/usr/bin/bash
# LinguaSafe pipeline for Qwen3.5-122B-A10B (gpu5, 8×RTX 6000 Ada, 384 GB VRAM).
#
# Stage 1: P1→P2 — skipped if qwen35_122b_p2.jsonl already has >=3267 rows
# Stage 2: Helpful-assistant baseline (O1)
# Stage 3: Join O1+P2 → refinement_input, then refinement with thinking mode
#
# Submit: sbatch src/slurm/p2/job_qwen35_122b_linguasafe_pipeline.slurm

#SBATCH -n 1
#SBATCH --nodelist=gpu5
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:8
#SBATCH --time=96:00:00
#SBATCH --job-name=qwen122b_ls_pipeline
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
unset -f python python3 2>/dev/null || true
unalias python python3 2>/dev/null || true

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
echo "======================================================"
echo "Qwen3.5-122B-A10B LinguaSafe Pipeline"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version

# HF cache — use gpu5 local /playpen
if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
else
    export HF_HOME="${HOME}/.cache/huggingface"
fi
export HF_HUB_CACHE="${HF_HOME}/hub"
echo "HF_HOME: ${HF_HOME}"

# vLLM settings for 122B MoE across 8 GPUs
export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE=1
export VLLM_ONLY=1
export VLLM_TENSOR_PARALLEL_SIZE=8
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLLM_MAX_MODEL_LEN=16384
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Resolve model path
MODEL_NAME="qwen35_122b"
MODEL_REPO="Qwen/Qwen3.5-122B-A10B"
MODEL_PATH=""
for snap in "${HF_HOME}/hub/models--Qwen--Qwen3.5-122B-A10B/snapshots/"/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && MODEL_PATH="${snap%/}" && break
done
[[ -n "$MODEL_PATH" ]] || MODEL_PATH="${MODEL_REPO}"
echo "Model: ${MODEL_PATH}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Paths
P2_FILE="${ROOT_DIR}/output/Linguasafe_Experiments/p2s/${MODEL_NAME}_p2.jsonl"
O1_DIR="${ROOT_DIR}/output/Linguasafe_Experiments/helpful_assistant_multimodel/${MODEL_NAME}"
O1_FILE="${O1_DIR}/p1_baseline_outputs.jsonl"
REF_DIR="${ROOT_DIR}/output/Linguasafe_Experiments/linguasafe_refinement_from_baseline_multimodel/${MODEL_NAME}"
REF_INPUT="${REF_DIR}/refinement_input.jsonl"
REF_FILE="${REF_DIR}/baseline_outputs.jsonl"

BASELINE_PROMPT="${ROOT_DIR}/src/prompts/p1_helpful_assistant_system_prompt.txt"
REFINEMENT_PROMPT="${ROOT_DIR}/src/prompts/p2_response_system_prompt_refinement.txt"
N_EXPECTED=3267
BATCH_SIZE="${BATCH_SIZE:-32}"

# ============================================================
# Stage 1: P1->P2 — skip if already done
# ============================================================
echo ""
echo "----- Stage 1: P1->P2 -----"
N_P2=0
[[ -f "${P2_FILE}" ]] && N_P2=$(wc -l < "${P2_FILE}")
if [[ "${N_P2}" -ge "${N_EXPECTED}" ]]; then
    echo "Stage 1: already done (${N_P2} rows). Skipping."
else
    echo "ERROR: P2 file has only ${N_P2}/${N_EXPECTED} rows."
    echo "Run job_linguasafe_p1_to_p2_per_model_all_models.slurm first."
    exit 1
fi

# ============================================================
# Stage 2: Helpful-assistant baseline (O1)
# ============================================================
echo ""
echo "----- Stage 2: Baseline (O1) -----"
mkdir -p "${O1_DIR}"
N_O1=0
[[ -f "${O1_FILE}" ]] && N_O1=$(wc -l < "${O1_FILE}")
if [[ "${N_O1}" -ge "${N_P2}" ]]; then
    echo "Stage 2: already done (${N_O1} rows). Skipping."
else
    echo "Stage 2: Baseline inference (${N_P2} rows, done so far: ${N_O1})..."
    export WANDB_RUN_NAME="baseline_${MODEL_NAME}"
    "${PYTHON}" src/baselines/generate_p1_baseline_outputs.py \
        --input-jsonl        "${P2_FILE}" \
        --output-dir         "${O1_DIR}" \
        --system-prompt-file "${BASELINE_PROMPT}" \
        --use-local-model \
        --local-model        "${MODEL_PATH}" \
        --temperature 0.3 \
        --max-tokens 2000 \
        --batch-size "${BATCH_SIZE}" \
        --wandb
    echo "Stage 2 done at $(date)"
fi

N_O1=$(wc -l < "${O1_FILE}" 2>/dev/null || echo 0)
echo "O1 rows: ${N_O1}"

# ============================================================
# Stage 3: Refinement
# ============================================================
echo ""
echo "----- Stage 3: Refinement -----"
mkdir -p "${REF_DIR}"

# Phase 3a: Join O1 + P2 -> refinement_input
N_RI=0
[[ -f "${REF_INPUT}" ]] && N_RI=$(wc -l < "${REF_INPUT}")
if [[ "${N_RI}" -ge "${N_P2}" ]]; then
    echo "Phase 3a (join): already done (${N_RI} rows). Skipping."
else
    echo "Phase 3a: Joining O1 with P2 records..."
    O1_FILE_ENV="${O1_FILE}" P2_FILE_ENV="${P2_FILE}" OUT_FILE_ENV="${REF_INPUT}" \
    "${PYTHON}" - <<'PYEOF'
import json, os

o1_file  = os.environ["O1_FILE_ENV"]
p2_file  = os.environ["P2_FILE_ENV"]
out_file = os.environ["OUT_FILE_ENV"]

def norm(s):
    return " ".join((s or "").split()).strip()

o1_by_id = {}
o1_by_p1 = {}
with open(o1_file) as f:
    for line in f:
        r = json.loads(line)
        pid = r.get("p1_id")
        p1k = norm(r.get("p1", ""))
        out = r.get("output", "")
        if pid is not None:
            o1_by_id[str(pid)] = out
        if p1k:
            o1_by_p1[p1k] = out

written = skipped = 0
with open(p2_file) as fin, open(out_file, "w") as fout:
    for line in fin:
        r = json.loads(line)
        pid = r.get("p1_id")
        p1k = norm(r.get("p1", ""))
        baseline = None
        if pid is not None and str(pid) in o1_by_id:
            baseline = o1_by_id[str(pid)]
        elif p1k and p1k in o1_by_p1:
            baseline = o1_by_p1[p1k]
        if baseline is None:
            skipped += 1
            continue
        r["baseline_output"] = baseline
        fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        written += 1

print(f"Joined {written} records -> {out_file}  (skipped {skipped})")
PYEOF
    [[ $? -ne 0 ]] && { echo "ERROR: Join step failed"; exit 1; }
fi

N_RI=$(wc -l < "${REF_INPUT}" 2>/dev/null || echo 0)

# Phase 3b: Refinement with thinking mode
N_REF=0
[[ -f "${REF_FILE}" ]] && N_REF=$(wc -l < "${REF_FILE}")
if [[ "${N_REF}" -ge "${N_RI}" ]]; then
    echo "Phase 3b (refinement): already done (${N_REF} rows). Skipping."
else
    echo "Phase 3b: Refinement with thinking mode (${N_RI} rows, done: ${N_REF})..."
    export WANDB_RUN_NAME="refinement_${MODEL_NAME}"
    "${PYTHON}" src/baselines/generate_p2_controlled_outputs.py \
        --input-jsonl        "${REF_INPUT}" \
        --output-dir         "${REF_DIR}" \
        --system-prompt-file "${REFINEMENT_PROMPT}" \
        --use-local-model \
        --local-model        "${MODEL_PATH}" \
        --temperature 0.3 \
        --max-tokens 2000 \
        --refinement-mode \
        --enable-thinking \
        --batch-size "${BATCH_SIZE}" \
        --max-tokens 6000 \
        --wandb
    echo "Phase 3b done at $(date)"
fi

echo ""
echo "======================================================"
echo "Pipeline complete | $(date)"
echo "P2:         ${N_P2} rows  -> ${P2_FILE}"
echo "Baseline:   $(wc -l < "${O1_FILE}" 2>/dev/null || echo '?') rows  -> ${O1_FILE}"
echo "Refinement: $(wc -l < "${REF_FILE}" 2>/dev/null || echo '?') rows  -> ${REF_FILE}"
echo "======================================================"
