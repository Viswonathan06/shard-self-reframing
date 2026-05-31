#!/usr/bin/bash
# DNA (DoNotAnswer) pipeline for Qwen3.5-122B-A10B (gpu5, 8×RTX 6000 Ada, 384 GB VRAM).
# Requires gpu5 — only node confirmed to expose all 8 CUDA GPUs for TP=8.
# Model already cached on gpu5's /playpen from prior LinguaSafe run.
#
# Stage 1: P1→P2  — generate per-model reframed prompts from DNA dataset
# Stage 2: Baseline (O1) — helpful-assistant response to P1
# Stage 3: Join O1+P2 → refinement_input, then refinement with thinking mode
# Stage 4: Gemma4 judge — baseline vs refinement
#
# Outputs:
#   output/DNA_New_Experiments/p2s/qwen35_122b_p2.jsonl
#   output/DNA_New_Experiments/dna_baseline_multimodel/qwen35_122b/p1_baseline_outputs.jsonl
#   output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/qwen35_122b/baseline_outputs.jsonl
#   output/DNA_New_Experiments/evaluation/baseline_vs_refinement_qwen35_122b_gemma4_judge/
#
# Submit: sbatch src/slurm/p2/job_qwen35_122b_dna_pipeline.slurm

#SBATCH -n 1
#SBATCH --nodelist=gpu5
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=96:00:00
#SBATCH --job-name=qwen122b_dna_pipeline
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
echo "Qwen3.5-122B-A10B DNA Pipeline"
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"
echo "======================================================"
"${PYTHON}" --version

# Model cache: use node-local /playpen if it has >=250 GB free, else abort.
PLAYPEN_HUB="$HF_HOME"
if [[ -d "${PLAYPEN_HUB}" ]] && [[ -w "${PLAYPEN_HUB}" ]]; then
    PLAYPEN_FREE_GB=$(df -BG "${PLAYPEN_HUB}" | awk 'NR==2{gsub("G","",$4); print $4}')
    # gpu5: model already cached (234 GB); skip free-space check if model snapshot exists
    MODEL_SNAP_EXISTS=0
    for snap in "${PLAYPEN_HUB}/models--Qwen--Qwen3.5-122B-A10B/snapshots/"/*/; do
        [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && MODEL_SNAP_EXISTS=1 && break
    done
    if [[ "${MODEL_SNAP_EXISTS}" -eq 0 ]] && [[ "${PLAYPEN_FREE_GB}" -lt 250 ]]; then
        echo "ERROR: /playpen only has ${PLAYPEN_FREE_GB} GB free and model not cached — need >=250 GB."
        exit 1
    fi
    echo "/playpen free: ${PLAYPEN_FREE_GB} GB — sufficient."
    export HF_HOME="/playpen/huggingface"
else
    echo "ERROR: ${PLAYPEN_HUB} not accessible on $(hostname)."
    exit 1
fi
export HF_HUB_CACHE="${HF_HOME}/hub"

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
for snap in "${PLAYPEN_HUB}/models--Qwen--Qwen3.5-122B-A10B/snapshots/"/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && MODEL_PATH="${snap%/}" && break
done
if [[ -z "$MODEL_PATH" ]]; then
    echo "Model not found locally — downloading ${MODEL_REPO} to ${PLAYPEN_HUB}..."
    MODEL_PATH=$("${PYTHON}" -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='${MODEL_REPO}', cache_dir='${PLAYPEN_HUB}', resume_download=True)
print(p)
") || { echo "ERROR: Download failed"; exit 1; }
fi
echo "Model: ${MODEL_PATH}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Paths
DNA_JSONL="${ROOT_DIR}/dataset/donotanswer_no_outputs.jsonl"
GUIDELINES_DIR="${ROOT_DIR}/output/guidelines/guidelines_openai_one"
UNIVERSAL_GUIDELINES="${ROOT_DIR}/src/prompts/guidelines.txt"

P2S_DIR="${ROOT_DIR}/output/DNA_New_Experiments/p2s"
P2_WORK_DIR="${P2S_DIR}/_work_${MODEL_NAME}"
P2_FILE="${P2S_DIR}/${MODEL_NAME}_p2.jsonl"

O1_DIR="${ROOT_DIR}/output/DNA_New_Experiments/dna_baseline_multimodel/${MODEL_NAME}"
O1_FILE="${O1_DIR}/p1_baseline_outputs.jsonl"

REF_DIR="${ROOT_DIR}/output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/${MODEL_NAME}"
REF_INPUT="${REF_DIR}/refinement_input.jsonl"
REF_FILE="${REF_DIR}/baseline_outputs.jsonl"

BASELINE_PROMPT="${ROOT_DIR}/src/prompts/p1_helpful_assistant_system_prompt.txt"
REFINEMENT_PROMPT="${ROOT_DIR}/src/prompts/p2_response_system_prompt_refinement.txt"

N_EXPECTED=939
BATCH_SIZE="${BATCH_SIZE:-32}"

# WandB
export WANDB_PROJECT="multi-lingual-alignment-llms"
[[ -f "${ROOT_DIR}/.wandb_key" ]] && export WANDB_API_KEY=$(cat "${ROOT_DIR}/.wandb_key")

# ============================================================
# Stage 1: P1→P2
# ============================================================
echo ""
echo "----- Stage 1: P1→P2 -----"
N_P2=0
[[ -f "${P2_FILE}" ]] && N_P2=$(wc -l < "${P2_FILE}")
if [[ "${N_P2}" -ge "${N_EXPECTED}" ]]; then
    echo "Stage 1: already done (${N_P2} rows). Skipping."
else
    echo "Stage 1: Generating P2 reframings (${N_EXPECTED} DNA prompts)..."
    mkdir -p "${P2_WORK_DIR}"
    "${PYTHON}" "${ROOT_DIR}/src/prompt_generation/run_p1_to_p2_guidelines.py" \
        --jsonl-path           "${DNA_JSONL}" \
        --language             "en" \
        --guidelines-mode      full \
        --guidelines-dir       "${GUIDELINES_DIR}" \
        --universal-guidelines "${UNIVERSAL_GUIDELINES}" \
        --output-dir           "${P2_WORK_DIR}" \
        --use-local-model \
        --local-model          "${MODEL_PATH}" \
        --temperature 0.3 \
        --max-tokens 512

    PRODUCED=""
    for cand in \
        "${P2_WORK_DIR}/p1_p2_outputs_openai_one.jsonl" \
        "${P2_WORK_DIR}/p1_p2_outputs.jsonl" \
        "${P2_WORK_DIR}/p1_p2_outputs_v4.jsonl" \
        "${P2_WORK_DIR}/p1_p2_outputs_v3.jsonl" \
        "${P2_WORK_DIR}/p1_p2_outputs_v2.jsonl"; do
        [[ -f "${cand}" ]] && PRODUCED="${cand}" && break
    done
    if [[ -z "${PRODUCED}" ]]; then
        echo "ERROR: No p1_p2_outputs*.jsonl found under ${P2_WORK_DIR}"; exit 1
    fi
    cp -f "${PRODUCED}" "${P2_FILE}"
    N_P2=$(wc -l < "${P2_FILE}")
    echo "Stage 1 done: ${N_P2} rows -> ${P2_FILE}  ($(date))"
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
    export WANDB_RUN_NAME="baseline_${MODEL_NAME}_dna"
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
# Stage 3: Refinement (P1 + P2 + O1 → O2)
# ============================================================
echo ""
echo "----- Stage 3: Refinement -----"
mkdir -p "${REF_DIR}"

# Phase 3a: Join O1 + P2 -> refinement_input
N_RI=0
[[ -f "${REF_INPUT}" ]] && N_RI=$(wc -l < "${REF_INPUT}")
if [[ "${N_RI}" -ge "${N_O1}" ]]; then
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
    export WANDB_RUN_NAME="refinement_${MODEL_NAME}_dna"
    "${PYTHON}" src/baselines/generate_p2_controlled_outputs.py \
        --input-jsonl        "${REF_INPUT}" \
        --output-dir         "${REF_DIR}" \
        --system-prompt-file "${REFINEMENT_PROMPT}" \
        --use-local-model \
        --local-model        "${MODEL_PATH}" \
        --temperature 0.3 \
        --refinement-mode \
        --enable-thinking \
        --batch-size "${BATCH_SIZE}" \
        --max-tokens 6000 \
        --wandb
    echo "Phase 3b done at $(date)"
fi

# ============================================================
# Stage 4: Gemma4 judge — Baseline vs Refinement
# ============================================================
echo ""
echo "----- Stage 4: Gemma4 Judge (Baseline vs Refinement) -----"

EVAL_OUT_DIR="${ROOT_DIR}/output/DNA_New_Experiments/evaluation/baseline_vs_refinement_${MODEL_NAME}_gemma4_judge"
mkdir -p "${EVAL_OUT_DIR}"

N_EVAL=0
[[ -f "${EVAL_OUT_DIR}/eval_results.jsonl" ]] && N_EVAL=$(wc -l < "${EVAL_OUT_DIR}/eval_results.jsonl")
N_REF_NOW=$(wc -l < "${REF_FILE}" 2>/dev/null || echo 0)

if [[ "${N_EVAL}" -ge "${N_REF_NOW}" ]] && [[ "${N_EVAL}" -gt 0 ]]; then
    echo "Stage 4: already done (${N_EVAL} rows). Skipping."
else
    echo "Stage 4: Judging ${N_REF_NOW} pairs (done so far: ${N_EVAL})..."

    # Switch to Gemma4-31B: 4-way TP
    export VLLM_TENSOR_PARALLEL_SIZE=4
    export VLLM_GPU_MEMORY_UTILIZATION=0.70
    export VLLM_MAX_MODEL_LEN=8192
    unset VLLM_ENFORCE_EAGER 2>/dev/null || true

    GEMMA4_MODEL=""
    for snap in "${PLAYPEN_HUB}/models--google--gemma-4-31B-it/snapshots/"/*/; do
        [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && GEMMA4_MODEL="${snap%/}" && break
    done
    [[ -n "${GEMMA4_MODEL}" ]] || GEMMA4_MODEL="google/gemma-4-31B-it"
    echo "Judge model: ${GEMMA4_MODEL}"

    export JUDGE_MODEL="${GEMMA4_MODEL}"
    export LOCAL_MODEL="${GEMMA4_MODEL}"

    "${PYTHON}" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
        --refinement-jsonl "${REF_FILE}" \
        --output-dir       "${EVAL_OUT_DIR}" \
        --evals            all \
        --temperature      0.1 \
        --max-tokens       512 \
        --use-local-model \
        --judge-model      "${GEMMA4_MODEL}"

    echo "Stage 4 done at $(date)"
fi

echo ""
echo "======================================================"
echo "Pipeline complete | $(date)"
echo "P2:         $(wc -l < "${P2_FILE}" 2>/dev/null || echo '?') rows  -> ${P2_FILE}"
echo "Baseline:   $(wc -l < "${O1_FILE}" 2>/dev/null || echo '?') rows  -> ${O1_FILE}"
echo "Refinement: $(wc -l < "${REF_FILE}" 2>/dev/null || echo '?') rows  -> ${REF_FILE}"
echo "Eval:       $(wc -l < "${EVAL_OUT_DIR}/eval_results.jsonl" 2>/dev/null || echo '?') rows  -> ${EVAL_OUT_DIR}"
echo "======================================================"
