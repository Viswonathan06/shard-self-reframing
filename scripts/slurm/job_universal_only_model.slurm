#!/usr/bin/bash
# Universal-only pipeline for a single model (no category-specific guidelines).
#   Stage 1: P1→P2 (universal-only) for DNA + LinguaSafe
#   Stage 2: Refinement O2 for DNA + LinguaSafe
#   Stage 3: Gemma-4 judge O1 vs O2 for DNA + LinguaSafe
#
# Usage:
#   MODEL=llama33_70b  sbatch scripts/slurm/job_universal_only_model.slurm
#   MODEL=llama8b      sbatch scripts/slurm/job_universal_only_model.slurm
#   MODEL=mistral_24b  sbatch scripts/slurm/job_universal_only_model.slurm
#   MODEL=mistral_7b   sbatch scripts/slurm/job_universal_only_model.slurm
#   MODEL=phi4         sbatch scripts/slurm/job_universal_only_model.slurm
#   MODEL=qwen35_27b   sbatch scripts/slurm/job_universal_only_model.slurm
#   MODEL=qwen35_9b    sbatch scripts/slurm/job_universal_only_model.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu3
#SBATCH --time=24:00:00
#SBATCH --job-name=univ_only_%x
#SBATCH --output=$PROJECT_ROOT/logs/universal_only_%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/universal_only_%x_%j.err

set -eo pipefail

MODEL="${MODEL:?Set MODEL before sbatch, e.g. MODEL=phi4 sbatch ...}"

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB="${HF_HOME}/hub"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_AIO=0
export DS_SKIP_CUDA_CHECK=1
export TMPDIR=/tmp
export VLLM_USE_V1=0

# gpu1 has broken physical GPUs 3 and 7 — strip them
if [[ "$(hostname -s)" == "gpu1" ]]; then
    CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^3$' | grep -v '^7$' | paste -sd ',')
    export CUDA_VISIBLE_DEVICES
    echo "gpu1: filtered CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)

# ── Model registry ────────────────────────────────────────────────────────────
declare -A HF_IDS=(
    [llama33_70b]="meta-llama/Llama-3.3-70B-Instruct"
    [llama8b]="meta-llama/Llama-3.1-8B-Instruct"
    [mistral_24b]="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    [mistral_7b]="mistralai/Mistral-7B-Instruct-v0.3"
    [phi4]="microsoft/phi-4"
    [qwen35_27b]="Qwen/Qwen3.5-27B"
    [qwen35_9b]="Qwen/Qwen3.5-9B"
)

declare -A MODEL_TP=(
    [llama33_70b]=4
    [llama8b]=1
    [mistral_24b]=2
    [mistral_7b]=1
    [phi4]=1
    [qwen35_27b]=2
    [qwen35_9b]=1
)

HF_ID="${HF_IDS[$MODEL]:?Unknown MODEL=$MODEL}"
TP="${MODEL_TP[$MODEL]}"
(( TP > NUM_GPUS )) && TP=$NUM_GPUS
# Round TP down to the nearest power of 2 — transformer attention heads are
# powers of 2, so TP must be too (e.g. 3 GPUs after gpu1 filtering → TP=2)
_p=1; while (( _p * 2 <= TP )); do _p=$(( _p * 2 )); done; TP=$_p

# ── Resolve model path ────────────────────────────────────────────────────────
# Hardcoded overrides for models where dynamic find fails
declare -A DIRECT_PATHS=(
    [mistral_24b]="$HF_HOME/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506/snapshots/95a6d26c4bfb886c58daf9d3f7332c857cb27b43"
    [mistral_7b]="$HF_HOME/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71"
    [qwen35_27b]="$HF_HOME/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
    [qwen35_9b]="$HF_HOME/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
MODEL_PATH="${DIRECT_PATHS[$MODEL]:-}"
# Validate DIRECT_PATH on this node; fall back to find loop if missing
if [[ -n "$MODEL_PATH" ]] && [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "  DIRECT_PATH not valid on $(hostname -s), falling back to find loop..."
    MODEL_PATH=""
fi

if [[ -z "$MODEL_PATH" ]]; then
    DIR_NAME="models--$(echo "$HF_ID" | sed 's|/|--|g')"
    for cache in "$HF_HOME" "$HF_HOME" "$HF_HOME"; do
        _p=$(find "${cache}/${DIR_NAME}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
        if [[ -f "${_p}/config.json" ]]; then MODEL_PATH="$_p"; break; fi
    done
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Model not found in any cache — downloading $HF_ID to $HF_HOME ..."
    MODEL_PATH=$("$PYTHON" -c "
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='${HF_ID}', cache_dir='$HF_HOME')
print(path)
" 2>&1 | tail -1)
    echo "Downloaded to: $MODEL_PATH"
fi
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "ERROR: $MODEL not found after download attempt"; exit 1; }

# ── Gemma-4 judge path ────────────────────────────────────────────────────────
GEMMA4_PATH=$(find "$HF_HOME/models--google--gemma-4-31B-it/snapshots" \
    -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "${GEMMA4_PATH}/config.json" ]] || { echo "ERROR: Gemma-4 not found"; exit 1; }

# ── Static paths ──────────────────────────────────────────────────────────────
DNA_JSONL="dataset/donotanswer_no_outputs.jsonl"
LS_CSV="dataset/linguasafe.csv"
GUIDELINES_DIR="output/guidelines/guidelines_openai_one"
UNIVERSAL_GUIDELINES="src/prompts/guidelines.txt"
REWRITER_PROMPT="src/prompts/safety_rewriter_prompt_universal_only.txt"
REFINE_SYSTEM="src/prompts/p2_response_system_prompt_refinement.txt"

DNA_P2_JSONL="output/DNA_New_Experiments/p2s_without_cat/${MODEL}_p2.jsonl"
LS_P2_JSONL="output/Linguasafe_Experiments/p2s_without_cat/${MODEL}_p2.jsonl"
DNA_O1="output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/${MODEL}/p1_baseline_outputs.jsonl"
LS_O1="output/Linguasafe_Experiments/helpful_assistant_multimodel/${MODEL}/p1_baseline_outputs.jsonl"
DNA_REFINE_DIR="output/DNA_Per_Model_Experiments/refinement_without_cat/${MODEL}"
LS_REFINE_DIR="output/Linguasafe_Experiments/refinement_without_cat/${MODEL}"
DNA_JUDGE_DIR="output/DNA_Per_Model_Experiments/evaluation/without_cat_${MODEL}_gemma4_judge"
LS_JUDGE_DIR="output/Linguasafe_Experiments/evaluation/without_cat_${MODEL}_gemma4_judge"

mkdir -p "output/DNA_New_Experiments/p2s_without_cat" \
         "output/Linguasafe_Experiments/p2s_without_cat" \
         "$DNA_REFINE_DIR" "$LS_REFINE_DIR" "$DNA_JUDGE_DIR" "$LS_JUDGE_DIR"

echo "========================================================"
echo "Universal-only pipeline: $MODEL"
echo "Job ID:  ${SLURM_JOB_ID:-local}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "GPUs:    $NUM_GPUS  (TP=$TP)"
echo "Model:   $MODEL_PATH"
echo "Judge:   $GEMMA4_PATH"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: P1→P2 (universal-only, this model as rewriter)
# ══════════════════════════════════════════════════════════════════════════════

echo ">>> STAGE 1a: DNA P1→P2"
if [[ -f "$DNA_P2_JSONL" ]] && [[ $(wc -l < "$DNA_P2_JSONL") -ge 939 ]]; then
    echo "  SKIP: $(wc -l < "$DNA_P2_JSONL") rows done"
else
    "$PYTHON" scripts/run_p1_to_p2_vllm.py \
        --model                   "$MODEL_PATH" \
        --jsonl-path              "$DNA_JSONL" \
        --language                "en" \
        --guidelines-dir          "$GUIDELINES_DIR" \
        --universal-guidelines    "$UNIVERSAL_GUIDELINES" \
        --rewriter-prompt         "$REWRITER_PROMPT" \
        --output                  "$DNA_P2_JSONL" \
        --tensor-parallel-size    "$TP" \
        --gpu-memory-utilization  0.80 \
        --max-model-len           4096 \
        --temperature             0.3 \
        --max-tokens              512 \
        --batch-size              128
    echo "  Wrote: $(wc -l < "$DNA_P2_JSONL") rows → $DNA_P2_JSONL"
fi
echo ""

echo ">>> STAGE 1b: LinguaSafe P1→P2"
if [[ -f "$LS_P2_JSONL" ]] && [[ $(wc -l < "$LS_P2_JSONL") -ge 3267 ]]; then
    echo "  SKIP: $(wc -l < "$LS_P2_JSONL") rows done"
else
    "$PYTHON" scripts/run_p1_to_p2_vllm.py \
        --model                   "$MODEL_PATH" \
        --csv-path                "$LS_CSV" \
        --language                "en" \
        --guidelines-dir          "$GUIDELINES_DIR" \
        --universal-guidelines    "$UNIVERSAL_GUIDELINES" \
        --rewriter-prompt         "$REWRITER_PROMPT" \
        --output                  "$LS_P2_JSONL" \
        --tensor-parallel-size    "$TP" \
        --gpu-memory-utilization  0.80 \
        --max-model-len           4096 \
        --temperature             0.3 \
        --max-tokens              512 \
        --batch-size              128
    echo "  Wrote: $(wc -l < "$LS_P2_JSONL") rows → $LS_P2_JSONL"
fi
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Refinement — join O1 + P2, generate O2
# ══════════════════════════════════════════════════════════════════════════════

export VLLM_TENSOR_PARALLEL_SIZE=$TP
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=4096
export VLLM_USE_V1=0

run_refinement() {
    local label="$1"
    local o1_file="$2"
    local p2_file="$3"
    local out_dir="$4"

    echo ">>> STAGE 2: Refinement — $label"

    if [[ ! -f "$o1_file" ]]; then
        echo "  WARN: O1 not found: $o1_file — skipping"
        return
    fi

    local refinement_input="${out_dir}/refinement_input.jsonl"
    local refined_file="${out_dir}/baseline_outputs.jsonl"

    if [[ ! -f "$refinement_input" ]] || [[ $(wc -l < "$refinement_input") -lt 10 ]]; then
        echo "  Joining O1 + P2..."
        O1_FILE="$o1_file" P2_FILE="$p2_file" OUT_FILE="$refinement_input" \
        "$PYTHON" - <<'PYEOF'
import json, os
def norm(s): return " ".join((s or "").split()).strip()
with open(os.environ["O1_FILE"]) as f:
    o1_by_id = {str(r["p1_id"]): r for l in f if l.strip() and (r := json.loads(l))}
with open(os.environ["P2_FILE"]) as f:
    p2_by_id = {str(r.get("p1_id", r.get("id",""))): r for l in f if l.strip() and (r := json.loads(l))}
rows = []
for pid, o1 in o1_by_id.items():
    p2r = p2_by_id.get(pid)
    if p2r is None:
        continue
    rows.append({
        "p1_id":           pid,
        "p1":              norm(o1.get("p1","")),
        "p2":              norm(p2r.get("p2", p2r.get("output",""))),
        "baseline_output": norm(o1.get("output","")),
        "category":        o1.get("category", p2r.get("category","")),
        "lang":            o1.get("lang", p2r.get("lang","en")),
        "level":           o1.get("level", p2r.get("level")),
    })
with open(os.environ["OUT_FILE"], "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Joined {len(rows)} rows → {os.environ['OUT_FILE']}")
PYEOF
    else
        echo "  Join done ($(wc -l < "$refinement_input") rows)"
    fi

    local n_input
    n_input=$(wc -l < "$refinement_input")

    if [[ -f "$refined_file" ]] && [[ $(wc -l < "$refined_file") -ge "$n_input" ]]; then
        echo "  SKIP: already complete ($(wc -l < "$refined_file") rows)"
        return
    fi

    "$PYTHON" src/baselines/generate_p2_controlled_outputs.py \
        --input-jsonl         "$refinement_input" \
        --output-dir          "$out_dir" \
        --system-prompt-file  "$REFINE_SYSTEM" \
        --use-local-model \
        --local-model         "$MODEL_PATH" \
        --temperature         0.3 \
        --max-tokens          2000 \
        --refinement-mode \
        --exit-on-null
    echo "  Wrote: $(wc -l < "$refined_file") rows → $refined_file"
}

run_refinement "DNA"        "$DNA_O1" "$DNA_P2_JSONL" "$DNA_REFINE_DIR"
run_refinement "LinguaSafe" "$LS_O1"  "$LS_P2_JSONL"  "$LS_REFINE_DIR"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Gemma-4 judge
# ══════════════════════════════════════════════════════════════════════════════

_SLURM_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
[[ -n "$_SLURM_CUDA_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA_DEVICES"
JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=6144
export VLLM_USE_V1=0
echo "Judge: Gemma-4 TP=$JUDGE_TP"

run_judge() {
    local label="$1"
    local refine_dir="$2"
    local judge_dir="$3"
    local refined_file="${refine_dir}/baseline_outputs.jsonl"

    echo ">>> STAGE 3: Judge — $label"

    if [[ ! -f "$refined_file" ]]; then
        echo "  WARN: $refined_file not found — skipping"
        return
    fi

    local n_refined
    n_refined=$(wc -l < "$refined_file")
    local eval_file="${judge_dir}/eval_results.jsonl"
    if [[ -f "$eval_file" ]] && [[ $(wc -l < "$eval_file") -ge "$n_refined" ]]; then
        echo "  SKIP: already done ($(wc -l < "$eval_file") rows)"
        return
    fi

    "$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
        --refinement-jsonl  "$refined_file" \
        --output-dir        "$judge_dir" \
        --evals             all \
        --temperature       0.1 \
        --max-tokens        512 \
        --use-local-model \
        --judge-model       "$GEMMA4_PATH"
    echo "  Done: $judge_dir"
}

run_judge "DNA"        "$DNA_REFINE_DIR" "$DNA_JUDGE_DIR"
run_judge "LinguaSafe" "$LS_REFINE_DIR"  "$LS_JUDGE_DIR"

echo ""
echo "========================================================"
echo "Universal-only pipeline done: $MODEL  $(date)"
echo "========================================================"
