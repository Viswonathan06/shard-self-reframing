#!/usr/bin/bash
# Gemma-4 judge: rational baseline vs helpful-assistant baseline.
# Compares rational_response (O2) vs helpful-assistant output (O1) on
# both DNA and LinguaSafe.
#
# Usage:
#   MODEL=llama8b    sbatch scripts/slurm/job_rational_judge.slurm
#   MODEL=mistral7b  sbatch scripts/slurm/job_rational_judge.slurm
#   MODEL=mistral24b sbatch scripts/slurm/job_rational_judge.slurm
#   MODEL=phi4       sbatch scripts/slurm/job_rational_judge.slurm
#   MODEL=qwen35_9b  sbatch scripts/slurm/job_rational_judge.slurm
#   MODEL=qwen35_27b sbatch scripts/slurm/job_rational_judge.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --nodelist=gpu1
#SBATCH --time=12:00:00
#SBATCH --job-name=rational_judge_%x
#SBATCH --output=$PROJECT_ROOT/logs/rational_judge_%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/rational_judge_%x_%j.err

set -euo pipefail

MODEL="${MODEL:?Set MODEL=llama8b|mistral7b|mistral24b|phi4|qwen35_9b|qwen35_27b}"

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0

# ── Map rational model tag → helpful_assistant dir name ───────────────────────
case "$MODEL" in
    mistral7b)  HA_MODEL="mistral_7b"  ;;
    mistral24b) HA_MODEL="mistral_24b" ;;
    *)          HA_MODEL="$MODEL"      ;;
esac

# ── Gemma-4 path ──────────────────────────────────────────────────────────────
_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1 || true)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma4 not found in $_GEMMA4_BASE"; exit 1; }

source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true

JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENFORCE_EAGER=0

echo "========================================================"
echo "Rational baseline judge: $MODEL"
echo "Job:      ${SLURM_JOB_ID}"
echo "Judge:    $GEMMA4_PATH  (TP=$JUDGE_TP)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Helper: join rational output with helpful-assistant baseline ───────────────
build_merged() {
    local dataset="$1"       # dna | linguasafe
    local rational_file="$2"
    local baseline_file="$3"
    local out_file="$4"

    "$PYTHON" - <<PYEOF
import json

with open("$rational_file") as f:
    rational = {str(r["p1_id"]): r for l in f if (r := json.loads(l))}

with open("$baseline_file") as f:
    baseline = {str(r["p1_id"]): r for l in f if (r := json.loads(l))}

rows = []
for pid, rat in rational.items():
    bl = baseline.get(pid)
    if bl is None:
        continue
    rows.append({
        "p1_id":          pid,
        "p1":             rat.get("p1", bl.get("p1", "")),
        "p2":             "",
        "baseline_output": bl.get("output", ""),
        "output":         rat.get("rational_response", ""),
        "category":       rat.get("category", ""),
        "level":          rat.get("level"),
        "dataset":        "$dataset",
    })

with open("$out_file", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Merged {len(rows)} rows -> $out_file")
PYEOF
}

run_judge() {
    local label="$1"
    local merged_file="$2"
    local judge_dir="$3"

    mkdir -p "$judge_dir"
    echo ">>> Judge — $label  ($(wc -l < "$merged_file") rows)"
    "$PYTHON" src/evaluation_ranking/evaluate_safety_helpfulness_single_random.py \
        --refinement-jsonl  "$merged_file" \
        --output-dir        "$judge_dir" \
        --evals             helpfulness,harmlessness \
        --temperature       0.1 \
        --max-tokens        512 \
        --use-local-model \
        --judge-model       "$GEMMA4_PATH" \
        --batch-size        64
    echo "    Done -> $judge_dir  ($(wc -l < "$judge_dir/eval_results.jsonl" 2>/dev/null || echo 0) rows)"
    echo ""
}

# ── DNA ───────────────────────────────────────────────────────────────────────
DNA_RATIONAL="output/rational_baseline/${MODEL}/dna/inference_compliance.jsonl"
DNA_BASELINE="output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/${HA_MODEL}/p1_baseline_outputs.jsonl"
DNA_MERGED="/tmp/rational_judge_${MODEL}_dna_merged.jsonl"
DNA_JUDGE_DIR="output/DNA_Per_Model_Experiments/evaluation/rational_baseline_${MODEL}_gemma4_judge"

[[ -f "$DNA_RATIONAL" ]] || { echo "ERROR: $DNA_RATIONAL not found"; exit 1; }
[[ -f "$DNA_BASELINE" ]] || { echo "ERROR: $DNA_BASELINE not found"; exit 1; }

echo ">>> Building DNA merged file..."
build_merged "dna" "$DNA_RATIONAL" "$DNA_BASELINE" "$DNA_MERGED"
run_judge "DNA" "$DNA_MERGED" "$DNA_JUDGE_DIR"

# ── LinguaSafe ────────────────────────────────────────────────────────────────
LS_RATIONAL="output/rational_baseline/${MODEL}/linguasafe/inference_compliance.jsonl"
LS_BASELINE="output/Linguasafe_Experiments/helpful_assistant_multimodel/${HA_MODEL}/p1_baseline_outputs.jsonl"
LS_MERGED="/tmp/rational_judge_${MODEL}_ls_merged.jsonl"
LS_JUDGE_DIR="output/Linguasafe_Experiments/evaluation/rational_baseline_${MODEL}_gemma4_judge"

[[ -f "$LS_RATIONAL" ]] || { echo "ERROR: $LS_RATIONAL not found"; exit 1; }
[[ -f "$LS_BASELINE" ]] || { echo "ERROR: $LS_BASELINE not found"; exit 1; }

echo ">>> Building LinguaSafe merged file..."
build_merged "linguasafe" "$LS_RATIONAL" "$LS_BASELINE" "$LS_MERGED"
run_judge "LinguaSafe" "$LS_MERGED" "$LS_JUDGE_DIR"

echo "========================================================"
echo "Rational judge done: $MODEL"
echo "DNA:        $DNA_JUDGE_DIR"
echo "LinguaSafe: $LS_JUDGE_DIR"
echo "========================================================"
