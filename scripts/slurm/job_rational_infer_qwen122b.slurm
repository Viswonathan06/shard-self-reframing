#!/usr/bin/bash
# Rational inference (SCR prompt-as-user, no thinking) for Qwen3.5-122B-A10B.
# Runs on gpu5 (8×RTX 6000 Ada) — only node with 122B cached and TP=8 viable.
# Produces rational_full_output for DNA and LinguaSafe p1 prompts so any
# downstream model can be fine-tuned on 122B teacher reasoning chains.
#
# Output:
#   output/rational_baseline/qwen35_122b_user_prompt/dna/inference_compliance.jsonl
#   output/rational_baseline/qwen35_122b_user_prompt/linguasafe/inference_compliance.jsonl
#
# Usage:
#   sbatch scripts/slurm/job_rational_infer_qwen122b.slurm

#SBATCH -n 1
#SBATCH --nodelist=gpu5
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --job-name=rational_infer_qwen122b
#SBATCH --output=$PROJECT_ROOT/logs/rational_infer_qwen122b_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/rational_infer_qwen122b_%j.err

set -euo pipefail

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME=/playpen/huggingface
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_TRUST_REMOTE_CODE=1
export TMPDIR=/tmp

PLAYPEN_HUB="$HF_HOME"
[[ -d "$PLAYPEN_HUB" ]] || { echo "ERROR: $PLAYPEN_HUB not accessible on $(hostname)"; exit 1; }

# Resolve 122B model path
MODEL_PATH=""
for snap in "${PLAYPEN_HUB}/models--Qwen--Qwen3.5-122B-A10B/snapshots"/*/; do
    [[ -d "$snap" ]] && [[ -f "${snap%/}/config.json" ]] && MODEL_PATH="${snap%/}" && break
done
[[ -n "$MODEL_PATH" ]] || { echo "ERROR: Qwen3.5-122B-A10B not found in $PLAYPEN_HUB"; exit 1; }

# p1 prompts — same across all models, use qwen35_9b's baseline run as source
DNA_INPUT="$ROOT/output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/qwen35_9b/p1_baseline_outputs.jsonl"
LS_INPUT="$ROOT/output/Linguasafe_Experiments/helpful_assistant_multimodel/qwen35_9b/p1_baseline_outputs.jsonl"
[[ -f "$DNA_INPUT" ]] || { echo "ERROR: DNA input not found: $DNA_INPUT"; exit 1; }
[[ -f "$LS_INPUT"  ]] || { echo "ERROR: LS input not found:  $LS_INPUT";  exit 1; }

DNA_OUT="$ROOT/output/rational_baseline/qwen35_122b_user_prompt/dna/inference_compliance.jsonl"
LS_OUT="$ROOT/output/rational_baseline/qwen35_122b_user_prompt/linguasafe/inference_compliance.jsonl"

echo "========================================================"
echo "Rational Inference: Qwen3.5-122B-A10B (prompt-as-user, no thinking)"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Model:     $MODEL_PATH"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

for pair in "dna:$DNA_INPUT:$DNA_OUT" "linguasafe:$LS_INPUT:$LS_OUT"; do
    dataset="${pair%%:*}"; rest="${pair#*:}"; input="${rest%%:*}"; output="${rest#*:}"
    mkdir -p "$(dirname "$output")"
    echo ">>> Dataset: $dataset  |  $(wc -l < "$input") rows"
    "$PYTHON" scripts/rational_baseline/rational_inference_vllm.py \
        --model                  "$MODEL_PATH" \
        --input                  "$input" \
        --output                 "$output" \
        --model-tag              "qwen35_122b_user_prompt" \
        --mode                   compliance \
        --prompt-as-user \
        --tensor-parallel-size   8 \
        --gpu-memory-utilization 0.90 \
        --max-model-len          16384 \
        --max-new-tokens         8192 \
        --temperature            0.3 \
        --batch-size             32
    echo "  Done: $(wc -l < "$output") rows -> $output"
    echo ""
done

echo "========================================================"
echo "Rational inference complete at: $(date)"
echo "========================================================"
