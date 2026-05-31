#!/usr/bin/bash
# Judge eval: LLaMA-3.1-8B SFT (non-reasoning test set) vs helpful_assistant baseline
# Input:  output/SFT/qwen35_122b_teacher/llama8b/inference.jsonl
# Baseline: output/Linguasafe_Experiments/helpful_assistant_multimodel/llama8b/p1_baseline_outputs.jsonl
# Output: output/Linguasafe_Experiments/evaluation/sft_vs_baseline_llama8b_gemma4_judge/

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_sft_llama8b
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

echo "=== SFT vs Baseline Eval: LLaMA-3.1-8B — Gemma 4 31B judge ==="
echo "Job: ${SLURM_JOB_ID:-local}  Node: $(hostname)  Start: $(date)"

# ── Gemma 4 31B judge ────────────────────────────────────────────────────────
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_MAX_MODEL_LEN=4096
export FLASHINFER_DISABLE_VERSION_CHECK=1

_COMMON_GEMMA_REV="/common/guam/hf_cache/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343"
_DEFAULT_GEMMA4_REV="$HF_HOME/models--google--gemma-4-31b-it/snapshots/145dc2508c480a64b47242f160d286cff94a2343"

_gemma_ok() {
  local d="${1%/}"
  [[ -d "$d" ]] && [[ -r "${d}/config.json" ]] || return 1
  local f; shopt -s nullglob
  for f in "${d}"/*.safetensors; do [[ -r "$f" ]] && { shopt -u nullglob; return 0; }; done
  shopt -u nullglob; return 1
}

_resolve_gemma4() {
  local root f rev
  _gemma_ok "${_COMMON_GEMMA_REV}" && echo "${_COMMON_GEMMA_REV}" && return 0
  for root in \
      "/common/guam/hf_cache/models--google--gemma-4-31b-it/snapshots" \
      "/common/guam/hf_cache/hub/models--google--gemma-4-31b-it/snapshots"; do
    [[ -d "$root" ]] || continue
    shopt -s nullglob
    for f in "${root}"/*/model.safetensors.index.json; do
      rev="$(dirname "$f")"
      if _gemma_ok "${rev}"; then echo "${rev}"; shopt -u nullglob; return 0; fi
    done
    shopt -u nullglob
  done
  _gemma_ok "${_DEFAULT_GEMMA4_REV}" && echo "${_DEFAULT_GEMMA4_REV}" && return 0
  for root in \
      "$HF_HOME/models--google--gemma-4-31b-it/snapshots" \
      "$HF_HOME/models--google--gemma-4-31B-it/snapshots"; do
    [[ -d "$root" ]] || continue
    shopt -s nullglob
    for f in "${root}"/*/model.safetensors.index.json; do
      rev="$(dirname "$f")"
      if _gemma_ok "${rev}"; then echo "${rev}"; shopt -u nullglob; return 0; fi
    done
    shopt -u nullglob
  done
  return 1
}

GEMMA4_PATH="$(_resolve_gemma4)" || { echo "ERROR: Gemma4 judge not found"; exit 1; }
echo "Judge: ${GEMMA4_PATH}"

export HF_HUB_OFFLINE=1

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
echo ""

SFT_RESPONSES="${ROOT_DIR}/output/SFT/qwen35_122b_teacher/llama8b/inference.jsonl"
BASELINE="${ROOT_DIR}/output/Linguasafe_Experiments/helpful_assistant_multimodel/llama8b/p1_baseline_outputs.jsonl"
OUT_DIR="${ROOT_DIR}/output/Linguasafe_Experiments/evaluation/sft_vs_baseline_llama8b_gemma4_judge"

[[ -f "${SFT_RESPONSES}" ]] || { echo "ERROR: SFT responses not found (run inference first): ${SFT_RESPONSES}"; exit 1; }
[[ -f "${BASELINE}" ]]      || { echo "ERROR: Baseline not found: ${BASELINE}"; exit 1; }

mkdir -p "${OUT_DIR}"

"${PYTHON}" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "${SFT_RESPONSES}" \
    --baseline       "${BASELINE}" \
    --output-dir     "${OUT_DIR}" \
    --judge-model    "${GEMMA4_PATH}" \
    --model-tag      "llama8b_sft"

echo ""
echo "======================================================"
echo "Done: $(date)"
echo "Output: ${OUT_DIR}"
echo "======================================================"
