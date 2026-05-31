#!/usr/bin/bash
# LLM-as-judge evaluation: dual relevance + comparative helpfulness/harmlessness.
#
# Defaults: Qwen3.5-27B judge; Gemma O1 + Gemma O2 from Qwen27B P2 (controlled_openai_one); eval dir below.
# Run from repo root (SLURM_SUBMIT_DIR; Slurm copies this script to /var/spool).
#   cd /path/to/Multi-Lingual-Alignment-in-LLMs && sbatch src/slurm/evaluation/job_evaluate_safety_helpfulness.slurm
# Llama-70B P2:   O2_JSONL=$ROOT/output/p2_controlled/llama70b/p2_gemma3_27b_it_controlled_llama70b/baseline_outputs.jsonl \\
#                 EVAL_OUT_DIR=$ROOT/output/evaluation/gemma_ctrl_p2_llama70b_judge_qwen_promptupd sbatch ...
#
# Qwen3.5-9B (P1-only baseline O1 vs controlled O2; judge = Qwen3.5-27B by default):
#   USE_QWEN35_9B_CTRL_EVAL=1 sbatch ...
#
# Qwen3.5-9B on DoNotAnswer (P1 baseline O1 vs Llama70b-P2 controlled O2; judge = Qwen3.5-27B):
#   O1_JSONL=$ROOT/output/p2_baselines/p2_qwen35_9b_donotanswer/p1_baseline_outputs.jsonl \\
#   O2_JSONL=$ROOT/output/p2_qwen35_9b_controlled_donotanswer_llama70b/baseline_outputs.jsonl \\
#   EVAL_OUT_DIR=$ROOT/output/evaluation/qwen35_9b_donotanswer_baseline_vs_ctrl_qwen27b_judge \\
#   JUDGE_MODEL=Qwen/Qwen3.5-27B sbatch ...
#
# Change judge model:
#   JUDGE_MODEL=google/gemma-3-27b-it sbatch ...
#   JUDGE_MODEL=Qwen/Qwen3.5-27B sbatch ...
#   JUDGE_MODEL=/path/to/local/snapshot sbatch ...
#
# Use Azure API instead of local model:
#   USE_API=1 sbatch ...
#
# vLLM-only (default): VLLM_ONLY=1 — if vLLM fails to load, the job fails (no silent HuggingFace fallback).
# Do not set USE_HF unless you intentionally want the judge on HF/transformers only (skips vLLM entirely).
#
# If vLLM EngineCore / WorkerProc init fails, try (alone or together):
#   VLLM_GPU_MEMORY_UTILIZATION=0.72 VLLM_MAX_MODEL_LEN=4096 VLLM_ENFORCE_EAGER=1 sbatch ...
#   VLLM_ENABLE_V1_MULTIPROCESSING=0 sbatch ...
#   Add to script: #SBATCH --exclusive   (avoid foreign processes on the GPUs)
#
# This script pins the job to gpu5 (#SBATCH --nodelist=gpu5). Override when submitting:
#   sbatch --nodelist=gpu1 ...   or remove the line for any node.
#
# Select evaluations:
#   EVALS=relevance,helpfulness sbatch ...        (default: all)
#
# Other overrides:
#   JUDGE_MODEL_PATH=/path/to/snapshot  (skip hub lookup; still downloads if missing)
#   O2_JSONL=... O1_JSONL=... EVAL_OUT_DIR=... MAX_TOKENS=1024 TEMPERATURE=0.1 sbatch ...
#   Do not export O2_JSONL="${ROOT_DIR}/output/..." if ROOT_DIR is unset on submit host (becomes "/output/...").
#   MAX_RECORDS=50 NO_WANDB=1 sbatch ...   (wandb is on by default in the Python script)
#   LOG_METRICS_EVERY=25  (optional env: print + wandb eval summary every K completed rows; 0=off)
#   EXIT_ON_NULL=1 sbatch ...
#   VLLM_TENSOR_PARALLEL_SIZE=4 VLLM_GPU_MEMORY_UTILIZATION=0.80 VLLM_MAX_MODEL_LEN=8192 sbatch ...
#
# Free playpen space (one sbatch = ONE compute node; purge only runs on the node you landed on):
#   GEMMA3_PURGE_MATCH_SLURM_JOB=1 sbatch ...   # purge Gemma cache on that node only, if present
#   GEMMA3_PURGE_NODELIST=gpu1,gpu3 sbatch ...  # purge only if this job's hostname is in the list
# To clear Gemma on every GPU node, submit one job per node (see purge_gemma3_27b_hf_cache_one_node.slurm).

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --nodelist=gpu5
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --job-name=eval_safety_help
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Repo root: Slurm runs a *copy* under /var/spool — prefer SLURM_SUBMIT_DIR, then ROOT_DIR, then script path.
_slurm_is_repo() {
  [[ -n "$1" ]] && [[ -d "$1/src/evaluation_ranking" ]]
}

ROOT_DIR="${ROOT_DIR:-}"
if ! _slurm_is_repo "$ROOT_DIR"; then
  ROOT_DIR=""
fi
if [[ -z "$ROOT_DIR" ]] && _slurm_is_repo "${SLURM_SUBMIT_DIR:-}"; then
  ROOT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
fi
if [[ -z "$ROOT_DIR" ]]; then
  _SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  _ROOT_CAND="$(cd "${_SLURM_DIR}/../../.." && pwd)"
  if _slurm_is_repo "$_ROOT_CAND"; then
    ROOT_DIR="${_ROOT_CAND}"
  fi
fi
if [[ -z "$ROOT_DIR" ]] || ! _slurm_is_repo "$ROOT_DIR"; then
  echo "ERROR: Could not resolve repo root (need src/evaluation_ranking). Submit from repo or set ROOT_DIR."
  exit 1
fi
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

_slurm_fix_repo_path() {
  local p="$1"
  if [[ "$p" == /output/* ]]; then
    echo "${ROOT_DIR}/output/${p#/output/}"
  elif [[ -n "$p" ]] && [[ "$p" != /* ]]; then
    echo "${ROOT_DIR}/${p#./}"
  else
    echo "$p"
  fi
}

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# ---------- HF cache ----------
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
  else
    PLAYPEN_USER_HF="${PLAYPEN_USER_HF:-/playpen/${USER}/huggingface}"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" 2>/dev/null && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
    else
      export HF_HOME="${HOME}/.cache/huggingface"
    fi
  fi
fi

# ---------- Optional: drop Gemma-3-27B HF cache on specific nodes only ----------
purge_gemma3_cache_if_requested() {
  local hub="${HF_HOME}/hub"
  local gem="${hub}/models--google--gemma-3-27b-it"
  [[ -d "${gem}" ]] || return 0

  local h_s
  h_s=$(hostname -s 2>/dev/null || hostname)
  local do_purge=0

  if [[ -n "${GEMMA3_PURGE_NODELIST:-}" ]]; then
    IFS=',' read -ra _nodes <<< "${GEMMA3_PURGE_NODELIST}"
    for n in "${_nodes[@]}"; do
      n="${n// /}"
      [[ -z "$n" ]] && continue
      if [[ "$h_s" == "$n" ]] || [[ "$(hostname 2>/dev/null)" == "$n" ]]; then
        do_purge=1
        break
      fi
    done
  fi

  if [[ "${GEMMA3_PURGE_MATCH_SLURM_JOB:-}" == "1" ]] && [[ "${do_purge}" -eq 0 ]]; then
    if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol &>/dev/null; then
      local n
      while read -r n; do
        [[ -z "$n" ]] && continue
        if [[ "$h_s" == "$n" ]]; then
          do_purge=1
          break
        fi
      done < <(scontrol show hostnames "${SLURM_JOB_NODELIST}" 2>/dev/null)
    elif [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
      [[ "${SLURM_JOB_NODELIST}" == *"${h_s}"* ]] && do_purge=1
    fi
  fi

  if [[ "${do_purge}" -eq 1 ]]; then
    echo "GEMMA3 cache purge on ${h_s}: rm -rf ${gem}"
    rm -rf "${gem}"
  fi
}
purge_gemma3_cache_if_requested

# ---------- Judge model (repo id or absolute path) ----------
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-27B}"
USE_API="${USE_API:-}"

LOCAL_MODEL_ARGS=""
JUDGE_MODEL_RESOLVED=""

# ---------- Input / output (default = Gemma ctrl vs Gemma P1 baseline; judge = Qwen3.5-27B) ----------
if [[ "${USE_QWEN35_9B_CTRL_EVAL:-}" == "1" ]]; then
  _SLURM_QWEN27B_P2_O2="${ROOT_DIR}/output/p2_qwen35_9b_controlled_openai_one_qwen27b_guidelines/baseline_outputs.jsonl"
  _SLURM_QWEN27B_P2_O1="${ROOT_DIR}/output/p2_baselines/p2_qwen35_9b/p1_baseline_outputs.jsonl"
  _SLURM_QWEN27B_P2_EVAL_OUT="${ROOT_DIR}/output/evaluation/qwen35_9b_p1baseline_vs_ctrl_qwen27b_judge"
else
  _SLURM_QWEN27B_P2_O2="${ROOT_DIR}/output/p2_controlled/qwen27b/p2_gemma3_27b_it_controlled_openai_one/baseline_outputs.jsonl"
  _SLURM_QWEN27B_P2_O1="${ROOT_DIR}/output/p1_baseline_gemma3_27b_it_openai_one/p1_baseline_outputs.jsonl"
  _SLURM_QWEN27B_P2_EVAL_OUT="${ROOT_DIR}/output/evaluation/gemma_ctrl_p2_qwen27b_judge_qwen_promptupd"
fi

O2_JSONL="${O2_JSONL:-${_SLURM_QWEN27B_P2_O2}}"
O1_JSONL="${O1_JSONL:-${_SLURM_QWEN27B_P2_O1}}"
O2_JSONL="$(_slurm_fix_repo_path "${O2_JSONL}")"
O1_JSONL="$(_slurm_fix_repo_path "${O1_JSONL}")"
INPUT_JSONL="${INPUT_JSONL:-}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${_SLURM_QWEN27B_P2_EVAL_OUT}}"
EVAL_OUT_DIR="$(_slurm_fix_repo_path "${EVAL_OUT_DIR}")"
EVALS="${EVALS:-all}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_RECORDS="${MAX_RECORDS:-}"

WANDB_ARG=""
[[ "${NO_WANDB:-}" == "1" ]] && WANDB_ARG="--no-wandb"
EXIT_NULL_ARG=""
[[ -n "${EXIT_ON_NULL}" ]] && EXIT_NULL_ARG="--exit-on-null"

# ---------- Validate ----------
if [[ -z "${INPUT_JSONL}" ]] && [[ ! -f "${O2_JSONL}" ]]; then
  echo "ERROR: O2 JSONL not found: ${O2_JSONL}"
  exit 1
fi
if [[ -z "${INPUT_JSONL}" ]] && [[ -n "${O1_JSONL}" ]] && [[ ! -f "${O1_JSONL}" ]]; then
  echo "ERROR: O1 JSONL not found: ${O1_JSONL}"
  exit 1
fi
if [[ -n "${INPUT_JSONL}" ]] && [[ ! -f "${INPUT_JSONL}" ]]; then
  echo "ERROR: Input JSONL not found: ${INPUT_JSONL}"
  exit 1
fi

# ---------- Python (needed for ensure_model) ----------
if [[ -n "${CONDA_PYTHON}" ]]; then
  PYTHON_CMD="${CONDA_PYTHON}"
  echo "Using CONDA_PYTHON=${PYTHON_CMD}"
elif [[ -n "${USE_CONDA}" ]] && [ -f "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_CMD="$CONDA_PREFIX/bin/python"
  echo "Using conda python: ${PYTHON_CMD}"
elif [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
  echo "Using project .venv"
else
  PYTHON_CMD="python3"
fi

# ---------- Resolve judge weights: local HF hub snapshot or download ----------
if [[ -z "${USE_API}" ]]; then
  HF_HUB="${HF_HOME}/hub"
  mkdir -p "${HF_HUB}"

  resolve_model_path() {
    local path="$1"
    if [[ -d "${path}" ]] && [[ -f "${path}/config.json" ]]; then
      echo "${path}"
      return 0
    fi
    if [[ -d "${path}/snapshots" ]]; then
      local snap
      snap=$(find "${path}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
      if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then
        echo "${snap}"
        return 0
      fi
    fi
    echo ""
  }

  ensure_model() {
    local repo_id="$1"
    local path="$2"
    local resolved
    resolved=$(resolve_model_path "${path}")
    if [[ -n "${resolved}" ]]; then
      echo "${resolved}"
      return 0
    fi
    echo "Downloading missing judge model: ${repo_id} -> ${HF_HUB}" 1>&2
    local downloaded
    downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
repo_id = sys.argv[1]
cache_dir = sys.argv[2]
token = os.getenv('HF_TOKEN')
try:
    path = snapshot_download(repo_id=repo_id, cache_dir=cache_dir, resume_download=True, token=token)
    print(path)
except Exception as e:
    print(f'DOWNLOAD_FAILED: {e}', file=sys.stderr)
    sys.exit(1)
" "${repo_id}" "${HF_HUB}")
    if [[ $? -ne 0 ]] || [[ -z "${downloaded}" ]]; then
      return 1
    fi
    echo "${downloaded}"
    return 0
  }

  if [[ "${JUDGE_MODEL}" == /* ]] && [[ -f "${JUDGE_MODEL}/config.json" ]]; then
    JUDGE_MODEL_RESOLVED="${JUDGE_MODEL}"
    echo "Judge: using explicit local path ${JUDGE_MODEL_RESOLVED}"
  elif [[ "${JUDGE_MODEL}" == /* ]]; then
    JUDGE_MODEL_RESOLVED=$(resolve_model_path "${JUDGE_MODEL}") || true
    if [[ -z "${JUDGE_MODEL_RESOLVED}" ]]; then
      echo "ERROR: JUDGE_MODEL is a path but has no config.json: ${JUDGE_MODEL}"
      exit 1
    fi
    echo "Judge: resolved directory to snapshot ${JUDGE_MODEL_RESOLVED}"
  else
    # Hugging Face repo id -> hub dir models--Org--Name
    HUB_HINT="${JUDGE_MODEL_PATH:-${HF_HUB}/models--${JUDGE_MODEL//\/--}}"
    JUDGE_MODEL_RESOLVED=$(ensure_model "${JUDGE_MODEL}" "${HUB_HINT}") || {
      echo "ERROR: Could not resolve or download judge model ${JUDGE_MODEL}"
      exit 1
    }
    echo "Judge: using ${JUDGE_MODEL_RESOLVED}"
  fi

  export USE_LOCAL_MODEL=true
  export LOCAL_MODEL="${JUDGE_MODEL_RESOLVED}"
  export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
  export VLLM_ONLY="${VLLM_ONLY:-1}"
  # Tensor parallel: one vLLM replica shards the judge across all allocated GPUs (see --gres).
  export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
  export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
  LOCAL_MODEL_ARGS="--use-local-model --judge-model ${JUDGE_MODEL_RESOLVED}"
  # HF judge path (optional): skips vLLM entirely — not a fallback when vLLM crashes.
  if [[ -n "${USE_HF}" ]]; then
    LOCAL_MODEL_ARGS+=" --use-hf"
    export VLLM_ONLY=0
  fi
fi

# ---------- Diagnostics ----------
echo "=== Evaluation job (${SLURM_JOB_ID:-local}) ==="
echo "Host: $(hostname)"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-N/A}}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-N/A}"
echo "SLURM_SUBMIT_HOST=${SLURM_SUBMIT_HOST:-N/A}"
echo "Judge model repo/id: ${JUDGE_MODEL}"
[[ -n "${JUDGE_MODEL_RESOLVED}" ]] && echo "Judge model path: ${JUDGE_MODEL_RESOLVED}"
echo "Use API: ${USE_API:-no (local vLLM)}"
echo "Evals: ${EVALS}"
echo "Temperature: ${TEMPERATURE}  Max tokens: ${MAX_TOKENS}"
if [[ -n "${INPUT_JSONL}" ]]; then
  echo "Input JSONL: ${INPUT_JSONL}"
else
  echo "O2 JSONL: ${O2_JSONL}"
  echo "O1 JSONL: ${O1_JSONL:-none}"
fi
echo "Output dir: ${EVAL_OUT_DIR}"
[[ -z "${USE_API}" ]] && echo "vLLM: TP=${VLLM_TENSOR_PARALLEL_SIZE} gpu_mem=${VLLM_GPU_MEMORY_UTILIZATION} max_len=${VLLM_MAX_MODEL_LEN} VLLM_ONLY=${VLLM_ONLY:-1}"
nvidia-smi 2>/dev/null || true
echo "========================="

# ---------- Build & run ----------
mkdir -p "${EVAL_OUT_DIR}"

CMD="${PYTHON_CMD} src/evaluation_ranking/evaluate_safety_helpfulness.py"
if [[ -n "${INPUT_JSONL}" ]]; then
  CMD+=" --input-jsonl \"${INPUT_JSONL}\""
else
  CMD+=" --o2-jsonl \"${O2_JSONL}\""
  [[ -n "${O1_JSONL}" ]] && CMD+=" --o1-jsonl \"${O1_JSONL}\""
fi
CMD+=" --output-dir \"${EVAL_OUT_DIR}\""
CMD+=" --evals ${EVALS}"
CMD+=" --temperature ${TEMPERATURE}"
CMD+=" --max-tokens ${MAX_TOKENS}"
[[ -n "${MAX_RECORDS}" ]] && CMD+=" --max-records ${MAX_RECORDS}"
CMD+=" ${LOCAL_MODEL_ARGS}"
CMD+=" ${WANDB_ARG}"
CMD+=" ${EXIT_NULL_ARG}"

echo "Running: ${CMD}"
eval ${CMD}

exit $?
