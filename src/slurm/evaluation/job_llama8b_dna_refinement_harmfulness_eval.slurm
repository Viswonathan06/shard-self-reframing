#!/usr/bin/bash
# Evaluate llama8b baseline (O1) vs dna_refinement (O2) with the NEW harmfulness judge prompt.
#
# [[A]] = more harmful; results stored under "harmfulness" keys and flipped by
# harmlessness_final_block so summary harmlessness_o2_winrate = fraction where O2 is LESS harmful.
#
# Submit from repo root:
#   cd $PROJECT_ROOT
#   sbatch src/slurm/evaluation/job_llama8b_dna_refinement_harmfulness_eval.slurm
#
# Overrides:
#   JUDGE_MODEL=Qwen/Qwen3.5-27B sbatch ...
#   EVAL_OUT_DIR=output/evaluation/my_dir sbatch ...
#   EVALS=helpfulness,harmlessness sbatch ...   (default: helpfulness,harmlessness — no relevance)
#   MAX_RECORDS=50 NO_WANDB=1 sbatch ...
#   VLLM_GPU_MEMORY_UTILIZATION=0.72 VLLM_MAX_MODEL_LEN=4096 VLLM_ENFORCE_EAGER=1 sbatch ...

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --nodelist=gpu5
#SBATCH --gres=gpu:4
#SBATCH --time=72:00:00
#SBATCH --job-name=eval_dna_refine_harm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------- Repo root ----------
_slurm_is_repo() {
  [[ -n "$1" ]] && [[ -d "$1/src/evaluation_ranking" ]]
}

ROOT_DIR="${ROOT_DIR:-}"
if ! _slurm_is_repo "$ROOT_DIR"; then ROOT_DIR=""; fi
if [[ -z "$ROOT_DIR" ]] && _slurm_is_repo "${SLURM_SUBMIT_DIR:-}"; then
  ROOT_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
fi
if [[ -z "$ROOT_DIR" ]]; then
  _SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
  _ROOT_CAND="$(cd "${_SLURM_DIR}/../../.." && pwd)"
  if _slurm_is_repo "$_ROOT_CAND"; then ROOT_DIR="${_ROOT_CAND}"; fi
fi
if [[ -z "$ROOT_DIR" ]] || ! _slurm_is_repo "$ROOT_DIR"; then
  echo "ERROR: Could not resolve repo root. Submit from repo or set ROOT_DIR."
  exit 1
fi
export ROOT_DIR
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
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

# ---------- Python ----------
if [[ -n "${CONDA_PYTHON}" ]]; then
  PYTHON_CMD="${CONDA_PYTHON}"
elif [[ -n "${USE_CONDA}" ]] && [ -f "$CONDA_PREFIX/bin/python" ]; then
  PYTHON_CMD="$CONDA_PREFIX/bin/python"
elif [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

# ---------- Input files ----------
# The refinement JSONL has both baseline_output (O1) and output (O2) in one file.
# We preprocess it into an --input-jsonl with explicit o1/o2 fields.
REFINEMENT_JSONL="${ROOT_DIR}/output/p2_controlled/donotanswer/p2_llama8b_dna_refinement/baseline_outputs.jsonl"
INPUT_JSONL="${ROOT_DIR}/output/p2_controlled/donotanswer/p2_llama8b_dna_refinement/eval_input_o1_o2.jsonl"

if [[ ! -f "${REFINEMENT_JSONL}" ]]; then
  echo "ERROR: Refinement file not found: ${REFINEMENT_JSONL}"
  exit 1
fi

echo "Preprocessing: mapping baseline_output->o1, output->o2 ..."
${PYTHON_CMD} - <<'PYEOF'
import json, os, sys

root = os.environ["ROOT_DIR"]
src  = f"{root}/output/p2_controlled/donotanswer/p2_llama8b_dna_refinement/baseline_outputs.jsonl"
dst  = f"{root}/output/p2_controlled/donotanswer/p2_llama8b_dna_refinement/eval_input_o1_o2.jsonl"

with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        r = json.loads(line)
        out = {
            "p1_id":    r["p1_id"],
            "p1":       r["p1"],
            "p2":       r["p2"],
            "o1":       r["baseline_output"],
            "o2":       r["output"],
            "category": r.get("category", ""),
            "lang":     r.get("lang", ""),
            "level":    r.get("level", None),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"Wrote {dst}", flush=True)
PYEOF

if [[ $? -ne 0 ]]; then
  echo "ERROR: Preprocessing step failed."
  exit 1
fi

# ---------- Judge model ----------
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.5-27B}"

HF_HUB="${HF_HOME}/hub"
mkdir -p "${HF_HUB}"

resolve_model_path() {
  local path="$1"
  if [[ -d "${path}" ]] && [[ -f "${path}/config.json" ]]; then echo "${path}"; return 0; fi
  if [[ -d "${path}/snapshots" ]]; then
    local snap
    snap=$(find "${path}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then echo "${snap}"; return 0; fi
  fi
  echo ""
}

ensure_model() {
  local repo_id="$1" path="$2" resolved
  resolved=$(resolve_model_path "${path}")
  if [[ -n "${resolved}" ]]; then echo "${resolved}"; return 0; fi
  echo "Downloading missing judge: ${repo_id} -> ${HF_HUB}" 1>&2
  local downloaded
  downloaded=$(${PYTHON_CMD} -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2],
                         resume_download=True, token=os.getenv('HF_TOKEN'))
print(path)
" "${repo_id}" "${HF_HUB}")
  [[ $? -ne 0 ]] || [[ -z "${downloaded}" ]] && return 1
  echo "${downloaded}"
}

HUB_HINT="${JUDGE_MODEL_PATH:-${HF_HUB}/models--${JUDGE_MODEL//\/--}}"
JUDGE_MODEL_RESOLVED=$(ensure_model "${JUDGE_MODEL}" "${HUB_HINT}") || {
  echo "ERROR: Could not resolve judge model ${JUDGE_MODEL}"; exit 1
}
echo "Judge: ${JUDGE_MODEL_RESOLVED}"

export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${JUDGE_MODEL_RESOLVED}"
export VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
export VLLM_ONLY="${VLLM_ONLY:-1}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-4}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

# ---------- Output ----------
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${ROOT_DIR}/output/evaluation/llama8b_donotanswer_baseline_vs_dna_refinement_harmfulness_judge_qwen35_27b}"
EVALS="${EVALS:-helpfulness,harmlessness}"
MAX_TOKENS="${MAX_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.1}"
MAX_RECORDS="${MAX_RECORDS:-}"
WANDB_ARG=""
[[ "${NO_WANDB:-}" == "1" ]] && WANDB_ARG="--no-wandb"

mkdir -p "${EVAL_OUT_DIR}"

# ---------- Diagnostics ----------
echo "=== DNA Refinement Harmfulness Eval (${SLURM_JOB_ID:-local}) ==="
echo "Host: $(hostname)"
echo "Judge: ${JUDGE_MODEL_RESOLVED}"
echo "Input JSONL: ${INPUT_JSONL}"
echo "Output dir: ${EVAL_OUT_DIR}"
echo "Evals: ${EVALS}  (harmlessness uses eval_harmfulness.txt via --harmfulness-mode)"
echo "vLLM: TP=${VLLM_TENSOR_PARALLEL_SIZE} gpu_mem=${VLLM_GPU_MEMORY_UTILIZATION} max_len=${VLLM_MAX_MODEL_LEN}"
nvidia-smi 2>/dev/null || true
echo "========================="

# ---------- Run ----------
CMD="${PYTHON_CMD} src/evaluation_ranking/evaluate_safety_helpfulness.py"
CMD+=" --input-jsonl \"${INPUT_JSONL}\""
CMD+=" --output-dir \"${EVAL_OUT_DIR}\""
CMD+=" --evals ${EVALS}"
CMD+=" --harmfulness-mode"
CMD+=" --temperature ${TEMPERATURE}"
CMD+=" --max-tokens ${MAX_TOKENS}"
[[ -n "${MAX_RECORDS}" ]] && CMD+=" --max-records ${MAX_RECORDS}"
CMD+=" --use-local-model --judge-model ${JUDGE_MODEL_RESOLVED}"
CMD+=" ${WANDB_ARG}"

echo "Running: ${CMD}"
eval ${CMD}

exit $?
