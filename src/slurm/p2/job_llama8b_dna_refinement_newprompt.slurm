#!/usr/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --job-name=p2_llama8b_dna_refine_v2
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$PROJECT_ROOT"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export ROOT_DIR
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export USE_LOCAL_MODEL=true
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_TRUST_REMOTE_CODE=1

# Match the known-good refinement job env selection
if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

resolve_model_path() {
  local path="$1"
  if [[ -d "${path}" ]] && [[ -f "${path}/config.json" ]]; then
    echo "${path}"; return 0
  fi
  if [[ -d "${path}/snapshots" ]]; then
    local snap
    snap=$(find "${path}/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
    if [[ -n "${snap}" ]] && [[ -f "${snap}/config.json" ]]; then
      echo "${snap}"; return 0
    fi
  fi
  echo ""
}

SHARED_HUB="$HF_HOME"
USER_HUB="/playpen/${USER}/huggingface/hub"
LLAMA8B_RESOLVED=$(resolve_model_path "${SHARED_HUB}/models--meta-llama--Llama-3.1-8B-Instruct")

if [[ -z "${LLAMA8B_RESOLVED}" ]]; then
  echo "Llama-3.1-8B not in shared hub — downloading to ${USER_HUB}..."
  mkdir -p "${USER_HUB}"
  LLAMA8B_RESOLVED=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    cache_dir=sys.argv[1],
    token=os.getenv('HF_TOKEN')
)
print(path)
" "${USER_HUB}")
  if [[ $? -ne 0 ]] || [[ -z "${LLAMA8B_RESOLVED}" ]]; then
    echo "ERROR: Failed to download Llama-3.1-8B-Instruct"
    exit 1
  fi
fi

BASELINE_JSONL="${ROOT_DIR}/output/baseline_outputs_donotanswer.jsonl"
P1P2_JSONL="${ROOT_DIR}/output/p1_p2_dna_llama70b_gpt5cat.jsonl"
JOINED_JSONL="${ROOT_DIR}/output/p1_p2_dna_refinement_joined.jsonl"
OUTPUT_DIR="${ROOT_DIR}/output/p2_controlled/donotanswer/p2_llama8b_dna_refinement_newprompt"
SYSTEM_PROMPT="${ROOT_DIR}/src/prompts/p2_response_system_prompt_refinement.txt"
export OUTPUT_DIR

if [[ ! -f "${BASELINE_JSONL}" ]]; then
  echo "ERROR: baseline file not found: ${BASELINE_JSONL}"; exit 1
fi
if [[ ! -f "${P1P2_JSONL}" ]]; then
  echo "ERROR: p1_p2 file not found: ${P1P2_JSONL}"; exit 1
fi

echo "=== Joining baseline outputs with p1_p2 records ==="
$PYTHON_CMD - <<'PYEOF'
import json, math, os

root = os.environ["ROOT_DIR"]
baseline_file = os.path.join(root, "output/baseline_outputs_donotanswer.jsonl")
p1p2_file     = os.path.join(root, "output/p1_p2_dna_llama70b_gpt5cat.jsonl")
out_file      = os.path.join(root, "output/p1_p2_dna_refinement_joined.jsonl")

def norm_text(s):
    return " ".join((s or "").split()).strip()

baseline_by_p1 = {}
baseline_by_id = {}
with open(baseline_file) as f:
    for line in f:
        r = json.loads(line)
        pid = r.get("p1_id")
        p1  = r.get("p1", "")
        out = r.get("output", "")
        p1k = norm_text(p1)
        if p1k:
            baseline_by_p1[p1k] = out
        try:
            key = str(int(float(pid))) if pid is not None else None
        except (ValueError, TypeError):
            key = None
        if key:
            baseline_by_id[key] = {"p1": p1, "output": out}

written = matched_text = matched_id = skipped = 0
with open(p1p2_file) as fin, open(out_file, "w") as fout:
    for line in fin:
        r = json.loads(line)
        pid = r.get("p1_id")
        p1  = r.get("p1", "")
        p1k = norm_text(p1)
        try:
            key = str(int(float(pid))) if pid is not None and not math.isnan(float(pid)) else None
        except (ValueError, TypeError):
            key = None

        if p1k and p1k in baseline_by_p1:
            r["baseline_output"] = baseline_by_p1[p1k]
            fout.write(json.dumps(r) + "\n")
            written += 1; matched_text += 1
            continue

        if key and key in baseline_by_id:
            b = baseline_by_id[key]
            if norm_text(b.get("p1", "")) == p1k and p1k:
                r["baseline_output"] = b.get("output", "")
                fout.write(json.dumps(r) + "\n")
                written += 1; matched_id += 1
                continue

        skipped += 1

print(f"Joined {written} records -> {out_file}")
print(f"  matched by text: {matched_text}, by id: {matched_id}, skipped: {skipped}")
PYEOF

if [[ $? -ne 0 ]]; then
  echo "ERROR: Join step failed"; exit 1
fi

echo ""
echo "=== Phase 1: Llama-3.1-8B refinement (new prompt) ==="
echo "Model:         ${LLAMA8B_RESOLVED}"
echo "Input:         ${JOINED_JSONL}"
echo "Output:        ${OUTPUT_DIR}"
echo "System prompt: ${SYSTEM_PROMPT}"
nvidia-smi 2>/dev/null || true
echo ""

mkdir -p "${OUTPUT_DIR}"

$PYTHON_CMD src/baselines/generate_p2_controlled_outputs.py \
  --input-jsonl        "${JOINED_JSONL}" \
  --output-dir         "${OUTPUT_DIR}" \
  --system-prompt-file "${SYSTEM_PROMPT}" \
  --use-local-model \
  --local-model        "${LLAMA8B_RESOLVED}" \
  --temperature 0.3 \
  --max-tokens 2000 \
  --refinement-mode \
  --exit-on-null

INFER_EXIT=$?
echo "Inference done at: $(date) | Exit: ${INFER_EXIT}"
if [[ ${INFER_EXIT} -ne 0 ]]; then
  echo "ERROR: Inference failed — skipping judge phase."
  exit ${INFER_EXIT}
fi

# ============================================================
# PHASE 2: Judge evaluation (Qwen3.5-27B)
# Full eval: relevance + helpfulness + harmfulness.
# ============================================================
echo ""
echo "=== Phase 2: Judge evaluation (Qwen3.5-27B, all evals) ==="

# ---------- Reset env from inference to match judge requirements ----------
unset LOCAL_MODEL
export VLLM_ONLY=1
export VLLM_GPU_MEMORY_UTILIZATION=0.80   # lower than inference (27B is larger)
export VLLM_MAX_MODEL_LEN=8192

# HF cache — judge script sets this explicitly; set it here too
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d $HF_HOME ]] && [[ -w $HF_HOME ]]; then
    export HF_HOME=/playpen/huggingface
  else
    PLAYPEN_USER_HF="/playpen/${USER}/huggingface"
    if mkdir -p "${PLAYPEN_USER_HF}/hub" 2>/dev/null && [[ -w "${PLAYPEN_USER_HF}/hub" ]]; then
      export HF_HOME="${PLAYPEN_USER_HF}"
    else
      export HF_HOME="${HOME}/.cache/huggingface"
    fi
  fi
fi
HF_HUB="${HF_HOME}/hub"

# ---------- Resolve Qwen3.5-27B ----------
JUDGE_MODEL="Qwen/Qwen3.5-27B"
JUDGE_RESOLVED=$(resolve_model_path "${HF_HUB}/models--Qwen--Qwen3.5-27B")
if [[ -z "${JUDGE_RESOLVED}" ]]; then
  echo "Downloading Qwen3.5-27B judge to ${HF_HUB}..."
  JUDGE_RESOLVED=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2],
                         resume_download=True, token=os.getenv('HF_TOKEN'))
print(path)
" "${JUDGE_MODEL}" "${HF_HUB}")
  if [[ $? -ne 0 ]] || [[ -z "${JUDGE_RESOLVED}" ]]; then
    echo "ERROR: Could not resolve judge model"; exit 1
  fi
fi
echo "Judge: ${JUDGE_RESOLVED}"
export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${JUDGE_RESOLVED}"

# ---------- Prepare input JSONL (p1, p2, o1, o2) for evaluator ----------
# Refinement output already has baseline_output (O1), output (O2), p1, p2.
EVAL_INPUT="${OUTPUT_DIR}/eval_input_o1_o2.jsonl"
$PYTHON_CMD - <<'PYEOF'
import json, os

out_dir = os.environ["OUTPUT_DIR"]
src = f"{out_dir}/baseline_outputs.jsonl"
dst = f"{out_dir}/eval_input_o1_o2.jsonl"

written = 0
with open(src) as fin, open(dst, "w") as fout:
    for line in fin:
        r = json.loads(line)
        row = {
            "p1_id":    r.get("p1_id"),
            "p1":       r.get("p1", ""),
            "p2":       r.get("p2") or "",
            "o1":       r.get("baseline_output", ""),
            "o2":       r.get("output", ""),
            "category": r.get("category", ""),
            "lang":     r.get("lang", ""),
            "level":    r.get("level"),
        }
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
print(f"Prepared {written} eval records -> {dst}")
PYEOF

if [[ $? -ne 0 ]]; then
  echo "ERROR: eval input preparation failed"; exit 1
fi

# ---------- Run judge ----------
echo "Running judge..."
$PYTHON_CMD src/evaluation_ranking/evaluate_safety_helpfulness.py \
  --input-jsonl  "${EVAL_INPUT}" \
  --output-dir   "${OUTPUT_DIR}" \
  --evals        all \
  --harmfulness-mode \
  --temperature  0.1 \
  --max-tokens   1024 \
  --use-local-model \
  --judge-model  "${JUDGE_RESOLVED}" \
  --no-wandb \
  > >(tee "${OUTPUT_DIR}/eval_judge.out") \
  2> >(tee "${OUTPUT_DIR}/eval_judge.err" >&2)

JUDGE_EXIT=$?
echo "Judge done at: $(date) | Exit: ${JUDGE_EXIT}"
exit ${JUDGE_EXIT}
