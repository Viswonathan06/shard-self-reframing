#!/usr/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --job-name=p2_llama8b_dna_refinement
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$HOME"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export ROOT_DIR
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export USE_LOCAL_MODEL=true
export VLLM_TENSOR_PARALLEL_SIZE=4
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_TRUST_REMOTE_CODE=1

if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
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
OUTPUT_DIR="${ROOT_DIR}/output/p2_controlled/donotanswer/p2_llama8b_dna_refinement"
SYSTEM_PROMPT="${ROOT_DIR}/src/prompts/p2_response_system_prompt_refinement.txt"

echo "=== Joining baseline outputs with p1_p2 records ==="
$PYTHON_CMD - <<'PYEOF'
import json, math, os

root = os.environ["ROOT_DIR"]
baseline_file = os.path.join(root, "output/baseline_outputs_donotanswer.jsonl")
p1p2_file     = os.path.join(root, "output/p1_p2_dna_llama70b_gpt5cat.jsonl")
out_file      = os.path.join(root, "output/p1_p2_dna_refinement_joined.jsonl")

def norm_text(s):
    return " ".join((s or "").split()).strip()

baseline_by_id = {}
baseline_by_p1 = {}
with open(baseline_file) as f:
    for line in f:
        r = json.loads(line)
        pid = r.get("p1_id")
        p1 = r.get("p1", "")
        out = r.get("output", "")
        if pid is not None:
            baseline_by_id[str(int(float(pid)))] = {"p1": p1, "output": out}
        # Primary key: prompt text (robust when p1_id namespaces differ across files)
        p1k = norm_text(p1)
        if p1k:
            baseline_by_p1[p1k] = out

written = 0
matched_text = 0
matched_id = 0
skipped = 0
with open(p1p2_file) as fin, open(out_file, "w") as fout:
    for line in fin:
        r = json.loads(line)
        pid = r.get("p1_id")
        p1 = r.get("p1", "")
        p1k = norm_text(p1)
        try:
            key = str(int(float(pid))) if pid is not None and not math.isnan(float(pid)) else None
        except (ValueError, TypeError):
            key = None

        # 1) Prefer text match (avoids cross-dataset ID collisions)
        if p1k and p1k in baseline_by_p1:
            r["baseline_output"] = baseline_by_p1[p1k]
            fout.write(json.dumps(r) + "\n")
            written += 1
            matched_text += 1
            continue

        # 2) Fallback: ID match only if P1 text also agrees
        if key and key in baseline_by_id:
            b = baseline_by_id[key]
            if norm_text(b.get("p1", "")) == p1k and p1k:
                r["baseline_output"] = b.get("output", "")
                fout.write(json.dumps(r) + "\n")
                written += 1
                matched_id += 1
                continue

        skipped += 1

print(f"Joined {written} records -> {out_file}")
print(f"  matched by text: {matched_text}")
print(f"  matched by id:   {matched_id}")
print(f"  skipped:         {skipped}")
PYEOF

if [[ $? -ne 0 ]]; then
  echo "ERROR: Join step failed"
  exit 1
fi

echo ""
echo "=== Llama-3.1-8B refinement output (DoNotAnswer) ==="
echo "Model:  ${LLAMA8B_RESOLVED}"
echo "Input:  ${JOINED_JSONL}"
echo "Output: ${OUTPUT_DIR}"
echo "System prompt: ${SYSTEM_PROMPT}"
nvidia-smi 2>/dev/null || true
echo ""

mkdir -p "${OUTPUT_DIR}"

$PYTHON_CMD src/baselines/generate_p2_controlled_outputs.py \
  --input-jsonl "${JOINED_JSONL}" \
  --output-dir  "${OUTPUT_DIR}" \
  --system-prompt-file "${SYSTEM_PROMPT}" \
  --use-local-model \
  --local-model "${LLAMA8B_RESOLVED}" \
  --temperature 0.3 \
  --max-tokens 2000 \
  --refinement-mode

EXIT_CODE=$?
echo "Done at: $(date) | Exit: ${EXIT_CODE}"
exit ${EXIT_CODE}
