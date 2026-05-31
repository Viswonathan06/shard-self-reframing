#!/usr/bin/bash
# Refinement experiment (DoNotAnswer): Llama-3.1-8B, Qwen3-7B, Qwen3.5-27B, Llama-3.3-70B,
# Mistral-7B, Mistral-Small-3.2-24B — consecutive per model.
# Each model: baseline (P1→O1) → join → refinement (P1,P2,O1→O2) → judge (gemma-3-27b-it).
# Judge compares each model's own baseline vs its own refined output.
# WandB: runs grouped under WANDB_RUN_GROUP=dna_refinement_multimodel_gemma_judge.

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --job-name=dna_refinement_all_models
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

ROOT_DIR="$PROJECT_ROOT"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export ROOT_DIR
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export USE_LOCAL_MODEL=true
export VLLM_TRUST_REMOTE_CODE=1

if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
  PYTHON_CMD="python"
else
  PYTHON_CMD="python3"
fi

# WandB: all runs from this job share a group so they appear together in the dashboard.
EXPERIMENT_NAME="dna_refinement_multimodel_gemma_judge"
export WANDB_PROJECT="multi-lingual-alignment-llms"
export WANDB_RUN_GROUP="${EXPERIMENT_NAME}"
export WANDB_TAGS="${EXPERIMENT_NAME},refinement,donotanswer,gemma_judge"

if [[ -z "${WANDB_API_KEY}" ]] && [[ -f "${ROOT_DIR}/.wandb_key" ]]; then
  export WANDB_API_KEY=$(cat "${ROOT_DIR}/.wandb_key")
fi
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WARNING: WANDB_API_KEY not set and .wandb_key not found — WandB logging will be disabled."
  export WANDB_MODE=disabled
fi

# HF cache
if [[ -z "${HF_HOME}" ]]; then
  if [[ -d /playpen/huggingface ]] && [[ -w /playpen/huggingface ]]; then
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
mkdir -p "${HF_HUB}"

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

ensure_model() {
  local repo_id="$1"
  local path="$2"
  local resolved
  resolved=$(resolve_model_path "${path}")
  if [[ -n "${resolved}" ]]; then echo "${resolved}"; return 0; fi
  echo "Downloading ${repo_id} to ${HF_HUB}..." 1>&2
  local downloaded
  downloaded=$($PYTHON_CMD -c "
from huggingface_hub import snapshot_download
import os, sys
path = snapshot_download(repo_id=sys.argv[1], cache_dir=sys.argv[2],
                         resume_download=True, token=os.getenv('HF_TOKEN'))
print(path)
" "${repo_id}" "${HF_HUB}")
  if [[ $? -ne 0 ]] || [[ -z "${downloaded}" ]]; then echo "" 1>&2; return 1; fi
  echo "${downloaded}"
}

# ============================================================
# Paths
# ============================================================
P1P2_JSONL="${ROOT_DIR}/output/p1_p2_dna_llama70b_gpt5cat.jsonl"
SYSTEM_PROMPT="${ROOT_DIR}/src/prompts/p2_response_system_prompt_refinement.txt"
EXPERIMENT_DIR="${ROOT_DIR}/output/dna_refinement_multimodel_gemma_judge"

if [[ ! -f "${P1P2_JSONL}" ]]; then
  echo "ERROR: p1_p2 file not found: ${P1P2_JSONL}"; exit 1
fi

# ============================================================
# STEP 1: Resolve judge model once (reused for all models)
# ============================================================
JUDGE_MODEL_ID="google/gemma-3-27b-it"
JUDGE_PATH=$(ensure_model "${JUDGE_MODEL_ID}" "${HF_HUB}/models--google--gemma-3-27b-it")
if [[ -z "${JUDGE_PATH}" ]]; then
  echo "ERROR: Could not resolve judge model ${JUDGE_MODEL_ID}"; exit 1
fi
echo "Judge resolved: ${JUDGE_PATH}"

# ============================================================
# run_refinement_model
# Per-model flow: baseline → join → refinement → judge
# Args: $1=short_name  $2=resolved_model_path
# Output layout:
#   {name}/baseline/   — p1_baseline_outputs.jsonl  (O1)
#   {name}/refined/    — baseline_outputs.jsonl      (O2, also carries baseline_output=O1)
#   {name}/eval/       — eval_input_o1_o2.jsonl, judge outputs
# ============================================================
run_refinement_model() {
  local name="$1"
  local model_path="$2"
  local out_dir="${EXPERIMENT_DIR}/${name}"
  local baseline_dir="${out_dir}/baseline"
  local refined_dir="${out_dir}/refined"
  local eval_dir="${out_dir}/eval"
  local refinement_input="${out_dir}/refinement_input.jsonl"

  echo ""
  echo "=========================================="
  echo "=== Model: ${name}"
  echo "=== Path:  ${model_path}"
  echo "=== Experiment dir: ${out_dir}"
  echo "=========================================="
  nvidia-smi 2>/dev/null || true
  echo ""

  mkdir -p "${baseline_dir}" "${refined_dir}" "${eval_dir}"

  # ----------------------------------------------------------
  # Phase 1: Baseline — send P1 directly to model → O1
  # ----------------------------------------------------------
  export VLLM_TENSOR_PARALLEL_SIZE=4
  export VLLM_GPU_MEMORY_UTILIZATION=0.85
  unset VLLM_ONLY
  unset VLLM_MAX_MODEL_LEN
  export WANDB_JOB_TYPE="baseline"
  export WANDB_RUN_NAME="baseline_${name}"

  echo "--- Phase 1: Baseline inference (P1 → O1) ---"
  $PYTHON_CMD src/baselines/generate_p1_baseline_outputs.py \
    --input-jsonl   "${P1P2_JSONL}" \
    --output-dir    "${baseline_dir}" \
    --use-local-model \
    --local-model   "${model_path}" \
    --temperature 0.3 \
    --max-tokens 2000 \
    --exit-on-null \
    --wandb

  local baseline_exit=$?
  echo "Baseline done: ${name} | Exit: ${baseline_exit} | $(date)"
  if [[ ${baseline_exit} -ne 0 ]]; then
    echo "ERROR: Baseline inference failed for ${name} — skipping refinement and judge."
    return ${baseline_exit}
  fi

  # ----------------------------------------------------------
  # Phase 2: Join — attach each model's O1 to p1_p2 records
  # ----------------------------------------------------------
  echo "--- Phase 2: Join baseline outputs with p1_p2 records ---"
  export BASELINE_DIR="${baseline_dir}"
  export REFINEMENT_INPUT="${refinement_input}"

  $PYTHON_CMD - <<'PYEOF'
import json, math, os

p1p2_file    = os.environ["P1P2_JSONL"]
baseline_file = os.path.join(os.environ["BASELINE_DIR"], "p1_baseline_outputs.jsonl")
out_file     = os.environ["REFINEMENT_INPUT"]

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
    echo "ERROR: Join step failed for ${name} — skipping refinement and judge."
    return 1
  fi

  # ----------------------------------------------------------
  # Phase 3: Refinement — P1 + P2 + O1 → O2
  # ----------------------------------------------------------
  export WANDB_JOB_TYPE="refinement"
  export WANDB_RUN_NAME="refinement_${name}"

  echo "--- Phase 3: Refinement inference (P1, P2, O1 → O2) ---"
  $PYTHON_CMD src/baselines/generate_p2_controlled_outputs.py \
    --input-jsonl        "${refinement_input}" \
    --output-dir         "${refined_dir}" \
    --system-prompt-file "${SYSTEM_PROMPT}" \
    --use-local-model \
    --local-model        "${model_path}" \
    --temperature 0.3 \
    --max-tokens 2000 \
    --refinement-mode \
    --exit-on-null \
    --wandb

  local refine_exit=$?
  echo "Refinement done: ${name} | Exit: ${refine_exit} | $(date)"
  if [[ ${refine_exit} -ne 0 ]]; then
    echo "ERROR: Refinement failed for ${name} — skipping judge."
    return ${refine_exit}
  fi

  # ----------------------------------------------------------
  # Phase 4: Prep eval input — O1 from refined output (baseline_output field), O2 from output field
  # ----------------------------------------------------------
  export REFINED_DIR="${refined_dir}"
  export EVAL_DIR="${eval_dir}"

  $PYTHON_CMD - <<'PYEOF'
import json, os
refined_dir = os.environ["REFINED_DIR"]
eval_dir    = os.environ["EVAL_DIR"]
src = os.path.join(refined_dir, "baseline_outputs.jsonl")
dst = os.path.join(eval_dir, "eval_input_o1_o2.jsonl")
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
    echo "ERROR: Eval input prep failed for ${name}"; return 1
  fi

  # ----------------------------------------------------------
  # Phase 5: Judge (gemma-3-27b-it) — O1 vs O2
  # ----------------------------------------------------------
  export VLLM_ONLY=1
  export VLLM_GPU_MEMORY_UTILIZATION=0.80
  export VLLM_MAX_MODEL_LEN=8192
  export USE_LOCAL_MODEL=true
  export LOCAL_MODEL="${JUDGE_PATH}"
  export WANDB_JOB_TYPE="judge"
  export WANDB_RUN_NAME="refinement_${name}_judge"

  echo "--- Phase 5: Judge evaluation (O1 vs O2) ---"
  $PYTHON_CMD src/evaluation_ranking/evaluate_safety_helpfulness.py \
    --input-jsonl  "${eval_dir}/eval_input_o1_o2.jsonl" \
    --output-dir   "${eval_dir}" \
    --evals        all \
    --harmfulness-mode \
    --temperature  0.1 \
    --max-tokens   1024 \
    --use-local-model \
    --judge-model  "${JUDGE_PATH}" \
    --wandb \
    > >(tee "${eval_dir}/eval_judge.out") \
    2> >(tee "${eval_dir}/eval_judge.err" >&2)

  local judge_exit=$?
  echo "Judge done: ${name} | Exit: ${judge_exit} | $(date)"
  return ${judge_exit}
}

# ============================================================
# STEP 2: Resolve inference models
# ============================================================
LLAMA8B=$(ensure_model "meta-llama/Llama-3.1-8B-Instruct" \
  "${HF_HUB}/models--meta-llama--Llama-3.1-8B-Instruct") \
  || { echo "SKIP llama8b: resolve failed"; LLAMA8B=""; }

QWEN3_7B=$(ensure_model "Qwen/Qwen3-7B" \
  "${HF_HUB}/models--Qwen--Qwen3-7B") \
  || { echo "SKIP qwen3_7b: resolve failed"; QWEN3_7B=""; }

QWEN35_27B=$(ensure_model "Qwen/Qwen3.5-27B" \
  "${HF_HUB}/models--Qwen--Qwen3.5-27B") \
  || { echo "SKIP qwen35_27b: resolve failed"; QWEN35_27B=""; }

LLAMA33_70B=$(ensure_model "meta-llama/Llama-3.3-70B-Instruct" \
  "${LOCAL_MODEL_SNAPSHOT:-$HF_HOME/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b}") \
  || { echo "SKIP llama33_70b: resolve failed"; LLAMA33_70B=""; }

MISTRAL_7B=$(ensure_model "mistralai/Mistral-7B-Instruct-v0.3" \
  "${HF_HUB}/models--mistralai--Mistral-7B-Instruct-v0.3") \
  || { echo "SKIP mistral_7b: resolve failed"; MISTRAL_7B=""; }

MISTRAL_24B=$(ensure_model "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
  "${HF_HUB}/models--mistralai--Mistral-Small-3.2-24B-Instruct-2506") \
  || { echo "SKIP mistral_24b: resolve failed"; MISTRAL_24B=""; }

# ============================================================
# STEP 3: Run consecutively
# ============================================================
[[ -n "${LLAMA8B}" ]]     && run_refinement_model "llama8b"     "${LLAMA8B}"     || true
[[ -n "${QWEN3_7B}" ]]    && run_refinement_model "qwen3_7b"    "${QWEN3_7B}"    || true
[[ -n "${QWEN35_27B}" ]]  && run_refinement_model "qwen35_27b"  "${QWEN35_27B}"  || true
[[ -n "${LLAMA33_70B}" ]] && run_refinement_model "llama33_70b" "${LLAMA33_70B}" || true
[[ -n "${MISTRAL_7B}" ]]  && run_refinement_model "mistral_7b"  "${MISTRAL_7B}"  || true
[[ -n "${MISTRAL_24B}" ]] && run_refinement_model "mistral_24b" "${MISTRAL_24B}" || true

echo ""
echo "All models done at: $(date)"
