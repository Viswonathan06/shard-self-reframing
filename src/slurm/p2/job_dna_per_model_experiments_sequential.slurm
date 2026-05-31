#!/usr/bin/bash
# DNA per-model experiments: all 5 stages using per-model p2s.
# Mirrors the LinguaSafe sequential pipeline but for the DoNotAnswer dataset.
#
# Stage 0 (prerequisite): per-model p2s must already exist in DNA_P2S_DIR.
#   Run job_dna_p1_to_p2_per_model_all_models.slurm first (job 13033).
#
# Stage 1: Helpful-assistant baseline (O1) — needed by stages 4 and 5.
# Stage 2: Benign-intent multimodel
# Stage 3: Benign-intent + guidelines multimodel
# Stage 4: Refinement-from-baseline (O1 + per-model p2 → O2)
# Stage 5: Refinement-from-baseline + guidelines
#
# All outputs go to output/DNA_Per_Model_Experiments/ — a fresh root with no
# old checkpoints from the shared-p2 runs in DNA_New_Experiments/.
#
# Submit after job 13033 finishes:
#   sbatch src/slurm/p2/job_dna_per_model_experiments_sequential.slurm

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=336:00:00
#SBATCH --job-name=dna_per_model_seq
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true

export ROOT_DIR
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

if [ -f "${ROOT_DIR}/.venv/bin/python" ]; then
  source "${ROOT_DIR}/.venv/bin/activate"
fi

# Per-model p2s produced by job_dna_p1_to_p2_per_model_all_models.slurm
export LINGUASAFE_P2S_DIR="${ROOT_DIR}/output/DNA_New_Experiments/p2s"

# Fresh output root — no overlap with old DNA_New_Experiments shared-p2 runs
export LINGUASAFE_EXPERIMENTS_ROOT="${ROOT_DIR}/output/DNA_Per_Model_Experiments"

mkdir -p "${LINGUASAFE_EXPERIMENTS_ROOT}"
mkdir -p "${ROOT_DIR}/logs"

echo "DNA per-model sequential experiments | $(date) | HOST=$(hostname)"
echo "P2S dir:      ${LINGUASAFE_P2S_DIR}"
echo "Output root:  ${LINGUASAFE_EXPERIMENTS_ROOT}"

# Verify all 6 per-model p2 files exist and are complete before starting
MODELS=(llama8b qwen35_9b qwen35_27b llama33_70b mistral_7b mistral_24b)
for m in "${MODELS[@]}"; do
  f="${LINGUASAFE_P2S_DIR}/${m}_p2.jsonl"
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: per-model p2 missing: ${f}"; exit 1
  fi
  n=$(wc -l < "${f}" 2>/dev/null || echo 0)
  if [[ "${n}" -lt 939 ]]; then
    echo "ERROR: ${f} has only ${n}/939 rows — stage 0 may not be complete"; exit 1
  fi
  echo "  OK: ${m}_p2.jsonl (${n} rows)"
done

STAGE_RETRIES=2
STAGE_RETRY_SLEEP_SECONDS=120

run_stage() {
  local label="$1"
  local script="$2"
  local attempt=1
  echo ""
  echo "----- ${label} | $(date) -----"
  while [[ ${attempt} -le ${STAGE_RETRIES} ]]; do
    echo "[${label}] attempt ${attempt}/${STAGE_RETRIES}"
    local tmp_script="/tmp/$(basename "${script}").$$.${attempt}"
    cp "${script}" "${tmp_script}" || { echo "[${label}] failed to copy script"; return 1; }
    chmod +x "${tmp_script}"
    if bash "${tmp_script}"; then
      rm -f "${tmp_script}"
      echo "[${label}] succeeded at $(date)"
      return 0
    fi
    rm -f "${tmp_script}"
    echo "[${label}] attempt ${attempt} failed"
    attempt=$((attempt + 1))
    if [[ ${attempt} -le ${STAGE_RETRIES} ]]; then
      echo "[${label}] retrying in ${STAGE_RETRY_SLEEP_SECONDS}s..."
      sleep ${STAGE_RETRY_SLEEP_SECONDS}
    fi
  done
  echo "[${label}] FAILED after ${STAGE_RETRIES} attempts"
  return 1
}

SCRIPTS="${ROOT_DIR}/src/slurm/p2"

echo ""
echo "----- Stage 1/5: Helpful-assistant baseline (O1 producer) -----"
run_stage "Stage 1/5 helpful_assistant" "${SCRIPTS}/job_dna_helpful_assistant_all_models.slurm"

echo ""
echo "----- Stage 2/5: Benign-intent multimodel -----"
run_stage "Stage 2/5 benign_intent" "${SCRIPTS}/job_dna_benign_intent_all_models.slurm"

echo ""
echo "----- Stage 3/5: Benign-intent + guidelines multimodel -----"
run_stage "Stage 3/5 benign_intent_guidelines" "${SCRIPTS}/job_dna_benign_intent_guidelines_all_models.slurm"

echo ""
echo "----- Stage 4/5: Refinement-from-baseline -----"
run_stage "Stage 4/5 refinement_from_baseline" "${SCRIPTS}/job_dna_refinement_from_baseline_all_models.slurm"

echo ""
echo "----- Stage 5/5: Refinement-from-baseline + guidelines -----"
run_stage "Stage 5/5 refinement_from_baseline_guidelines" "${SCRIPTS}/job_dna_refinement_from_baseline_guidelines_all_models.slurm"

echo ""
echo "All DNA per-model stages done at: $(date)"
echo "Results under: ${LINGUASAFE_EXPERIMENTS_ROOT}"
