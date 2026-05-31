#!/bin/bash
# LinguaSafe-only sequential pipeline (no DNA / DoNotAnswer paths).
#
# Stage 0: P1→P2 per inference model → output/Linguasafe_Experiments/p2s/<name>_p2.jsonl
# Stage 1: Helpful-assistant baseline (O1) under Linguasafe_Experiments/helpful_assistant_multimodel/
# Stage 2: Benign-intent multimodel
# Stage 3: Refinement-from-baseline (O1 from stage 1, P1+P2 from stage 0)
# Stage 4: Refinement-from-baseline + guidelines
#
# Requires: LINGUASAFE_EXPERIMENTS_ROOT and LINGUASAFE_P2S_DIR exported for child jobs
# (defaults below). Child scripts: job_dna_* are reused with these env vars only.

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=336:00:00
#SBATCH --job-name=linguasafe_exp_seq_all_models
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }
source ~/.bashrc 2>/dev/null || true
mkdir -p "${ROOT_DIR}/logs"

export LINGUASAFE_EXPERIMENTS_ROOT="${LINGUASAFE_EXPERIMENTS_ROOT:-${ROOT_DIR}/output/Linguasafe_Experiments}"
export LINGUASAFE_P2S_DIR="${LINGUASAFE_P2S_DIR:-${LINGUASAFE_EXPERIMENTS_ROOT}/p2s}"
export P2S_OUT_DIR="${LINGUASAFE_P2S_DIR}"
STAGE_RETRIES="${STAGE_RETRIES:-3}"
STAGE_RETRY_SLEEP_SECONDS="${STAGE_RETRY_SLEEP_SECONDS:-20}"

run_stage_with_retries() {
    local stage_label="$1"
    local stage_script="$2"
    local attempt=1
    local local_script=""

    while (( attempt <= STAGE_RETRIES )); do
        echo "[${stage_label}] attempt ${attempt}/${STAGE_RETRIES} at $(date)"

        # Copy to local disk first; stale NFS handles can break direct reads.
        local_script="/tmp/$(basename "${stage_script}").$$.${attempt}"
        if ! cp "${stage_script}" "${local_script}"; then
            echo "[${stage_label}] failed to stage script locally: ${stage_script}"
        else
            chmod +x "${local_script}" || true
            if bash "${local_script}"; then
                rm -f "${local_script}" || true
                echo "[${stage_label}] succeeded on attempt ${attempt}"
                return 0
            fi
            rm -f "${local_script}" || true
        fi

        if (( attempt == STAGE_RETRIES )); then
            echo "[${stage_label}] failed after ${STAGE_RETRIES} attempts"
            return 1
        fi

        echo "[${stage_label}] attempt ${attempt} failed; retrying in ${STAGE_RETRY_SLEEP_SECONDS}s..."
        sleep "${STAGE_RETRY_SLEEP_SECONDS}"
        attempt=$((attempt + 1))
    done
}

echo "===================================================="
echo "LinguaSafe experiments (sequential) | $(date)"
echo "LINGUASAFE_EXPERIMENTS_ROOT=${LINGUASAFE_EXPERIMENTS_ROOT}"
echo "LINGUASAFE_P2S_DIR=${LINGUASAFE_P2S_DIR}"
echo "===================================================="

echo ""
echo "----- Stage 0/5: LinguaSafe P1→P2 (per inference model) -----"
run_stage_with_retries "Stage 0/5" "${ROOT_DIR}/src/slurm/p2/job_linguasafe_p1_to_p2_per_model_all_models.slurm"

echo ""
echo "----- Stage 1/5: Helpful-assistant baseline (O1 producer) -----"
run_stage_with_retries "Stage 1/5" "${ROOT_DIR}/src/slurm/p2/job_dna_helpful_assistant_all_models.slurm"

echo ""
echo "----- Stage 2/5: Benign-intent multimodel -----"
run_stage_with_retries "Stage 2/5" "${ROOT_DIR}/src/slurm/p2/job_dna_benign_intent_all_models.slurm"

echo ""
echo "----- Stage 3/5: Refinement-from-baseline multimodel -----"
run_stage_with_retries "Stage 3/5" "${ROOT_DIR}/src/slurm/p2/job_dna_refinement_from_baseline_all_models.slurm"

echo ""
echo "----- Stage 4/5: Refinement-from-baseline + guidelines -----"
run_stage_with_retries "Stage 4/5" "${ROOT_DIR}/src/slurm/p2/job_dna_refinement_from_baseline_guidelines_all_models.slurm"

echo ""
echo "All LinguaSafe stages completed at: $(date)"
