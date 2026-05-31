#!/usr/bin/bash
# Refinement-only experiment (DoNotAnswer) with refinement prompt + guidelines.
# Reuses the same O1 baseline files and model loop as the base refinement job.

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --time=96:00:00
#SBATCH --job-name=dna_refine_from_baseline_guidelines_all_models
#SBATCH --output=$PROJECT_ROOT/logs/%x_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/%x_%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PROJECT_ROOT}"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }

LINGUASAFE_EXPERIMENTS_ROOT="${LINGUASAFE_EXPERIMENTS_ROOT:-}"
export SYSTEM_PROMPT="${ROOT_DIR}/src/prompts/p2_response_system_prompt_refinement_with_guidelines.txt"
if [[ -n "${LINGUASAFE_EXPERIMENTS_ROOT}" ]]; then
  export EXPERIMENT_NAME="${EXPERIMENT_NAME:-refinement_from_baseline_guidelines_multimodel}"
  export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${LINGUASAFE_EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}}"
else
  export EXPERIMENT_NAME="dna_refinement_from_baseline_guidelines_multimodel"
  export EXPERIMENT_DIR="${ROOT_DIR}/output/DNA_Per_Model_Experiments/refinement_from_baseline_guidelines_multimodel"
fi

bash "${ROOT_DIR}/src/slurm/p2/job_dna_refinement_from_baseline_all_models.slurm"
