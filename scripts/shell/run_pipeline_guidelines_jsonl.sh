#!/usr/bin/env bash
# Run the full guidelines pipeline with src/prompts/guidelines.txt:
#   1. Generate category guidelines (output/guidelines/)
#   2. P1 → P2 conversion (output/p1_to_p2/p1_p2_guidelines/)
#
# Usage:
#   SLURM (submit both; step 2 runs after step 1 succeeds):
#     ./scripts/shell/run_pipeline_guidelines_jsonl.sh
#
#   Or run steps manually:
#     sbatch src/slurm/guidelines/job_generate_category_guidelines_gpu.slurm
#     # After that job finishes:
#     GUIDELINES_DIR=output/guidelines UNIVERSAL_GUIDELINES=src/prompts/guidelines.txt sbatch src/slurm/guidelines/job_p1_to_p2_guidelines_gpu.slurm

set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GUIDELINES_TXT="${ROOT_DIR}/src/prompts/guidelines.txt"
GUIDELINES_DIR="${ROOT_DIR}/output/guidelines"

if [[ ! -f "${GUIDELINES_TXT}" ]]; then
  echo "ERROR: Universal guidelines not found: ${GUIDELINES_TXT}"
  exit 1
fi

echo "Pipeline with guidelines.txt"
echo "  Universal guidelines: ${GUIDELINES_TXT}"
echo "  Category guidelines output: ${GUIDELINES_DIR}"
echo "  P1→P2 output: ${ROOT_DIR}/output/p1_to_p2/p1_p2_guidelines"
echo ""

# Step 1: generate category guidelines (default in job is already guidelines.txt)
echo "Submitting Step 1 (generate category guidelines)..."
JOB1=$(sbatch --parsable --export=ALL,"UNIVERSAL_GUIDELINES=${GUIDELINES_TXT}" \
  src/slurm/guidelines/job_generate_category_guidelines_gpu.slurm)
echo "  Job ID: ${JOB1}"

# Step 2: P1 → P2 (depends on step 1)
echo "Submitting Step 2 (P1→P2) to run after Step 1..."
JOB2=$(sbatch --parsable --dependency=afterok:"${JOB1}" \
  --export=ALL,"GUIDELINES_DIR=${GUIDELINES_DIR}","UNIVERSAL_GUIDELINES=${GUIDELINES_TXT}" \
  src/slurm/guidelines/job_p1_to_p2_guidelines_gpu.slurm)
echo "  Job ID: ${JOB2}"

echo ""
echo "Done. Step 1 → ${JOB1}, Step 2 → ${JOB2} (runs after Step 1 succeeds)."
echo "Monitor: squeue -u \$USER"
