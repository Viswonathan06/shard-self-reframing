#!/usr/bin/bash
# Submit all rational SFT jobs that depend on job 14393 (rational_infer_qwen122b).
#
# Self-model rational SFT (model's own output):
#   Mistral 24B — train + infer + judge (SKIP rational infer + splits, already done)
#   Mistral 7B  — judge only            (already queued as 14431 with dependency set)
#
# 122B teacher rational SFT (qwen35_122b outputs used directly as training targets):
#   Llama 3.1 8B, Phi-4, Qwen 2.5 9B, Qwen 2.5 27B, Mistral 7B, Mistral 24B

set -euo pipefail
DEP="afterok:14393"
ROOT="$PROJECT_ROOT"

echo "Submitting Mistral 24B rational SFT (train+infer+judge) after job 14393..."
SKIP_RATIONAL_INFER=1 SKIP_SPLITS=1 \
  sbatch --dependency="$DEP" \
  "$ROOT/scripts/slurm/job_rational_sft_mistral24b.slurm"

echo ""
echo "Mistral 7B judge-only already queued as 14431 with dependency on 14393."
echo ""

echo "Submitting 122B teacher rational SFT jobs..."
sbatch --dependency="$DEP" "$ROOT/scripts/slurm/job_rational_sft_122b_llama8b.slurm"
sbatch --dependency="$DEP" "$ROOT/scripts/slurm/job_rational_sft_122b_phi4.slurm"
sbatch --dependency="$DEP" "$ROOT/scripts/slurm/job_rational_sft_122b_qwen9b.slurm"
sbatch --dependency="$DEP" "$ROOT/scripts/slurm/job_rational_sft_122b_qwen27b.slurm"
sbatch --dependency="$DEP" "$ROOT/scripts/slurm/job_rational_sft_122b_mistral7b.slurm"
sbatch --dependency="$DEP" "$ROOT/scripts/slurm/job_rational_sft_122b_mistral24b.slurm"

echo "Done."
