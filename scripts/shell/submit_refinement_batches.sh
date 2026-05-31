#!/bin/bash
# Submit refinement evaluation jobs in 2 parallel batches
# 
# Batch 1: llama8b, llama33_70b, mistral_7b
# Batch 2: mistral_24b, qwen35_9b, qwen35_27b
#
# Usage from repo root:
#   cd $PROJECT_ROOT
#   bash scripts/shell/submit_refinement_batches.sh

set -euo pipefail

# Repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== DNA Refinement Evaluation - 2 Parallel Batches ==="
echo "Repository: $REPO_ROOT"
echo "Timestamp: $(date)"
echo ""

echo "=== Submitting Parallel Batches ==="

# Submit Batch 1 (first 3 models)
echo "Submitting Batch 1: llama8b, llama33_70b, mistral_7b"
batch1_id=$(sbatch --parsable src/slurm/evaluation/job_eval_refinement_batch1_gemma4.slurm)
echo "  Job ID: $batch1_id"

# Submit Batch 2 (last 3 models)
echo "Submitting Batch 2: mistral_24b, qwen35_9b, qwen35_27b"
batch2_id=$(sbatch --parsable src/slurm/evaluation/job_eval_refinement_batch2_gemma4.slurm)
echo "  Job ID: $batch2_id"

echo ""
echo "=== Both Batches Submitted Successfully! ==="
echo ""
echo "📊 Evaluation Overview:"
echo "  • Batch 1 ($batch1_id): llama8b, llama33_70b, mistral_7b"
echo "  • Batch 2 ($batch2_id): mistral_24b, qwen35_9b, qwen35_27b"
echo "  • Records per model: ~939"
echo "  • Total comparisons: ~5,634 (2 batches × ~2,817 each)"
echo ""
echo "🔍 Monitor Progress:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/eval_refinement_batch1_gemma4_${batch1_id}.out"
echo "  tail -f logs/eval_refinement_batch2_gemma4_${batch2_id}.out"
echo ""
echo "📈 Expected Timeline:"
echo "  • Each batch: ~18-24 hours (3 models × 6-8 hours)"
echo "  • Both batches run in parallel!"
echo "  • Total wall time: ~18-24 hours (instead of 36-48 hours sequential)"
echo ""
echo "📂 Results will be saved to:"
echo "  output/DNA_New_Experiments/evaluation/"
echo "    ├── baseline_vs_refinement_llama8b_gemma4_judge/      (Batch 1)"
echo "    ├── baseline_vs_refinement_llama33_70b_gemma4_judge/  (Batch 1)" 
echo "    ├── baseline_vs_refinement_mistral_7b_gemma4_judge/   (Batch 1)"
echo "    ├── baseline_vs_refinement_mistral_24b_gemma4_judge/  (Batch 2)"
echo "    ├── baseline_vs_refinement_qwen35_9b_gemma4_judge/    (Batch 2)"
echo "    └── baseline_vs_refinement_qwen35_27b_gemma4_judge/   (Batch 2)"
echo ""
echo "✅ Parallel refinement evaluation batches submitted!"
echo "⚡ This approach is ~2x faster than sequential processing!"