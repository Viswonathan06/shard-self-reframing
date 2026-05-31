#!/bin/bash
# Submit refinement + guidelines evaluation jobs in 2 parallel batches
# 
# Batch 1: llama8b, llama33_70b, mistral_7b
# Batch 2: mistral_24b, qwen35_9b, qwen35_27b
#
# Usage from repo root:
#   cd $PROJECT_ROOT
#   bash scripts/shell/submit_refinement_guidelines_batches.sh

set -euo pipefail

# Repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== DNA Refinement + Guidelines Evaluation - 2 Parallel Batches ==="
echo "Repository: $REPO_ROOT"
echo "Timestamp: $(date)"
echo ""

# Models to check
MODELS=("llama8b" "llama33_70b" "mistral_7b" "mistral_24b" "qwen35_9b" "qwen35_27b")

# Check data availability for refinement + guidelines
echo "=== Checking Refinement + Guidelines Data Availability ==="
AVAILABLE_MODELS=()
MISSING_MODELS=()

for model in "${MODELS[@]}"; do
    refinement_guidelines_file="output/DNA_New_Experiments/dna_refinement_from_baseline_guidelines_multimodel/${model}/baseline_outputs.jsonl"
    
    if [[ -f "$refinement_guidelines_file" ]]; then
        record_count=$(wc -l < "$refinement_guidelines_file" 2>/dev/null || echo "0")
        echo "✅ $model: $record_count records in $refinement_guidelines_file"
        AVAILABLE_MODELS+=("$model")
    else
        echo "❌ $model: Missing $refinement_guidelines_file"
        MISSING_MODELS+=("$model")
    fi
done

echo ""
echo "Summary:"
echo "  Available models: ${#AVAILABLE_MODELS[@]}/6"
echo "  Missing models: ${#MISSING_MODELS[@]}/6"

if [[ ${#AVAILABLE_MODELS[@]} -eq 0 ]]; then
    echo "❌ ERROR: No refinement + guidelines data available. Cannot submit evaluation jobs."
    exit 1
fi

if [[ ${#MISSING_MODELS[@]} -gt 0 ]]; then
    echo "⚠️  WARNING: Some models missing refinement + guidelines data:"
    printf '    %s\n' "${MISSING_MODELS[@]}"
    echo ""
fi

# Submit jobs
echo "=== Submitting Refinement + Guidelines Evaluation Jobs ==="

# Job 1: Batch 1 (first 3 models)
echo "Submitting Batch 1: llama8b, llama33_70b, mistral_7b"
batch1_id=$(sbatch --parsable src/slurm/evaluation/job_eval_refinement_guidelines_batch1_gemma4.slurm)
echo "  Job ID: $batch1_id"

# Job 2: Batch 2 (last 3 models)
echo "Submitting Batch 2: mistral_24b, qwen35_9b, qwen35_27b"
batch2_id=$(sbatch --parsable src/slurm/evaluation/job_eval_refinement_guidelines_batch2_gemma4.slurm)
echo "  Job ID: $batch2_id"

echo ""
echo "=== Both Batches Submitted Successfully! ==="
echo ""
echo "📊 Evaluation Overview:"
echo "  • Batch 1 ($batch1_id): llama8b, llama33_70b, mistral_7b"
echo "  • Batch 2 ($batch2_id): mistral_24b, qwen35_9b, qwen35_27b"
echo "  • Comparison: Baseline vs Refinement + Guidelines"
echo "  • Records per model: ~939"
echo "  • Total comparisons: ~5,634 (2 batches × ~2,817 each)"
echo ""
echo "🔍 Monitor Progress:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/eval_refinement_guidelines_batch1_gemma4_${batch1_id}.out"
echo "  tail -f logs/eval_refinement_guidelines_batch2_gemma4_${batch2_id}.out"
echo ""
echo "📈 Expected Timeline:"
echo "  • Each batch: ~18-24 hours (3 models × 6-8 hours)"
echo "  • Both batches run in parallel!"
echo "  • Total wall time: ~18-24 hours"
echo ""
echo "📂 Results will be saved to:"
echo "  output/DNA_New_Experiments/evaluation/"
echo "    ├── baseline_vs_refinement_guidelines_llama8b_gemma4_judge/      (Batch 1)"
echo "    ├── baseline_vs_refinement_guidelines_llama33_70b_gemma4_judge/  (Batch 1)" 
echo "    ├── baseline_vs_refinement_guidelines_mistral_7b_gemma4_judge/   (Batch 1)"
echo "    ├── baseline_vs_refinement_guidelines_mistral_24b_gemma4_judge/  (Batch 2)"
echo "    ├── baseline_vs_refinement_guidelines_qwen35_9b_gemma4_judge/    (Batch 2)"
echo "    └── baseline_vs_refinement_guidelines_qwen35_27b_gemma4_judge/   (Batch 2)"
echo ""
echo "🎯 Research Impact:"
echo "  This evaluation will show whether adding explicit guidelines to the"
echo "  already-successful refinement approach provides additional benefits!"
echo ""
echo "✅ Parallel refinement + guidelines evaluation batches submitted!"
echo "⚡ This approach is ~2x faster than sequential processing!"