#!/bin/bash
# Submit refinement evaluation jobs for DNA New Experiments
# 
# This script submits jobs to evaluate:
# 1. Baseline vs Refinement (all 6 models)
#
# Usage from repo root:
#   cd $PROJECT_ROOT
#   bash scripts/shell/submit_refinement_evaluations.sh

set -euo pipefail

# Repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== DNA Refinement Evaluation Submission ==="
echo "Repository: $REPO_ROOT"
echo "Timestamp: $(date)"
echo ""

# Models to check
MODELS=("llama8b" "llama33_70b" "mistral_7b" "mistral_24b" "qwen35_9b" "qwen35_27b")

# Check data availability for refinement
echo "=== Checking Refinement Data Availability ==="
AVAILABLE_MODELS=()
MISSING_MODELS=()

for model in "${MODELS[@]}"; do
    refinement_file="output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/${model}/baseline_outputs.jsonl"
    
    if [[ -f "$refinement_file" ]]; then
        record_count=$(wc -l < "$refinement_file" 2>/dev/null || echo "0")
        echo "✅ $model: $record_count records in $refinement_file"
        AVAILABLE_MODELS+=("$model")
    else
        echo "❌ $model: Missing $refinement_file"
        MISSING_MODELS+=("$model")
    fi
done

echo ""
echo "Summary:"
echo "  Available models: ${#AVAILABLE_MODELS[@]}/6"
echo "  Missing models: ${#MISSING_MODELS[@]}/6"

if [[ ${#AVAILABLE_MODELS[@]} -eq 0 ]]; then
    echo "❌ ERROR: No refinement data available. Cannot submit evaluation jobs."
    exit 1
fi

if [[ ${#MISSING_MODELS[@]} -gt 0 ]]; then
    echo "⚠️  WARNING: Some models missing refinement data:"
    printf '    %s\n' "${MISSING_MODELS[@]}"
    echo ""
fi

# Submit jobs
echo "=== Submitting Refinement Evaluation Jobs ==="

# Job 1: Baseline vs Refinement (all available models)
echo "Submitting Job: Baseline vs Refinement (all ${#AVAILABLE_MODELS[@]} models)"
job1_id=$(sbatch --parsable src/slurm/evaluation/job_eval_all_models_baseline_vs_refinement_gemma4.slurm)
echo "  Job ID: $job1_id"
echo "  Monitor: tail -f logs/eval_all_models_baseline_vs_refinement_gemma4_${job1_id}.out"

echo ""
echo "=== Jobs Submitted Successfully! ==="
echo ""
echo "📊 Evaluation Overview:"
echo "  • Job 1 ($job1_id): Baseline vs Refinement"
echo "    - Models: ${AVAILABLE_MODELS[*]}"
echo "    - Records per model: ~939"
echo "    - Total comparisons: ~$((${#AVAILABLE_MODELS[@]} * 939))"
echo ""
echo "🔍 Monitor Progress:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs/eval_all_models_baseline_vs_refinement_gemma4_${job1_id}.out"
echo ""
echo "📈 Expected Timeline:"
echo "  • Each model: ~6-8 hours"
echo "  • Total job time: ~$((${#AVAILABLE_MODELS[@]} * 7)) hours"
echo ""
echo "📂 Results will be saved to:"
echo "  output/DNA_New_Experiments/evaluation/"
echo "    ├── baseline_vs_refinement_llama8b_gemma4_judge/"
echo "    ├── baseline_vs_refinement_llama33_70b_gemma4_judge/"
echo "    ├── baseline_vs_refinement_mistral_7b_gemma4_judge/"
echo "    ├── baseline_vs_refinement_mistral_24b_gemma4_judge/"
echo "    ├── baseline_vs_refinement_qwen35_9b_gemma4_judge/"
echo "    └── baseline_vs_refinement_qwen35_27b_gemma4_judge/"
echo ""
echo "✅ All refinement evaluation jobs submitted!"