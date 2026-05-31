#!/bin/bash
# Submit evaluation jobs for all 6 models in DNA experiments
# Usage: bash scripts/shell/submit_all_model_evaluations.sh

set -euo pipefail

echo "=== Submitting DNA Evaluation Jobs for All Models ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$ROOT_DIR"

# Available models
MODELS=("llama8b" "llama33_70b" "mistral_7b" "mistral_24b" "qwen35_9b" "qwen35_27b")

# Evaluation scripts
EVAL1_SCRIPT="src/slurm/evaluation/job_eval_baseline_vs_benign_intent_gemma4.slurm"
EVAL2_SCRIPT="src/slurm/evaluation/job_eval_baseline_vs_benign_intent_guidelines_gemma4.slurm"

echo "Models to evaluate: ${MODELS[*]}"
echo ""

declare -a JOB_IDS

for model in "${MODELS[@]}"; do
    echo "=== Submitting evaluations for $model ==="
    
    # Check if required files exist
    baseline_file="output/DNA_New_Experiments/dna_baseline_multimodel/${model}/p1_baseline_outputs.jsonl"
    benign_intent_file="output/DNA_New_Experiments/dna_benign_intent_multimodel/${model}/p1_baseline_outputs.jsonl"
    benign_guidelines_file="output/DNA_New_Experiments/dna_benign_intent_guidelines_multimodel/${model}/p1_baseline_outputs.jsonl"
    
    missing_files=()
    [[ ! -f "$baseline_file" ]] && missing_files+=("$baseline_file")
    [[ ! -f "$benign_intent_file" ]] && missing_files+=("$benign_intent_file")
    [[ ! -f "$benign_guidelines_file" ]] && missing_files+=("$benign_guidelines_file")
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        echo "⚠️  Skipping $model - missing files:"
        printf '    %s\n' "${missing_files[@]}"
        echo ""
        continue
    fi
    
    echo "✓ All files found for $model"
    
    # Submit evaluation 1: Baseline vs Benign Intent
    echo "Submitting: Baseline vs Benign Intent ($model)"
    job1_output=$(MODEL="$model" sbatch "$EVAL1_SCRIPT" 2>&1)
    if [[ $? -eq 0 ]]; then
        job1_id=$(echo "$job1_output" | grep -o '[0-9]\+' | tail -1)
        echo "  Job ID: $job1_id"
        JOB_IDS+=("$job1_id")
    else
        echo "  ❌ Failed: $job1_output"
    fi
    
    # Submit evaluation 2: Baseline vs Benign Intent + Guidelines  
    echo "Submitting: Baseline vs Benign Intent + Guidelines ($model)"
    job2_output=$(MODEL="$model" sbatch "$EVAL2_SCRIPT" 2>&1)
    if [[ $? -eq 0 ]]; then
        job2_id=$(echo "$job2_output" | grep -o '[0-9]\+' | tail -1)
        echo "  Job ID: $job2_id"
        JOB_IDS+=("$job2_id")
    else
        echo "  ❌ Failed: $job2_output"
    fi
    
    echo ""
done

echo "=== Submission Summary ==="
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""

echo "=== Monitor Jobs ==="
echo "Check status: squeue -u guam"
echo "Monitor logs: tail -f logs/eval_*_gemma4_*.out"
echo ""

echo "=== Expected Output Directories ==="
for model in "${MODELS[@]}"; do
    echo "Model $model:"
    echo "  baseline_vs_benign_intent_${model}_gemma4_judge/"
    echo "  baseline_vs_benign_intent_guidelines_${model}_gemma4_judge/"
done

echo ""
echo "🚀 All evaluation jobs submitted! Total runtime: ~24-48 hours for all models"