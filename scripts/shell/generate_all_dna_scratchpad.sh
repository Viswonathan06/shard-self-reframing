#!/bin/bash
# Generate scratchpad reasoning traces for all DNA refinement models
#
# This script processes all DNA refinement outputs and generates scratchpad
# traces that demonstrate the 4-step philosophical framework for each model.
#
# Usage from repo root:
#   cd $PROJECT_ROOT
#   bash scripts/shell/generate_all_dna_scratchpad.sh

set -euo pipefail

# Repo root
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== DNA Scratchpad Data Generation for All Models ==="
echo "Repository: $REPO_ROOT"
echo "Timestamp: $(date)"
echo ""

# Models to process
MODELS=("llama8b" "llama33_70b" "mistral_7b" "mistral_24b" "qwen35_9b" "qwen35_27b")

# Treatment types
TREATMENTS=("dna_refinement_from_baseline_multimodel" "dna_refinement_from_baseline_guidelines_multimodel")

echo "=== Checking Data Availability ==="
AVAILABLE_FILES=()
MISSING_FILES=()

for treatment in "${TREATMENTS[@]}"; do
    for model in "${MODELS[@]}"; do
        input_file="output/DNA_New_Experiments/${treatment}/${model}/baseline_outputs.jsonl"
        
        if [[ -f "$input_file" ]]; then
            record_count=$(wc -l < "$input_file" 2>/dev/null || echo "0")
            echo "✅ ${treatment}/${model}: $record_count records"
            AVAILABLE_FILES+=("${treatment}:${model}")
        else
            echo "❌ ${treatment}/${model}: Missing $input_file"
            MISSING_FILES+=("${treatment}:${model}")
        fi
    done
done

echo ""
echo "Summary:"
echo "  Available files: ${#AVAILABLE_FILES[@]}"
echo "  Missing files: ${#MISSING_FILES[@]}"

if [[ ${#AVAILABLE_FILES[@]} -eq 0 ]]; then
    echo "❌ ERROR: No DNA refinement data available. Cannot generate scratchpad data."
    exit 1
fi

echo ""
echo "=== Generating Scratchpad Data ==="

# Create output directory
mkdir -p "output/DNA_New_Experiments/scratchpad_data"

# Process each available file
declare -a SUCCESSFUL_GENERATIONS
declare -a FAILED_GENERATIONS

for file_combo in "${AVAILABLE_FILES[@]}"; do
    IFS=':' read -r treatment model <<< "$file_combo"
    
    echo ""
    echo "========================================"
    echo "Processing: $treatment / $model"
    echo "========================================"
    
    input_file="output/DNA_New_Experiments/${treatment}/${model}/baseline_outputs.jsonl"
    output_file="output/DNA_New_Experiments/scratchpad_data/${treatment}_${model}_scratchpad.jsonl"
    
    echo "Input:  $input_file"
    echo "Output: $output_file"
    
    # Generate scratchpad data
    if python3 scripts/generate_dna_scratchpad_data.py \
        --input "$input_file" \
        --output "$output_file" \
        --temperature 0.3 \
        --max-tokens 4096 \
        --delay 1.0; then
        
        echo "✅ SUCCESS: Generated scratchpad data for ${treatment}/${model}"
        SUCCESSFUL_GENERATIONS+=("${treatment}:${model}")
        
        # Show file size
        if [[ -f "$output_file" ]]; then
            line_count=$(wc -l < "$output_file" 2>/dev/null || echo "0")
            echo "   Generated $line_count scratchpad records"
        fi
    else
        echo "❌ FAILED: Could not generate scratchpad data for ${treatment}/${model}"
        FAILED_GENERATIONS+=("${treatment}:${model}")
    fi
done

# Final summary
echo ""
echo "========================================"
echo "SCRATCHPAD GENERATION SUMMARY"
echo "========================================"
echo "End time: $(date)"
echo "Total files processed: ${#AVAILABLE_FILES[@]}"
echo "Successful: ${#SUCCESSFUL_GENERATIONS[@]}"
echo "Failed: ${#FAILED_GENERATIONS[@]}"

if [[ ${#SUCCESSFUL_GENERATIONS[@]} -gt 0 ]]; then
    echo ""
    echo "✅ Successfully generated scratchpad data for:"
    for success in "${SUCCESSFUL_GENERATIONS[@]}"; do
        IFS=':' read -r treatment model <<< "$success"
        echo "  • $treatment / $model"
    done
    
    echo ""
    echo "📂 Scratchpad files saved to:"
    echo "  output/DNA_New_Experiments/scratchpad_data/"
fi

if [[ ${#FAILED_GENERATIONS[@]} -gt 0 ]]; then
    echo ""
    echo "❌ Failed to generate scratchpad data for:"
    for failure in "${FAILED_GENERATIONS[@]}"; do
        IFS=':' read -r treatment model <<< "$failure"
        echo "  • $treatment / $model"
    done
fi

echo ""
echo "🎯 Research Impact:"
echo "  The generated scratchpad data demonstrates the 4-step philosophical"
echo "  framework for simultaneous safety and helpfulness across all models!"
echo ""
echo "📊 Next Steps:"
echo "  1. Review generated scratchpad quality"
echo "  2. Use for supervised fine-tuning (SFT)"
echo "  3. Train models to show safety reasoning transparently"
echo ""
echo "✅ DNA scratchpad generation complete!"