#!/bin/bash
# Launch SFT training for Llama 3.1 8B with scratchpad data

echo "🚀 Launching Llama 3.1 8B SFT Training"
echo "======================================"

# Check if training data exists
if [ ! -f "data/sft_scratchpad/train_sft_sharegpt.jsonl" ]; then
    echo "❌ Training data not found: data/sft_scratchpad/train_sft_sharegpt.jsonl"
    echo "Please run the data preparation script first:"
    echo "python3 scripts/prepare_sft_data.py --input output/qwen480b_scratchpad_llama8b_full.jsonl --output-dir data/sft_scratchpad --format sharegpt"
    exit 1
fi

# Check conda environment
echo "🔬 Checking conda environment..."
if command -v conda &> /dev/null; then
    echo "✅ Conda available"
    if conda info --envs | grep -q "llm_safety"; then
        echo "✅ llm_safety environment found"
    else
        echo "⚠️  llm_safety environment not found, but continuing..."
    fi
else
    echo "⚠️  Conda not found, hoping environment is already set up..."
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit SLURM job
echo "📋 Submitting SLURM job..."
job_id=$(sbatch src/slurm/sft/job_sft_llama31_8b_scratchpad.slurm | grep -oE '[0-9]+')

if [ ! -z "$job_id" ]; then
    echo "✅ Job submitted successfully!"
    echo "🆔 Job ID: $job_id"
    echo "📄 Output: logs/sft_llama31_8b_scratchpad_${job_id}.out"
    echo "📄 Error: logs/sft_llama31_8b_scratchpad_${job_id}.err"
    echo ""
    echo "🔍 Monitor with: squeue -j $job_id"
    echo "📊 Follow logs: tail -f logs/sft_llama31_8b_scratchpad_${job_id}.out"
else
    echo "❌ Failed to submit job"
    exit 1
fi