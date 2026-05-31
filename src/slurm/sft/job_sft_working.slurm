#!/bin/bash
#SBATCH --job-name=sft_working
#SBATCH --output=$PROJECT_ROOT/logs/sft_working_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/sft_working_%j.err
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00

echo "=========================================="
echo "SFT Training - Working Version"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llm_safety

# Check environment
echo "🔍 Environment check:"
python --version
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Set offline mode for cached models
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "🚀 Starting SFT training..."

# Run training
python scripts/sft_working.py

echo "=========================================="
echo "Status: $?"
echo "End: $(date)"
echo "=========================================="

# Check results
if [ -d "models/llama31_sft_working" ]; then
    echo "✅ Training completed!"
    ls -la models/llama31_sft_working/
else
    echo "❌ Training failed"
fi