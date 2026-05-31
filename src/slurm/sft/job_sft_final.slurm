#!/bin/bash
#SBATCH --job-name=sft_final
#SBATCH --output=$PROJECT_ROOT/logs/sft_final_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/sft_final_%j.err
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00

echo "=========================================="
echo "Final SFT Training"
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llm_safety

# Check setup
echo "🔍 Environment:"
python --version
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check model path exists
echo "🔍 Checking model path:"
ls -la /playpen/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/ | head -5

echo "🚀 Starting FULL dataset training (939 samples)..."
python scripts/sft_final.py

echo "=========================================="
echo "Status: $?"
echo "End: $(date)"
echo "=========================================="

if [ -d "models/llama31_final" ]; then
    echo "✅ Success!"
    ls -la models/llama31_final/
else
    echo "❌ Failed"
fi