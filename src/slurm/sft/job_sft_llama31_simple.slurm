#!/bin/bash
#SBATCH --job-name=sft_llama31_simple
#SBATCH --output=$PROJECT_ROOT/logs/sft_llama31_simple_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/sft_llama31_simple_%j.err
#SBATCH --partition=a6000
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00

echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME" 
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=========================================="

# Set up environment
cd $PROJECT_ROOT

# Activate conda environment
echo "🔧 Activating llm_safety conda environment..."
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /common/guam/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate llm_safety

# Verify environment
echo "🧪 Checking environment..."
which python
python --version

# Test CUDA
python -c "
import torch
print(f'🔥 CUDA available: {torch.cuda.is_available()}')
print(f'🔥 GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo "🚀 Starting Simple Llama 3.1 8B SFT Training..."
echo "📊 Training on scratchpad reasoning data"
echo "🧠 Using LoRA for efficient fine-tuning"

# Run simple SFT training (single GPU for now)
python scripts/sft_llama31_8b_simple.py \
    --model_name="/playpen/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --train_data="data/sft_scratchpad/train_sft_sharegpt.jsonl" \
    --val_data="data/sft_scratchpad/val_sft_sharegpt.jsonl" \
    --output_dir="models/llama31_8b_scratchpad_sft_simple" \
    --num_train_epochs=1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=16 \
    --learning_rate=2e-4 \
    --max_seq_length=2048 \
    --use_lora=true \
    --lora_r=8 \
    --lora_alpha=16 \
    --logging_steps=10 \
    --save_steps=100 \
    --eval_steps=50 \
    --warmup_steps=20

echo "=========================================="
echo "Training Status: $?"
echo "End Time: $(date)"
echo "=========================================="

# Show training results
if [ -d "models/llama31_8b_scratchpad_sft_simple" ]; then
    echo "✅ SFT Training completed successfully!"
    echo "📁 Model saved to: models/llama31_8b_scratchpad_sft_simple"
    ls -la models/llama31_8b_scratchpad_sft_simple/
    
    if [ -f "models/llama31_8b_scratchpad_sft_simple/training_info.json" ]; then
        echo "📊 Training info:"
        cat models/llama31_8b_scratchpad_sft_simple/training_info.json
    fi
else
    echo "❌ SFT Training failed - no output directory found"
fi