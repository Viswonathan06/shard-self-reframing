#!/bin/bash
#SBATCH --job-name=train_classifier_focal
#SBATCH --output=logs/train_classifier_focal_%j.out
#SBATCH --error=logs/train_classifier_focal_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=small

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Activate environment if needed (adjust path as necessary)
# source /path/to/your/venv/bin/activate

# Change to the prompt_evaluation directory
cd /nas/longleaf$HOME

# Run training with focal loss and xlm-roberta-base
python train_multihead_classifier.py \
    --csv-path /nas/longleaf$HOME \
    --model-name xlm-roberta-base \
    --output-dir models/multihead_classifier_level4_multilingual_focal \
    --batch-size 16 \
    --epochs 5 \
    --learning-rate 2e-5 \
    --test-size 0.2 \
    --language en,bn,zh,ar \
    --max-length 256 \
    --use-intent-class-weights \
    --intent-loss-weight 2.0 \
    --label-smoothing 0.05 \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --seed 42

echo ""
echo "End Time: $(date)"
echo "Job completed!"

