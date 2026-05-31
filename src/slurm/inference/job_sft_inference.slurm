#!/bin/bash
#SBATCH --job-name=sft_inference
#SBATCH --output=$PROJECT_ROOT/logs/sft_inference_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/sft_inference_%j.err
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=16:00:00

echo "=========================================="
echo "SFT Model Inference"
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate llm_safety

# Check setup
echo "🔍 Environment check:"
python --version
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Check model exists
echo "🔍 Checking SFT model:"
ls -la models/llama31_full_sft/

echo "🚀 SFT inference: P1-only, full llama8b baseline_outputs.jsonl (~939), streamed JSONL..."

# --max-records=0 → entire input file; each line flushed to disk as it completes
python scripts/inference_sft_model.py \
    --base-model="$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" \
    --sft-adapter="models/llama31_full_sft" \
    --input-data="output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/llama8b/baseline_outputs.jsonl" \
    --output="output/sft_inference_results.jsonl" \
    --max-records=0 \
    --max-new-tokens=2048 \
    --temperature=0.3

echo "=========================================="
echo "Status: $?"
echo "End: $(date)"
echo "=========================================="

# Show results
if [ -f "output/sft_inference_results.jsonl" ]; then
    echo "✅ Inference completed!"
    echo "📊 Results:"
    wc -l output/sft_inference_results.jsonl
    echo ""
    echo "📄 Sample result:"
    head -1 output/sft_inference_results.jsonl | python -m json.tool | head -20
else
    echo "❌ Inference failed"
fi