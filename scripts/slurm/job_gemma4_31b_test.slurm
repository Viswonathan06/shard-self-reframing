#!/usr/bin/bash
# SLURM job for Gemma 4 31B with vLLM optimization
# Usage: sbatch scripts/slurm/job_gemma4_31b_test.slurm
#
# This script sets up Gemma 4 31B with optimal vLLM configuration for fast inference
#
# Environment overrides:
#   GEMMA4_31B_PATH=/path/to/snapshot sbatch ...
#   MAX_RECORDS=10 sbatch ...
#   INPUT_DATA=dataset/linguasafe.csv sbatch ...
#   OUTPUT_DIR=output/gemma4_test sbatch ...

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000                      # Use a6000 partition for 4 GPUs
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4                  # Request 4 GPUs for tensor parallel
#SBATCH --time=04:00:00               # 4 hours should be enough for testing
#SBATCH --job-name=gemma4_31b_vllm_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Project setup
ROOT_DIR="/common/guam/Safeshift"
cd "$ROOT_DIR" || { echo "ERROR: Cannot cd to $ROOT_DIR"; exit 1; }

# Load environment
source scripts/shell/setup_gemma4_vllm_env.sh

echo "=== SLURM Job Info ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "User: ${USER}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "Nodes: ${SLURM_NNODES}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo ""

# GPU diagnostics
echo "=== GPU Diagnostics ==="
nvidia-smi
echo ""

# Check and optimize GPU memory
echo "=== GPU Memory Check ==="
$PYTHON_CMD scripts/check_gpu_memory.py
echo ""

# Clear any existing GPU memory usage
echo "=== Clearing GPU Memory ==="
pkill -f python || echo "No existing Python processes"
$PYTHON_CMD -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('CUDA cache cleared')
" 2>/dev/null || true
echo ""

# Python environment
if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
    source "${ROOT_DIR}/.venv/bin/activate"
    PYTHON_CMD="python"
    echo "✓ Using project .venv"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
    echo "✓ Using system python3"
else
    echo "❌ No Python found"
    exit 1
fi

# Job parameters (can be overridden with environment variables)
INPUT_DATA="${INPUT_DATA:-dataset/linguasafe.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-output/gemma4_31b_test_${SLURM_JOB_ID}}"
MAX_RECORDS="${MAX_RECORDS:-5}"  # Start with small test
TEMPERATURE="${TEMPERATURE:-0.3}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p logs

echo "=== Test Configuration ==="
echo "Model: ${GEMMA4_31B_PATH}"
echo "Input data: ${INPUT_DATA}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max records: ${MAX_RECORDS}"
echo "Temperature: ${TEMPERATURE}"
echo "Max tokens: ${MAX_TOKENS}"
echo "vLLM TP size: ${VLLM_TENSOR_PARALLEL_SIZE}"
echo ""

# Test 1: Basic model loading and inference
echo "=== Test 1: Basic Model Loading ==="
$PYTHON_CMD -c "
import os
import sys
sys.path.insert(0, '${ROOT_DIR}')

print('Testing vLLM with Gemma 4 31B...')
try:
    from vllm import LLM, SamplingParams
    
    # Initialize model
    print(f'Loading model from: ${GEMMA4_31B_PATH}')
    llm = LLM(
        model='${GEMMA4_31B_PATH}',
        tensor_parallel_size=${VLLM_TENSOR_PARALLEL_SIZE},
        gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION},
        max_model_len=${VLLM_MAX_MODEL_LEN},
        trust_remote_code=True,
        dtype='bfloat16',  # Use bfloat16 for better performance
        enforce_eager=False  # Use CUDA graphs for speed
    )
    
    # Test inference
    sampling_params = SamplingParams(
        temperature=${TEMPERATURE},
        top_p=0.9,
        max_tokens=100
    )
    
    test_prompt = 'Hello! Please tell me about artificial intelligence safety.'
    outputs = llm.generate([test_prompt], sampling_params)
    
    print('✅ Model loaded successfully!')
    print('Test output:', outputs[0].outputs[0].text[:200] + '...')
    
except Exception as e:
    print(f'❌ Error loading model: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

# Test 2: Run on sample data (if available)
if [[ -f "${INPUT_DATA}" ]]; then
    echo ""
    echo "=== Test 2: Sample Data Processing ==="
    
    # Create a simple test script to run Gemma 4 on a few prompts
    cat > "${OUTPUT_DIR}/test_gemma4.py" << 'EOF'
import pandas as pd
import json
import sys
from pathlib import Path
import time
from vllm import LLM, SamplingParams

def test_gemma4_on_data(input_file, output_file, max_records=5):
    print(f"Loading data from: {input_file}")
    
    # Load data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_json(input_file, lines=True)
    
    if len(df) == 0:
        print("No data found!")
        return
        
    # Take first few records
    df_test = df.head(max_records)
    print(f"Processing {len(df_test)} records...")
    
    # Initialize model
    model_path = os.environ['GEMMA4_31B_PATH']
    print(f"Loading Gemma 4 31B from: {model_path}")
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=int(os.environ.get('VLLM_TENSOR_PARALLEL_SIZE', 4)),
        gpu_memory_utilization=float(os.environ.get('VLLM_GPU_MEMORY_UTILIZATION', 0.85)),
        max_model_len=int(os.environ.get('VLLM_MAX_MODEL_LEN', 8192)),
        trust_remote_code=True,
        dtype='bfloat16'
    )
    
    sampling_params = SamplingParams(
        temperature=float(os.environ.get('TEMPERATURE', 0.3)),
        top_p=0.9,
        max_tokens=int(os.environ.get('MAX_TOKENS', 1024))
    )
    
    results = []
    
    # Process prompts
    prompts = []
    if 'prompt' in df_test.columns:
        prompts = df_test['prompt'].tolist()
    elif 'p1' in df_test.columns:
        prompts = df_test['p1'].tolist()
    else:
        print("No 'prompt' or 'p1' column found, using first text column")
        text_cols = df_test.select_dtypes(include=['object']).columns
        if len(text_cols) > 0:
            prompts = df_test[text_cols[0]].tolist()
    
    if not prompts:
        print("No prompts found!")
        return
    
    print(f"Running inference on {len(prompts)} prompts...")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    inference_time = time.time() - start_time
    print(f"✅ Inference completed in {inference_time:.2f}s ({len(prompts)/inference_time:.2f} prompts/sec)")
    
    # Save results
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        result = {
            'prompt_id': i,
            'prompt': prompt,
            'response': output.outputs[0].text,
            'finish_reason': output.outputs[0].finish_reason,
            'model': 'gemma-4-31b-it',
            'temperature': sampling_params.temperature,
            'max_tokens': sampling_params.max_tokens
        }
        results.append(result)
    
    # Write results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✅ Results saved to: {output_file}")
    
    # Print first result as sample
    if results:
        print("\n=== Sample Result ===")
        print(f"Prompt: {results[0]['prompt'][:100]}...")
        print(f"Response: {results[0]['response'][:200]}...")

if __name__ == "__main__":
    import os
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    max_records = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    test_gemma4_on_data(input_file, output_file, max_records)
EOF

    $PYTHON_CMD "${OUTPUT_DIR}/test_gemma4.py" "${INPUT_DATA}" "${OUTPUT_DIR}/gemma4_test_results.jsonl" "${MAX_RECORDS}"
else
    echo "⚠ No input data found at ${INPUT_DATA}, skipping data test"
fi

echo ""
echo "=== Job Completed ==="
echo "Check results in: ${OUTPUT_DIR}/"
echo "Logs: logs/gemma4_31b_vllm_test_${SLURM_JOB_ID}.{out,err}"