#!/usr/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --nodelist=gpu4
#SBATCH --job-name=diagnose_gpu4
#SBATCH --output=$PROJECT_ROOT/logs/diagnose_gpu4_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/diagnose_gpu4_%j.err

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"

echo "=== GPU4 Diagnostic Report ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

echo "--- CUDA / GPU ---"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi failed"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
"$PYTHON" -c "
import torch
print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')
print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'torch.cuda.is_bf16_supported(): {torch.cuda.is_bf16_supported()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>&1
echo ""

echo "--- /playpen mount ---"
mountpoint /playpen 2>/dev/null && echo "mounted" || echo "NOT mounted"
ls /playpen/ 2>/dev/null || echo "cannot list /playpen"
echo ""

echo "--- $HF_HOME contents ---"
ls $HF_HOME/ 2>/dev/null | head -20 || echo "cannot list hub dir"
echo ""

echo "--- Key model directories ---"
for MODEL_DIR in \
    "$HF_HOME/models--microsoft--phi-4" \
    "$HF_HOME/models--google--gemma-4-31B-it" \
    "$HF_HOME/models--Qwen--Qwen3.5-9B" \
    "$HF_HOME/models--Qwen--Qwen3.5-27B" \
    "$HF_HOME/models--mistralai--Mistral-7B-Instruct-v0.3" \
    "$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct"
do
    if [[ -d "$MODEL_DIR" ]]; then
        PERM=$(stat -c "%a %U %G" "$MODEL_DIR" 2>/dev/null)
        SNAP=$(find "$MODEL_DIR/snapshots" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
        CONFIG=""
        [[ -f "${SNAP}/config.json" ]] && CONFIG="config.json OK" || CONFIG="NO config.json"
        echo "  EXISTS   $MODEL_DIR  ($PERM)  snap=$CONFIG"
    else
        echo "  MISSING  $MODEL_DIR"
    fi
done
echo ""

echo "--- Write test ---"
TEST_FILE="$HF_HOME/.write_test_$$"
if touch "$TEST_FILE" 2>/dev/null; then
    echo "$HF_HOME: WRITABLE"
    rm -f "$TEST_FILE"
else
    echo "$HF_HOME: READ-ONLY"
fi
echo ""

echo "=== Done ==="
