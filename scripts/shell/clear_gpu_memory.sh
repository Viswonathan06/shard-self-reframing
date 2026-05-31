#!/bin/bash
# Clear GPU memory and processes that might be using GPU memory

echo "=== Clearing GPU Memory ==="

# Kill any Python processes that might be holding GPU memory
echo "Checking for Python processes..."
pkill -f python || echo "No Python processes found"
pkill -f vllm || echo "No vLLM processes found"

# Clear CUDA cache
echo "Clearing CUDA cache..."
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'GPU memory cleared on {torch.cuda.device_count()} devices')
else:
    print('CUDA not available')
" 2>/dev/null || echo "Could not clear CUDA cache"

# Wait a moment for cleanup
sleep 2

# Show current GPU status
echo ""
echo "=== Current GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv

echo ""
echo "=== Memory Summary ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits | while read line; do
    IFS=',' read -r gpu_id used free <<< "$line"
    used_gb=$(echo "scale=1; $used / 1024" | bc -l 2>/dev/null || echo "N/A")
    free_gb=$(echo "scale=1; $free / 1024" | bc -l 2>/dev/null || echo "N/A")
    echo "GPU $gpu_id: ${used_gb}GB used, ${free_gb}GB free"
done

echo ""
echo "GPU memory cleared. You can now run your vLLM experiments."