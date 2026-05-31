#!/usr/bin/env python3
"""
Check GPU memory availability for Gemma 4 31B
"""

import torch
import subprocess
import sys

def check_gpu_memory():
    """Check available GPU memory and suggest optimal settings"""
    print("=== GPU Memory Analysis ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    gpu_info = []
    min_free_gb = float('inf')
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total_gb = props.total_memory / (1024**3)
        free_gb = total_gb - allocated - reserved
        
        gpu_info.append({
            'id': i,
            'name': props.name,
            'total': total_gb,
            'allocated': allocated,
            'reserved': reserved,
            'free': free_gb
        })
        
        min_free_gb = min(min_free_gb, free_gb)
        
        status = "✅" if free_gb > 20 else "⚠️" if free_gb > 10 else "❌"
        print(f"GPU {i} ({props.name}): {status}")
        print(f"  Total: {total_gb:.1f}GB")
        print(f"  Free: {free_gb:.1f}GB")
        print(f"  Allocated: {allocated:.1f}GB")
        print(f"  Reserved: {reserved:.1f}GB")
    
    print(f"\nMinimum free memory: {min_free_gb:.1f}GB")
    
    # Recommendations based on available memory
    print("\n=== Recommendations ===")
    
    if min_free_gb >= 30:
        print("✅ Excellent - Can run Gemma 4 31B with high settings")
        utilization = 0.85
        context_len = 8192
        tensor_parallel = 4
    elif min_free_gb >= 25:
        print("✅ Good - Can run Gemma 4 31B with medium settings")
        utilization = 0.80
        context_len = 6144
        tensor_parallel = 4
    elif min_free_gb >= 20:
        print("⚠️ Moderate - Can run with conservative settings")
        utilization = 0.75
        context_len = 4096
        tensor_parallel = 4
    elif min_free_gb >= 15:
        print("⚠️ Low memory - Use very conservative settings")
        utilization = 0.65
        context_len = 2048
        tensor_parallel = 4
    else:
        print("❌ Insufficient memory - Consider freeing GPU memory or using smaller model")
        utilization = 0.50
        context_len = 1024
        tensor_parallel = min(4, num_gpus)
    
    print(f"\nSuggested vLLM settings:")
    print(f"  VLLM_GPU_MEMORY_UTILIZATION={utilization}")
    print(f"  VLLM_MAX_MODEL_LEN={context_len}")
    print(f"  VLLM_TENSOR_PARALLEL_SIZE={tensor_parallel}")
    print(f"  VLLM_ENFORCE_EAGER=1  # For stability")
    
    # Check for processes using GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.stdout.strip():
            print(f"\n⚠️ Active GPU processes:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            print(f"\nTo free memory: pkill -f python; pkill -f vllm")
    except:
        pass
    
    return min_free_gb >= 15

def suggest_batch_sizes(min_free_gb):
    """Suggest optimal batch sizes based on available memory"""
    if min_free_gb >= 30:
        batch_size = 8
    elif min_free_gb >= 25:
        batch_size = 6
    elif min_free_gb >= 20:
        batch_size = 4
    elif min_free_gb >= 15:
        batch_size = 2
    else:
        batch_size = 1
    
    print(f"\nOptimal batch size: {batch_size} prompts")
    return batch_size

if __name__ == "__main__":
    can_run = check_gpu_memory()
    
    if torch.cuda.is_available():
        min_free = min([torch.cuda.get_device_properties(i).total_memory / (1024**3) - 
                       torch.cuda.memory_allocated(i) / (1024**3) - 
                       torch.cuda.memory_reserved(i) / (1024**3)
                       for i in range(torch.cuda.device_count())])
        suggest_batch_sizes(min_free)
    
    if can_run:
        print(f"\n🚀 Ready to run Gemma 4 31B!")
        sys.exit(0)
    else:
        print(f"\n❌ Not ready - free some GPU memory first")
        sys.exit(1)