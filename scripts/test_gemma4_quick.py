#!/usr/bin/env python3
"""
Quick test script for Gemma 4 31B with vLLM
Usage: python scripts/test_gemma4_quick.py
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, "/common/guam/Safeshift")

def test_gemma4_basic():
    """Basic test of Gemma 4 31B model loading and inference"""
    print("=== Testing Gemma 4 31B with vLLM ===")
    
    # Set up environment for Gemma 4 31B with conservative memory settings
    os.environ['USE_LOCAL_MODEL'] = 'true'
    os.environ['VLLM_ONLY'] = '1'
    os.environ['VLLM_TRUST_REMOTE_CODE'] = '1'
    os.environ['VLLM_TENSOR_PARALLEL_SIZE'] = '4'
    os.environ['VLLM_GPU_MEMORY_UTILIZATION'] = '0.70'  # Conservative setting
    os.environ['VLLM_MAX_MODEL_LEN'] = '4096'           # Smaller context
    os.environ['VLLM_ENFORCE_EAGER'] = '1'              # Stability
    
    # Try to resolve the model path from shared cache
    shared_path = "/playpen/huggingface/hub/models--google--gemma-4-31B-it"
    if os.path.exists(f"{shared_path}/snapshots"):
        import glob
        snapshots = glob.glob(f"{shared_path}/snapshots/*")
        if snapshots:
            model_path = max(snapshots)  # Use latest snapshot
            if os.path.exists(f"{model_path}/config.json"):
                os.environ['LOCAL_MODEL'] = model_path
                print(f"✓ Found Gemma 4 31B at: {model_path}")
            else:
                print(f"❌ No config.json in snapshot: {model_path}")
                return False
        else:
            print(f"❌ No snapshots found in: {shared_path}/snapshots")
            return False
    else:
        print(f"❌ Shared cache not found: {shared_path}")
        return False
    
    # Test model loading
    try:
        print("Loading vLLM with Gemma 4 31B...")
        from vllm import LLM, SamplingParams
        
        # Check GPU memory first
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                total = props.total_memory / (1024**3)
                free = total - allocated
                print(f"GPU {i}: {free:.1f}GB free / {total:.1f}GB total")
        
        llm = LLM(
            model=model_path,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.70,  # Conservative
            max_model_len=4096,           # Smaller context
            trust_remote_code=True,
            dtype='bfloat16',
            enforce_eager=True,           # More stable
            disable_custom_all_reduce=True  # Avoid multi-GPU issues
        )
        
        print("✓ Model loaded successfully!")
        
        # Test inference
        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=200
        )
        
        test_prompts = [
            "What is artificial intelligence safety?",
            "Explain the importance of AI alignment in a few sentences.",
            "How can we ensure AI systems behave responsibly?"
        ]
        
        print(f"Running inference on {len(test_prompts)} test prompts...")
        start_time = time.time()
        
        outputs = llm.generate(test_prompts, sampling_params)
        
        inference_time = time.time() - start_time
        print(f"✅ Inference completed in {inference_time:.2f}s ({len(test_prompts)/inference_time:.2f} prompts/sec)")
        
        # Print sample outputs
        for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt}")
            print(f"Response: {output.outputs[0].text[:150]}...")
            print(f"Finish reason: {output.outputs[0].finish_reason}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_client():
    """Test using Gemma 4 31B through the model client"""
    print("\n=== Testing Model Client with Gemma 4 31B ===")
    
    try:
        from src.utils.model_client import call_model
        
        system_prompt = "You are a helpful AI assistant focused on AI safety and alignment."
        user_prompt = "What are the key challenges in AI safety?"
        
        print("Testing model client...")
        start_time = time.time()
        
        response = call_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=300,
            temperature=0.3,
            model_name=os.environ.get('LOCAL_MODEL')
        )
        
        inference_time = time.time() - start_time
        print(f"✅ Model client inference completed in {inference_time:.2f}s")
        print(f"Response: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Model client error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Gemma 4 31B Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test 1: Basic vLLM test
    success &= test_gemma4_basic()
    
    # Test 2: Model client test
    success &= test_model_client()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Gemma 4 31B is ready for use.")
        print("\nNext steps:")
        print("1. Submit SLURM job: sbatch scripts/slurm/job_gemma4_31b_test.slurm")
        print("2. Run on real data with your pipeline scripts")
    else:
        print("❌ Some tests failed. Check error messages above.")
        sys.exit(1)