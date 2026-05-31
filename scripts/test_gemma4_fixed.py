#!/usr/bin/env python3
"""
Fixed test script for Gemma 4 31B with proper tokenization
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, "/common/guam/Safeshift")

def test_gemma4_with_proper_settings():
    """Test Gemma 4 31B with corrected tokenization settings"""
    print("=== Testing Gemma 4 31B with Fixed Settings ===")
    
    # Set up environment with corrected settings
    os.environ['USE_LOCAL_MODEL'] = 'true'
    os.environ['VLLM_TRUST_REMOTE_CODE'] = '1'
    os.environ['VLLM_TENSOR_PARALLEL_SIZE'] = '4'
    os.environ['VLLM_GPU_MEMORY_UTILIZATION'] = '0.70'
    os.environ['VLLM_MAX_MODEL_LEN'] = '4096'
    os.environ['VLLM_ENFORCE_EAGER'] = '1'
    
    # Use the HuggingFace repo ID to ensure proper tokenizer loading
    model_id = "google/gemma-4-31B-it"
    os.environ['LOCAL_MODEL'] = model_id
    
    try:
        print(f"Loading Gemma 4 31B from: {model_id}")
        from vllm import LLM, SamplingParams
        
        # Initialize model with proper settings for Gemma
        llm = LLM(
            model=model_id,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.70,
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
            dtype='bfloat16',
            disable_custom_all_reduce=True,
            # Gemma-specific settings
            tokenizer=model_id,  # Ensure tokenizer matches model
            skip_tokenizer_init=False,  # Important for Gemma
        )
        
        print("✅ Model loaded successfully!")
        
        # Test with proper sampling parameters for Gemma
        sampling_params = SamplingParams(
            temperature=0.7,  # Higher temperature for better generation
            top_p=0.9,
            max_tokens=150,   # Reasonable length
            stop=["\n\n", "<eos>", "</s>"],  # Proper stop tokens
            repetition_penalty=1.1,  # Reduce repetition
        )
        
        test_prompts = [
            "What is artificial intelligence safety?",
            "Explain the importance of AI alignment.",
            "How can we ensure AI systems behave responsibly?"
        ]
        
        print(f"Running inference on {len(test_prompts)} test prompts...")
        start_time = time.time()
        
        # Generate responses
        outputs = llm.generate(test_prompts, sampling_params)
        
        inference_time = time.time() - start_time
        print(f"✅ Inference completed in {inference_time:.2f}s")
        
        # Print results
        for i, (prompt, output) in enumerate(zip(test_prompts, outputs)):
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt}")
            response = output.outputs[0].text.strip()
            print(f"Response: {response}")
            print(f"Finish reason: {output.outputs[0].finish_reason}")
            
            # Check for repetition issues
            if len(set(response.replace(' ', '')[:50])) < 5:
                print("⚠️ Warning: Response seems repetitive")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_chat_template():
    """Test using chat template for better responses"""
    print("\n=== Testing with Chat Template ===")
    
    try:
        from vllm import LLM, SamplingParams
        
        model_id = "google/gemma-4-31B-it"
        
        llm = LLM(
            model=model_id,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.70,
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True
        )
        
        # Use chat format for Gemma
        messages = [
            {"role": "user", "content": "What are the key challenges in AI safety research?"}
        ]
        
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=200,
            repetition_penalty=1.2
        )
        
        print("Testing chat-based generation...")
        
        # Use chat method if available
        if hasattr(llm, 'chat'):
            outputs = llm.chat(messages, sampling_params=sampling_params)
            response = outputs[0].outputs[0].text
        else:
            # Fallback to manual template
            prompt = f"<start_of_turn>user\n{messages[0]['content']}<end_of_turn>\n<start_of_turn>model\n"
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
        
        print(f"Chat Response: {response}")
        return True
        
    except Exception as e:
        print(f"❌ Chat test error: {e}")
        return False

if __name__ == "__main__":
    print("Gemma 4 31B Fixed Test Suite")
    print("=" * 50)
    
    success = True
    
    # Test 1: Basic generation with fixed settings
    success &= test_gemma4_with_proper_settings()
    
    # Test 2: Chat template test
    success &= test_with_chat_template()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Fixed tests completed!")
        print("\nThe tokenization issue should be resolved.")
        print("If still seeing repetition, try:")
        print("1. Higher temperature (0.8-1.0)")
        print("2. Higher repetition_penalty (1.2-1.5)")  
        print("3. Different stop tokens")
    else:
        print("❌ Tests failed. Check error messages above.")
        sys.exit(1)