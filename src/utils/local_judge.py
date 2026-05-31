# utils/local_judge.py
"""
Local judge model for scoring P2 prompts.
Uses Hugging Face models instead of API calls.

Recommended models for multilingual judging:
1. Qwen/Qwen2-7B-Instruct - Best multilingual, excellent for Chinese/Arabic/English
2. meta-llama/Llama-3-8B-Instruct - Good multilingual support
3. mistralai/Mistral-7B-Instruct-v0.2 - Decent multilingual
4. microsoft/Phi-3-mini-4k-instruct - Smaller, efficient
"""

import json
import os
import torch
from typing import Dict, Optional, List, Tuple
import logging

logging.basicConfig(level=logging.WARNING)

# Try to import transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")


class LocalJudge:
    """Local judge model for scoring P2 prompts."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """
        Initialize local judge model.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda', 'cpu', or None (auto-detect)
            load_in_8bit: Use 8-bit quantization (saves memory)
            load_in_4bit: Use 4-bit quantization (saves more memory)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available. Install with: pip install transformers torch")
        
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading local judge model: {model_name} on {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if requested
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32,
        }
        
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            except ImportError:
                print("Warning: bitsandbytes not available for 4-bit quantization")
                load_in_4bit = False
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        
        # Check if accelerate is available for device_map="auto"
        try:
            import accelerate
            if device == "cuda":
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = None
        except ImportError:
            # If accelerate is not available, don't use device_map
            print("Warning: accelerate not available, using manual device placement")
            model_kwargs["device_map"] = None
            if device == "cuda":
                model_kwargs["device"] = 0
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        
        # Move model to device if not using device_map
        if model_kwargs.get("device_map") != "auto" and device == "cuda":
            self.model = self.model.to(device)
        
        # Create pipeline for text generation
        # When device_map="auto" is used, model is already on device via accelerate
        # so we shouldn't pass device argument to pipeline
        pipeline_kwargs = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32
        }
        
        # Only add device if not using device_map="auto"
        if model_kwargs.get("device_map") != "auto":
            pipeline_kwargs["device"] = 0 if device == "cuda" else -1
        
        self.pipeline = pipeline("text-generation", **pipeline_kwargs)
        
        print(f"✓ Local judge model loaded successfully")
    
    def judge(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 500,
        temperature: float = 0.3
    ) -> str:
        """
        Judge a single prompt and return JSON response.
        
        Args:
            system_prompt: System prompt for the judge
            user_prompt: User prompt with P1 and P2 to judge
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            JSON string with scores
        """
        # Format prompt based on model's chat template
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        # Apply chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple format
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
        
        # Generate
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            return_full_text=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = outputs[0]["generated_text"].strip()
        return self._extract_json(response)
    
    def judge_batch(
        self,
        judge_inputs: List[Tuple[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.3,
        batch_size: int = 8
    ) -> List[str]:
        """
        Judge multiple prompts in TRUE batch (parallel processing).
        
        This is much faster than sequential processing because the model
        processes multiple prompts simultaneously.
        
        Args:
            judge_inputs: List of (system_prompt, user_prompt) tuples
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            batch_size: Process this many at a time (to avoid OOM)
        
        Returns:
            List of JSON strings (same order as inputs)
        """
        results = []
        
        # Process in batches
        for i in range(0, len(judge_inputs), batch_size):
            batch = judge_inputs[i:i + batch_size]
            
            # Format all prompts in the batch
            formatted_prompts = []
            for system_prompt, user_prompt in batch:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                
                try:
                    # Apply chat template
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    # Fallback: simple format
                    prompt = f"{system_prompt}\n\n{user_prompt}\n\nResponse:"
                
                formatted_prompts.append(prompt)
            
            # TRUE BATCH PROCESSING: Process all prompts in batch simultaneously
            try:
                # Tokenize all prompts in batch
                tokenized = self.tokenizer(
                    formatted_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048  # Adjust based on model context
                )
                
                # Move to device
                if self.device == "cuda":
                    tokenized = {k: v.to(self.model.device) for k, v in tokenized.items()}
                
                # Generate for all prompts in batch
                with torch.no_grad():
                    outputs = self.model.generate(
                        **tokenized,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode all responses
                batch_results = []
                for i, output_ids in enumerate(outputs):
                    # Remove input tokens (only keep generated part)
                    input_length = tokenized['input_ids'][i].shape[0]
                    generated_ids = output_ids[input_length:]
                    
                    response = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True
                    ).strip()
                    
                    # Clean and extract JSON
                    response = self._extract_json(response)
                    batch_results.append(response)
                
                results.extend(batch_results)
                
            except Exception as e:
                # If batch processing fails, fall back to sequential
                logging.warning(f"Batch processing failed, falling back to sequential: {e}")
                for system_prompt, user_prompt in batch:
                    try:
                        result = self.judge(system_prompt, user_prompt, max_tokens, temperature)
                        results.append(result)
                    except Exception as e2:
                        logging.warning(f"Judge batch item failed: {e2}")
                        results.append("{}")  # Empty JSON on failure
        
        return results
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from model response."""
        response = response.strip()
        
        # Remove markdown code fences if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
        
        # Try to find JSON object
        try:
            # Try parsing as-is
            json.loads(response)
            return response
        except json.JSONDecodeError:
            # Try to extract JSON object
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                return json_match.group(0)
            # Last resort: return response and let caller handle it
            return response


# Global judge instance (lazy loaded)
_judge_instance = None

def get_local_judge(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> LocalJudge:
    """
    Get or create global local judge instance.
    
    Args:
        model_name: Model to use (default from env or Qwen2.5-7B)
        device: Device to use
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization
    
    Returns:
        LocalJudge instance
    """
    global _judge_instance
    
    if model_name is None:
        model_name = os.getenv("LOCAL_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    
    if _judge_instance is None or _judge_instance.model_name != model_name:
        _judge_instance = LocalJudge(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
    
    return _judge_instance

