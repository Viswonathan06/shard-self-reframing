#!/usr/bin/env python3
"""
Generate scratchpad reasoning traces using Qwen3 Coder 480B for SFT data.

This script uses AWS Bedrock API to call Qwen3-Coder-480B-A35B-Instruct 
to generate high-quality scratchpad reasoning for supervised fine-tuning.

Usage:
    python3 scripts/generate_scratchpad_data.py --input problems.jsonl --output scratchpad_data.jsonl
"""

import json
import boto3
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

class QwenScratchpadGenerator:
    def __init__(self, model_id: str = "qwen.qwen3-coder-480b-a35b-v1:0"):
        """Initialize the Qwen scratchpad generator."""
        self.model_id = model_id
        self.bedrock_client = boto3.client('bedrock-runtime')
        
        # Load scratchpad prompt template
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "scratchpad_sft_generation_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
    
    def generate_scratchpad(self, problem: str, max_tokens: int = 4096, temperature: float = 0.7) -> Dict:
        """Generate scratchpad reasoning for a given problem."""
        
        # Fill in the prompt template
        full_prompt = self.prompt_template.format(problem=problem)
        
        # Prepare request body
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["</scratchpad>"],  # Don't stop at scratchpad end
        }
        
        try:
            # Call Bedrock API
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            generated_text = response_body['choices'][0]['message']['content']
            
            return {
                "problem": problem,
                "scratchpad_response": generated_text,
                "model": self.model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "problem": problem,
                "scratchpad_response": None,
                "model": self.model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "success": False,
                "error": str(e)
            }
    
    def generate_batch(self, problems: List[str], output_file: str, 
                      max_tokens: int = 4096, temperature: float = 0.7,
                      delay: float = 1.0) -> None:
        """Generate scratchpad data for a batch of problems."""
        
        results = []
        failed_count = 0
        
        print(f"Generating scratchpad data for {len(problems)} problems...")
        print(f"Model: {self.model_id}")
        print(f"Output: {output_file}")
        
        with tqdm(total=len(problems), desc="Generating") as pbar:
            for i, problem in enumerate(problems):
                pbar.set_description(f"Processing problem {i+1}")
                
                result = self.generate_scratchpad(
                    problem=problem,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                results.append(result)
                
                if not result['success']:
                    failed_count += 1
                    print(f"\n❌ Failed problem {i+1}: {result['error']}")
                
                # Save intermediate results
                if (i + 1) % 10 == 0 or i == len(problems) - 1:
                    self._save_results(results, output_file)
                
                pbar.update(1)
                
                # Rate limiting
                if delay > 0 and i < len(problems) - 1:
                    time.sleep(delay)
        
        print(f"\n✅ Generation complete!")
        print(f"   Successful: {len(problems) - failed_count}/{len(problems)}")
        print(f"   Failed: {failed_count}/{len(problems)}")
        print(f"   Output saved to: {output_file}")
    
    def _save_results(self, results: List[Dict], output_file: str) -> None:
        """Save results to JSONL file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

def load_problems(input_file: str) -> List[str]:
    """Load problems from input file (JSONL or text)."""
    problems = []
    
    if input_file.endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # Try different common field names
                problem = (data.get('problem') or 
                          data.get('question') or 
                          data.get('prompt') or 
                          data.get('text') or
                          str(data))
                problems.append(problem)
    else:
        # Assume plain text file, one problem per line
        with open(input_file, 'r', encoding='utf-8') as f:
            problems = [line.strip() for line in f if line.strip()]
    
    return problems

def main():
    parser = argparse.ArgumentParser(description="Generate scratchpad reasoning data using Qwen3 Coder 480B")
    parser.add_argument("--input", "-i", required=True, help="Input file with problems (JSONL or text)")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file for scratchpad data")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--model-id", default="qwen.qwen3-coder-480b-a35b-v1:0", help="Bedrock model ID")
    
    args = parser.parse_args()
    
    # Load problems
    print(f"Loading problems from: {args.input}")
    problems = load_problems(args.input)
    print(f"Loaded {len(problems)} problems")
    
    if not problems:
        print("❌ No problems found in input file!")
        return
    
    # Initialize generator
    generator = QwenScratchpadGenerator(model_id=args.model_id)
    
    # Generate scratchpad data
    generator.generate_batch(
        problems=problems,
        output_file=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        delay=args.delay
    )

if __name__ == "__main__":
    main()