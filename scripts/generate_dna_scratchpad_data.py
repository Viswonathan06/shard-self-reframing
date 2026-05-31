#!/usr/bin/env python3
"""
Generate scratchpad reasoning traces for DNA safety intervention dataset.

This script takes your DNA refinement data and generates scratchpad reasoning
that demonstrates the 4-step philosophical framework for simultaneous safety and helpfulness.

Usage:
    python3 scripts/generate_dna_scratchpad_data.py --input output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/llama8b/baseline_outputs.jsonl --output output/scratchpad_dna_data.jsonl
"""

import json
import boto3
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

class DNAScratchpadGenerator:
    def __init__(self, model_id: str = "qwen.qwen3-coder-480b-a35b-v1:0"):
        """Initialize the DNA scratchpad generator."""
        self.model_id = model_id
        self.bedrock_client = boto3.client('bedrock-runtime')
        
        # Load DNA-specific scratchpad prompt template
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "dna_scratchpad_generation_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
    
    def generate_scratchpad_for_dna_record(self, record: Dict, max_tokens: int = 4096, temperature: float = 0.3) -> Dict:
        """Generate scratchpad reasoning for a DNA dataset record."""
        
        # Extract the DNA record fields
        p1 = record.get('p1', '')
        p2 = record.get('p2', '')
        o2 = record.get('output', '')  # The refined response
        p1_id = record.get('p1_id', '')
        category = record.get('category', 'Unknown')
        
        # Fill in the prompt template
        full_prompt = self.prompt_template.format(P1=p1, P2=p2, O2=o2)
        
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
            
            # Extract scratchpad and response sections
            scratchpad_section = ""
            response_section = ""
            
            if "<scratchpad>" in generated_text and "</scratchpad>" in generated_text:
                start_idx = generated_text.find("<scratchpad>") + len("<scratchpad>")
                end_idx = generated_text.find("</scratchpad>")
                scratchpad_section = generated_text[start_idx:end_idx].strip()
                
            if "<response>" in generated_text and "</response>" in generated_text:
                start_idx = generated_text.find("<response>") + len("<response>")
                end_idx = generated_text.find("</response>")
                response_section = generated_text[start_idx:end_idx].strip()
            
            return {
                "p1_id": p1_id,
                "p1": p1,
                "p2": p2,
                "original_o2": o2,
                "category": category,
                "generated_scratchpad": scratchpad_section,
                "generated_response": response_section,
                "full_generated_text": generated_text,
                "model": self.model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "success": True,
                "error": None,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "p1_id": p1_id,
                "p1": p1,
                "p2": p2,
                "original_o2": o2,
                "category": category,
                "generated_scratchpad": None,
                "generated_response": None,
                "full_generated_text": None,
                "model": self.model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def generate_batch_from_dna_file(self, input_file: str, output_file: str,
                                   max_tokens: int = 4096, temperature: float = 0.3,
                                   delay: float = 1.0, max_records: Optional[int] = None) -> None:
        """Generate scratchpad data for a DNA refinement file."""
        
        # Load DNA records
        records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        if max_records:
            records = records[:max_records]
        
        results = []
        failed_count = 0
        
        print(f"Generating scratchpad data for {len(records)} DNA records...")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Model: {self.model_id}")
        
        with tqdm(total=len(records), desc="Generating") as pbar:
            for i, record in enumerate(records):
                pbar.set_description(f"Processing record {i+1}")
                
                result = self.generate_scratchpad_for_dna_record(
                    record=record,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                results.append(result)
                
                if not result['success']:
                    failed_count += 1
                    print(f"\n❌ Failed record {i+1}: {result['error']}")
                
                # Save intermediate results
                if (i + 1) % 10 == 0 or i == len(records) - 1:
                    self._save_results(results, output_file)
                
                pbar.update(1)
                
                # Rate limiting
                if delay > 0 and i < len(records) - 1:
                    time.sleep(delay)
        
        print(f"\n✅ Generation complete!")
        print(f"   Successful: {len(records) - failed_count}/{len(records)}")
        print(f"   Failed: {failed_count}/{len(records)}")
        print(f"   Output saved to: {output_file}")
    
    def _save_results(self, results: List[Dict], output_file: str) -> None:
        """Save results to JSONL file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Generate scratchpad reasoning for DNA safety intervention data")
    parser.add_argument("--input", "-i", required=True, help="Input DNA refinement JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Output scratchpad JSONL file")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.3, help="Generation temperature (low for consistent safety reasoning)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--max-records", type=int, help="Limit number of records to process")
    parser.add_argument("--model-id", default="qwen.qwen3-coder-480b-a35b-v1:0", help="Bedrock model ID")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = DNAScratchpadGenerator(model_id=args.model_id)
    
    # Generate scratchpad data
    generator.generate_batch_from_dna_file(
        input_file=args.input,
        output_file=args.output,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        delay=args.delay,
        max_records=args.max_records
    )

if __name__ == "__main__":
    main()