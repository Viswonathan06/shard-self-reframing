#!/usr/bin/env python3
"""
Generate O2 responses with scratchpad reasoning using P1, O1, guidelines, and category.

This script generates improved responses (O2) with transparent reasoning processes
using the 4-step philosophical framework for simultaneous safety and helpfulness.

Usage:
    python3 scripts/generate_o2_with_scratchpad.py --input baseline_data.jsonl --output o2_with_scratchpad.jsonl --guidelines-dir src/guidelines/
"""

import json
import boto3
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

class O2ScratchpadGenerator:
    def __init__(self, model_id: str = "qwen.qwen3-coder-480b-a35b-v1:0"):
        """Initialize the O2 scratchpad generator."""
        self.model_id = model_id
        self.bedrock_client = boto3.client('bedrock-runtime')
        
        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "src" / "prompts" / "dna_generate_o2_with_scratchpad.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
    
    def load_guidelines(self, guidelines_dir: str) -> Dict[str, str]:
        """Load universal and category-specific guidelines."""
        guidelines_path = Path(guidelines_dir)
        guidelines = {}
        
        # Load universal guidelines
        universal_path = guidelines_path / "universal_guidelines.txt"
        if universal_path.exists():
            with open(universal_path, 'r', encoding='utf-8') as f:
                guidelines["universal"] = f.read().strip()
        else:
            guidelines["universal"] = "Prioritize human wellbeing and safety while being maximally helpful."
        
        # Load category-specific guidelines
        category_files = guidelines_path.glob("*_guidelines.txt")
        for cat_file in category_files:
            if cat_file.name != "universal_guidelines.txt":
                category = cat_file.stem.replace("_guidelines", "")
                with open(cat_file, 'r', encoding='utf-8') as f:
                    guidelines[category] = f.read().strip()
        
        return guidelines
    
    def get_category_guidelines(self, category: str, guidelines: Dict[str, str]) -> str:
        """Get guidelines for a specific category."""
        # Try exact match first
        if category in guidelines:
            return guidelines[category]
        
        # Try normalized matches
        category_normalized = category.lower().replace(" ", "_").replace("&", "and")
        for key in guidelines:
            if key.lower().replace(" ", "_").replace("&", "and") == category_normalized:
                return guidelines[key]
        
        # Default to universal if no specific guidelines found
        return guidelines.get("universal", "Apply general safety principles.")
    
    def generate_o2_with_scratchpad(self, p1: str, o1: str, category: str, 
                                  universal_guidelines: str, category_guidelines: str,
                                  max_tokens: int = 4096, temperature: float = 0.4) -> Dict:
        """Generate O2 response with scratchpad reasoning."""
        
        # Fill in the prompt template
        full_prompt = self.prompt_template.format(
            P1=p1,
            O1=o1, 
            CATEGORY=category,
            UNIVERSAL_GUIDELINES=universal_guidelines,
            CATEGORY_GUIDELINES=category_guidelines
        )
        
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
            
            # Extract scratchpad and O2 sections
            scratchpad_section = ""
            o2_response = ""
            
            if "<scratchpad>" in generated_text and "</scratchpad>" in generated_text:
                start_idx = generated_text.find("<scratchpad>") + len("<scratchpad>")
                end_idx = generated_text.find("</scratchpad>")
                scratchpad_section = generated_text[start_idx:end_idx].strip()
                
                # O2 is everything after </scratchpad>
                o2_start = generated_text.find("</scratchpad>") + len("</scratchpad>")
                o2_response = generated_text[o2_start:].strip()
            else:
                # If no scratchpad tags, treat entire response as O2
                o2_response = generated_text.strip()
            
            return {
                "p1": p1,
                "o1": o1,
                "category": category,
                "universal_guidelines": universal_guidelines,
                "category_guidelines": category_guidelines,
                "generated_scratchpad": scratchpad_section,
                "generated_o2": o2_response,
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
                "p1": p1,
                "o1": o1,
                "category": category,
                "universal_guidelines": universal_guidelines,
                "category_guidelines": category_guidelines,
                "generated_scratchpad": None,
                "generated_o2": None,
                "full_generated_text": None,
                "model": self.model_id,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def generate_batch(self, input_file: str, output_file: str, guidelines_dir: str,
                      max_tokens: int = 4096, temperature: float = 0.4,
                      delay: float = 1.0, max_records: Optional[int] = None) -> None:
        """Generate O2 responses with scratchpad for a batch of records."""
        
        # Load guidelines
        print(f"Loading guidelines from: {guidelines_dir}")
        guidelines = self.load_guidelines(guidelines_dir)
        print(f"Loaded guidelines for: {list(guidelines.keys())}")
        
        # Load input records
        records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        
        if max_records:
            records = records[:max_records]
        
        results = []
        failed_count = 0
        
        print(f"\nGenerating O2 responses with scratchpad for {len(records)} records...")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Model: {self.model_id}")
        
        with tqdm(total=len(records), desc="Generating O2+scratchpad") as pbar:
            for i, record in enumerate(records):
                pbar.set_description(f"Processing record {i+1}")
                
                # Extract required fields
                p1 = record.get('p1', '')
                o1 = record.get('output', record.get('o1', ''))  # Try both field names
                category = record.get('category', 'General')
                p1_id = record.get('p1_id', str(i))
                
                # Get appropriate guidelines
                universal_guidelines = guidelines.get("universal", "")
                category_guidelines = self.get_category_guidelines(category, guidelines)
                
                # Generate O2 with scratchpad
                result = self.generate_o2_with_scratchpad(
                    p1=p1,
                    o1=o1,
                    category=category,
                    universal_guidelines=universal_guidelines,
                    category_guidelines=category_guidelines,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Add original record metadata
                result["p1_id"] = p1_id
                for key in ["lang", "level", "response_mode", "reformulated_component", "preserved_benign_need"]:
                    if key in record:
                        result[key] = record[key]
                
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
    parser = argparse.ArgumentParser(description="Generate O2 responses with scratchpad reasoning")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL file with P1, O1, category")
    parser.add_argument("--output", "-o", required=True, help="Output JSONL file for O2+scratchpad data")
    parser.add_argument("--guidelines-dir", "-g", required=True, help="Directory containing guideline files")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.4, help="Generation temperature")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    parser.add_argument("--max-records", type=int, help="Limit number of records to process")
    parser.add_argument("--model-id", default="qwen.qwen3-coder-480b-a35b-v1:0", help="Bedrock model ID")
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return
    
    if not Path(args.guidelines_dir).exists():
        print(f"❌ Guidelines directory not found: {args.guidelines_dir}")
        return
    
    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = O2ScratchpadGenerator(model_id=args.model_id)
    
    # Generate O2 responses with scratchpad
    generator.generate_batch(
        input_file=args.input,
        output_file=args.output,
        guidelines_dir=args.guidelines_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        delay=args.delay,
        max_records=args.max_records
    )

if __name__ == "__main__":
    main()