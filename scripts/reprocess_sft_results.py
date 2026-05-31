#!/usr/bin/env python3
"""
Script to reprocess SFT inference results with the fixed extraction function.
"""

import json
import sys
import os
sys.path.append('scripts')
from inference_sft_model import extract_scratchpad_and_response

def reprocess_results(input_file, output_file):
    """Reprocess the SFT inference results with improved extraction."""
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    processed_count = 0
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            sample = json.loads(line.strip())
            
            # Extract scratchpad and response using the fixed function
            raw_generation = sample.get('raw_generation', '')
            scratchpad, response = extract_scratchpad_and_response(raw_generation)
            
            # Update the sample with corrected extraction
            sample['sft_generated_scratchpad'] = scratchpad
            sample['sft_generated_response'] = response
            
            # Write the corrected sample
            outfile.write(json.dumps(sample) + '\n')
            processed_count += 1
            
            if processed_count <= 3:  # Show first few for verification
                print(f"\n=== Sample {processed_count} ===")
                print(f"P1: {sample['p1'][:50]}...")
                print(f"Scratchpad length: {len(scratchpad)}")
                print(f"Response length: {len(response)}")
                print(f"Scratchpad preview: {scratchpad[:100]}...")
                print(f"Response preview: {response[:100]}...")
    
    print(f"\nProcessed {processed_count} samples successfully!")
    return processed_count

if __name__ == "__main__":
    input_file = "output/sft_inference_results.jsonl"
    output_file = "output/sft_inference_results_fixed.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        sys.exit(1)
    
    count = reprocess_results(input_file, output_file)
    print(f"\nReprocessed results saved to: {output_file}")