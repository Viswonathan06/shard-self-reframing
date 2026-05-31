#!/usr/bin/env python3
"""
Test script for scratchpad generation without AWS dependencies.
Simulates the Qwen3 Coder 480B response for testing purposes.
"""

import json
import argparse
from pathlib import Path


def simulate_qwen_response(p1, o1, category, universal_guidelines, category_guidelines):
    """Simulate a Qwen3 Coder 480B response for testing."""
    
    # Mock response that follows our expected format
    mock_response = f"""<scratchpad>
STEP 1 - HARM ASSESSMENT 
The user is asking about {category.lower()} content. This involves potential harm to individuals through inappropriate content exposure.
Category: {category}
Severity: medium (based on explicit nature but no direct physical harm)
Specific risks: Normalization of inappropriate content, potential discomfort to users seeking help

STEP 2 - LEGITIMATE NEED 
Underlying need: The user likely wants to understand appropriate boundaries and healthy relationship dynamics.
This is a legitimate interest in learning about respectful human interaction and social norms.
Benign applications: Educational content about healthy relationships, understanding social boundaries
Educational value: Learning about mutual respect and appropriate communication

STEP 3 - MINIMAL RESTRICTION 
Current baseline: {o1[:100]}...
Improvement opportunity: We can provide more educational value while maintaining clear boundaries
Least restrictive response: Provide information about healthy relationship dynamics and communication
without enabling inappropriate requests or normalizing harmful content.
Safe alternatives: Educational resources about respectful communication, relationship counseling information

STEP 4 - CARING ENGAGEMENT 
Tone: Respectful acknowledgment of the user's potential underlying need for information
Approach: Redirect toward educational and constructive content about healthy relationships
Value provided: Actual helpful information about appropriate social interaction
Boundaries: Clear explanation that explicit content cannot be provided but related educational content can be
</scratchpad>

I understand you might be curious about this topic, but I can't provide explicit content. Instead, I can offer information about healthy relationships and appropriate social interaction.

Healthy adult relationships are built on mutual respect, clear communication, and consent. If you're interested in learning about relationship dynamics, I'd be happy to discuss topics like:
- Building trust and emotional intimacy
- Effective communication skills
- Understanding boundaries and consent
- Resources for relationship counseling or education

Would any of these alternative topics be helpful to you?"""
    
    return mock_response


def extract_scratchpad_and_response(generated_text):
    """Extract scratchpad and response from the generated text."""
    
    if '<scratchpad>' in generated_text and '</scratchpad>' in generated_text:
        start_idx = generated_text.find('<scratchpad>') + len('<scratchpad>')
        end_idx = generated_text.find('</scratchpad>')
        scratchpad = generated_text[start_idx:end_idx].strip()
        response = generated_text[end_idx + len('</scratchpad>'):].strip()
    else:
        # Fallback if format is not perfect
        scratchpad = "STEP 1-4: Simulated scratchpad reasoning"
        response = generated_text
    
    return scratchpad, response


def load_guidelines(guidelines_dir, category):
    """Load universal and category-specific guidelines."""
    guidelines_path = Path(guidelines_dir)
    
    # Load universal guidelines
    universal_file = guidelines_path / "universal_guidelines.txt"
    if universal_file.exists():
        with open(universal_file, 'r') as f:
            universal_guidelines = f.read().strip()
    else:
        universal_guidelines = "General safety and helpfulness guidelines"
    
    # Load category-specific guidelines
    category_file = guidelines_path / f"{category.lower().replace(' & ', '_and_').replace(' ', '_')}_guidelines.txt"
    if category_file.exists():
        with open(category_file, 'r') as f:
            category_guidelines = f.read().strip()
    else:
        category_guidelines = f"Category-specific guidelines for {category}"
    
    return universal_guidelines, category_guidelines


def main():
    parser = argparse.ArgumentParser(description="Test scratchpad generation locally")
    parser.add_argument('--input', required=True, help='Input JSONL file with baseline data')
    parser.add_argument('--output', required=True, help='Output JSONL file for generated data')
    parser.add_argument('--guidelines-dir', required=True, help='Directory containing guideline files')
    parser.add_argument('--max-records', type=int, default=10, help='Maximum records to process')
    
    args = parser.parse_args()
    
    print(f"Processing input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Guidelines directory: {args.guidelines_dir}")
    
    processed_records = []
    
    with open(args.input, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.max_records:
                break
                
            record = json.loads(line.strip())
            
            # Extract required fields
            p1 = record['p1']
            o1 = record['baseline_output']  # This is the O1 baseline response
            category = record['category']
            
            # Load guidelines
            universal_guidelines, category_guidelines = load_guidelines(args.guidelines_dir, category)
            
            print(f"Processing record {i+1}: {category} - {p1[:50]}...")
            
            # Generate response (simulated)
            generated_text = simulate_qwen_response(p1, o1, category, universal_guidelines, category_guidelines)
            
            # Extract scratchpad and O2 response
            generated_scratchpad, generated_o2 = extract_scratchpad_and_response(generated_text)
            
            # Create output record
            output_record = {
                'p1_id': record.get('p1_id', str(i)),
                'p1': p1,
                'o1': o1,  # Original baseline
                'category': category,
                'generated_scratchpad': generated_scratchpad,
                'generated_o2': generated_o2,
                'model_used': 'qwen3-coder-480b-simulated',
                'universal_guidelines': universal_guidelines,
                'category_guidelines': category_guidelines
            }
            
            processed_records.append(output_record)
    
    # Write results
    with open(args.output, 'w') as f:
        for record in processed_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Successfully processed {len(processed_records)} records")
    print(f"Results written to: {args.output}")


if __name__ == '__main__':
    main()