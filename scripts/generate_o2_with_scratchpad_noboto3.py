#!/usr/bin/env python3
"""
Generate O2 responses with scratchpad reasoning using Qwen3 Coder 480B via AWS Bedrock.
This version uses a custom AWS client instead of boto3.
"""

import json
import argparse
from pathlib import Path
import subprocess
import hashlib
import hmac
from datetime import datetime
import os


class BedrockClient:
    def __init__(self, region='us-east-1'):
        self.region = region
        self.service = 'bedrock-runtime'
        self.host = f'{self.service}.{region}.amazonaws.com'
        
        # Get AWS credentials from environment
        self.access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        self.secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        
        if not self.access_key or not self.secret_key:
            raise ValueError("AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
    
    def _sign(self, key, msg):
        """Generate HMAC-SHA256 signature"""
        return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()
    
    def _get_signature_key(self, key, date_stamp, region_name, service_name):
        """Generate AWS signature key"""
        k_date = self._sign(('AWS4' + key).encode('utf-8'), date_stamp)
        k_region = self._sign(k_date, region_name)
        k_service = self._sign(k_region, service_name)
        k_signing = self._sign(k_service, 'aws4_request')
        return k_signing
    
    def invoke_model(self, model_id, body):
        """Invoke a model using AWS Bedrock Runtime"""
        
        # Prepare request details
        method = 'POST'
        uri = f'/model/{model_id}/invoke'
        query_string = ''
        
        # Current time
        t = datetime.utcnow()
        amzdate = t.strftime('%Y%m%dT%H%M%SZ')
        datestamp = t.strftime('%Y%m%d')
        
        # Create canonical request
        payload_hash = hashlib.sha256(body.encode('utf-8')).hexdigest()
        
        canonical_headers = f'host:{self.host}\nx-amz-date:{amzdate}\n'
        signed_headers = 'host;x-amz-date'
        canonical_request = f'{method}\n{uri}\n{query_string}\n{canonical_headers}\n{signed_headers}\n{payload_hash}'
        
        # Create string to sign
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = f'{datestamp}/{self.region}/{self.service}/aws4_request'
        string_to_sign = f'{algorithm}\n{amzdate}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'
        
        # Calculate signature
        signing_key = self._get_signature_key(self.secret_key, datestamp, self.region, self.service)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # Create authorization header
        authorization_header = f'{algorithm} Credential={self.access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}'
        
        # Make request using curl
        url = f'https://{self.host}{uri}'
        
        curl_cmd = [
            'curl', '-X', 'POST',
            '-H', f'Authorization: {authorization_header}',
            '-H', f'X-Amz-Date: {amzdate}',
            '-H', 'Content-Type: application/json',
            '-d', body,
            url
        ]
        
        try:
            result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise Exception(f"Curl failed: {result.stderr}")
            
            response = json.loads(result.stdout)
            return response
            
        except subprocess.TimeoutExpired:
            raise Exception("Request timed out after 120 seconds")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse response as JSON: {e}\nResponse: {result.stdout}")


def load_prompt_template(template_path):
    """Load the prompt template from file."""
    with open(template_path, 'r') as f:
        return f.read().strip()


def load_guidelines(guidelines_dir, category):
    """Load universal and category-specific guidelines."""
    guidelines_path = Path(guidelines_dir)
    
    # Load universal guidelines
    universal_file = guidelines_path / "universal_guidelines.txt"
    if universal_file.exists():
        with open(universal_file, 'r') as f:
            universal_guidelines = f.read().strip()
    else:
        universal_guidelines = "General safety and helpfulness guidelines apply."
    
    # Convert category name to filename format
    category_filename = category.lower().replace(' & ', '_and_').replace(' ', '_') + '_guidelines.txt'
    category_file = guidelines_path / category_filename
    
    if category_file.exists():
        with open(category_file, 'r') as f:
            category_guidelines = f.read().strip()
    else:
        category_guidelines = f"Category-specific guidelines for {category} apply."
    
    return universal_guidelines, category_guidelines


def generate_o2_with_scratchpad(client, prompt_template, p1, o1, category, universal_guidelines, category_guidelines):
    """Generate O2 response with scratchpad using Qwen3 Coder 480B."""
    
    # Fill in the prompt template
    filled_prompt = prompt_template.format(
        P1=p1,
        O1=o1,
        CATEGORY=category,
        UNIVERSAL_GUIDELINES=universal_guidelines,
        CATEGORY_GUIDELINES=category_guidelines
    )
    
    # Prepare request for Qwen3 Coder 480B
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": filled_prompt
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    # Call the model
    response = client.invoke_model('qwen.qwen3-coder-480b-a35b-v1:0', json.dumps(request_body))
    
    # Extract the generated text
    if 'output' in response and 'message' in response['output']:
        generated_text = response['output']['message']['content']
    else:
        # Try alternative response format
        generated_text = str(response)
    
    return generated_text


def extract_scratchpad_and_response(generated_text):
    """Extract scratchpad and O2 response from the generated text."""
    
    if '<scratchpad>' in generated_text and '</scratchpad>' in generated_text:
        # Extract scratchpad
        start_idx = generated_text.find('<scratchpad>') + len('<scratchpad>')
        end_idx = generated_text.find('</scratchpad>')
        scratchpad = generated_text[start_idx:end_idx].strip()
        
        # Extract O2 response (everything after </scratchpad>)
        o2_response = generated_text[end_idx + len('</scratchpad>'):].strip()
    else:
        # Fallback parsing
        if 'STEP 1 - HARM ASSESSMENT' in generated_text:
            # Try to find the end of the scratchpad
            lines = generated_text.split('\n')
            scratchpad_lines = []
            o2_lines = []
            in_scratchpad = True
            
            for line in lines:
                if 'STEP 4 - CARING ENGAGEMENT' in line:
                    scratchpad_lines.append(line)
                    # Look for the end of step 4
                    continue
                elif in_scratchpad and ('STEP' in line or any(keyword in line for keyword in ['Assessment', 'Need', 'Restriction', 'Engagement'])):
                    scratchpad_lines.append(line)
                elif in_scratchpad and line.strip() and not line.startswith('STEP'):
                    scratchpad_lines.append(line)
                else:
                    in_scratchpad = False
                    if line.strip():
                        o2_lines.append(line)
            
            scratchpad = '\n'.join(scratchpad_lines)
            o2_response = '\n'.join(o2_lines)
        else:
            # Complete fallback
            scratchpad = "Generated reasoning not in expected format"
            o2_response = generated_text
    
    return scratchpad.strip(), o2_response.strip()


def main():
    parser = argparse.ArgumentParser(description="Generate O2 responses with scratchpad using Qwen3 Coder 480B")
    parser.add_argument('--input', required=True, help='Input JSONL file with baseline data')
    parser.add_argument('--output', required=True, help='Output JSONL file for generated data')
    parser.add_argument('--prompt-template', default='src/prompts/dna_generate_o2_with_scratchpad.txt', help='Path to prompt template')
    parser.add_argument('--guidelines-dir', required=True, help='Directory containing guideline files')
    parser.add_argument('--max-records', type=int, default=None, help='Maximum records to process')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting O2 generation with Qwen3 Coder 480B")
    print(f"📥 Input: {args.input}")
    print(f"📤 Output: {args.output}")
    print(f"📋 Template: {args.prompt_template}")
    print(f"📚 Guidelines: {args.guidelines_dir}")
    
    # Initialize Bedrock client
    try:
        client = BedrockClient()
        print("✅ AWS Bedrock client initialized")
    except ValueError as e:
        print(f"❌ {e}")
        print("Please set your AWS credentials:")
        print("export AWS_ACCESS_KEY_ID=your_access_key")
        print("export AWS_SECRET_ACCESS_KEY=your_secret_key")
        return
    
    # Load prompt template
    prompt_template = load_prompt_template(args.prompt_template)
    print(f"✅ Loaded prompt template ({len(prompt_template)} chars)")
    
    # Process records
    processed_records = []
    
    with open(args.input, 'r') as f:
        for i, line in enumerate(f):
            if args.max_records and i >= args.max_records:
                break
                
            record = json.loads(line.strip())
            
            # Extract required fields
            p1 = record['p1']
            o1 = record['baseline_output']  # This is the O1 baseline response
            category = record['category']
            
            # Load guidelines
            universal_guidelines, category_guidelines = load_guidelines(args.guidelines_dir, category)
            
            print(f"\n📝 Processing {i+1}: {category}")
            print(f"   P1: {p1[:60]}...")
            
            try:
                # Generate O2 with scratchpad
                generated_text = generate_o2_with_scratchpad(
                    client, prompt_template, p1, o1, category, 
                    universal_guidelines, category_guidelines
                )
                
                # Extract components
                generated_scratchpad, generated_o2 = extract_scratchpad_and_response(generated_text)
                
                print(f"   ✅ Generated O2 ({len(generated_o2)} chars)")
                print(f"   🧠 Scratchpad ({len(generated_scratchpad)} chars)")
                
                # Create output record
                output_record = {
                    'p1_id': record.get('p1_id', str(i)),
                    'p1': p1,
                    'o1': o1,
                    'category': category,
                    'generated_scratchpad': generated_scratchpad,
                    'generated_o2': generated_o2,
                    'raw_generation': generated_text,
                    'model_used': 'qwen3-coder-480b-bedrock',
                    'universal_guidelines': universal_guidelines[:200] + "..." if len(universal_guidelines) > 200 else universal_guidelines,
                    'category_guidelines': category_guidelines[:200] + "..." if len(category_guidelines) > 200 else category_guidelines
                }
                
                processed_records.append(output_record)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                # Create error record
                error_record = {
                    'p1_id': record.get('p1_id', str(i)),
                    'p1': p1,
                    'o1': o1,
                    'category': category,
                    'error': str(e),
                    'model_used': 'qwen3-coder-480b-bedrock-error'
                }
                processed_records.append(error_record)
    
    # Write results
    with open(args.output, 'w') as f:
        for record in processed_records:
            f.write(json.dumps(record) + '\n')
    
    success_count = sum(1 for r in processed_records if 'error' not in r)
    error_count = len(processed_records) - success_count
    
    print(f"\n🎉 Completed!")
    print(f"✅ Successful generations: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📄 Results written to: {args.output}")


if __name__ == '__main__':
    main()