#!/bin/bash
# Setup AWS credentials for Qwen3 Coder 480B access

echo "🔑 AWS Credentials Setup for Qwen3 Coder 480B"
echo "=============================================="

if [[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo "✅ AWS credentials already set in environment"
    echo "Access Key ID: ${AWS_ACCESS_KEY_ID:0:20}..."
    echo "Secret Key: ${AWS_SECRET_ACCESS_KEY:0:10}..."
else
    echo "❌ AWS credentials not found in environment"
    echo ""
    echo "To set up AWS credentials, run:"
    echo "export AWS_ACCESS_KEY_ID=your_access_key_here"
    echo "export AWS_SECRET_ACCESS_KEY=your_secret_key_here"
    echo ""
    echo "Or create ~/.aws/credentials file with:"
    echo "[default]"
    echo "aws_access_key_id = your_access_key_here"
    echo "aws_secret_access_key = your_secret_key_here"
    echo ""
fi

echo ""
echo "📍 Available AWS regions for Bedrock:"
echo "- us-east-1 (N. Virginia) - Default"
echo "- us-west-2 (Oregon)"
echo "- eu-central-1 (Frankfurt)"
echo ""

# Test if credentials work
if [[ -n "$AWS_ACCESS_KEY_ID" && -n "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo "🧪 Testing credentials with a simple API call..."
    
    # Simple test using our custom client
    cd $PROJECT_ROOT
    python3 scripts/aws_bedrock_client.py
    
    if [[ $? -eq 0 ]]; then
        echo "✅ AWS credentials working!"
    else
        echo "❌ Credential test failed"
    fi
fi