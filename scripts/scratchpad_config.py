"""
Configuration for scratchpad data generation using Qwen3 Coder 480B.
"""

import os
from typing import Dict, Any

# AWS Bedrock Configuration
BEDROCK_CONFIG = {
    "model_id": "qwen.qwen3-coder-480b-a35b-v1:0",
    "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    "max_tokens": 4096,
    "temperature": 0.7,
    "top_p": 0.9,
    "request_delay": 1.0,  # Seconds between requests
}

# Generation Parameters
GENERATION_CONFIG = {
    "general": {
        "temperature": 0.7,
        "max_tokens": 4096,
        "use_safety_prompt": False,
    },
    "safety_critical": {
        "temperature": 0.3,  # More conservative for safety analysis
        "max_tokens": 6144,  # More space for detailed reasoning
        "use_safety_prompt": True,
    },
    "coding": {
        "temperature": 0.5,  # Balanced creativity/precision for code
        "max_tokens": 4096,
        "use_safety_prompt": False,
    },
    "math": {
        "temperature": 0.2,  # Low temperature for mathematical precision
        "max_tokens": 3072,
        "use_safety_prompt": False,
    }
}

# Quality Filters
QUALITY_FILTERS = {
    "min_scratchpad_tokens": 100,  # Minimum tokens in scratchpad section
    "required_scratchpad_elements": [
        "step",  # Should show step-by-step reasoning
        "because",  # Should include justifications
        "consider",  # Should consider alternatives
    ],
    "forbidden_patterns": [
        "I cannot help with that",  # Should provide reasoning, not just refuse
        "I don't know",  # Should attempt reasoning
    ]
}

# Output Configuration  
OUTPUT_CONFIG = {
    "format": "jsonl",
    "fields": [
        "problem",
        "scratchpad_response", 
        "extracted_scratchpad",
        "extracted_answer",
        "model",
        "category",
        "generation_config",
        "quality_score",
        "timestamp"
    ]
}

def get_config_for_category(category: str) -> Dict[str, Any]:
    """Get generation configuration for a specific problem category."""
    return GENERATION_CONFIG.get(category, GENERATION_CONFIG["general"])