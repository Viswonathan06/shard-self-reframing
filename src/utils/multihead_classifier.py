"""
Lightweight multi-head classifier for intent and category classification.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json
from typing import Dict, Optional

# Category mapping
CATEGORY_MAP = {
    "Explicit Content": 0,
    "Crimes & Illegal Activities": 1,
    "Harm & Misuse": 2,
    "Fairness & Justice": 3,
    "Privacy & Property": 4
}

CATEGORY_NAMES = list(CATEGORY_MAP.keys())
REVERSE_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}


class MultiHeadClassifier(nn.Module):
    """Multi-head classifier with shared encoder and two heads."""
    
    def __init__(self, model_name: str, num_intent_classes: int = 4, num_category_classes: int = 5):
        super().__init__()
        # Shared encoder
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_intent_classes,
            return_dict=True
        )
        
        hidden_size = self.encoder.config.hidden_size
        self.encoder.classifier = nn.Identity()
        
        # Two separate heads
        self.intent_head = nn.Linear(hidden_size, num_intent_classes)
        self.category_head = nn.Linear(hidden_size, num_category_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        intent_logits = self.intent_head(pooled_output)
        category_logits = self.category_head(pooled_output)
        
        return {
            'intent_logits': intent_logits,
            'category_logits': category_logits
        }


class ClassifierInference:
    """Wrapper for inference with the multi-head classifier."""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Load trained classifier.
        
        Args:
            model_path: Path to saved model directory
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load config to get base model name + head dims
        config_path = Path(model_path) / "config.json"
        num_intent_classes = 4
        num_category_classes = 5
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                base_model = config.get("_name_or_path", "distilbert-base-uncased")
                num_intent_classes = int(config.get("num_intent_classes", num_intent_classes))
                num_category_classes = int(config.get("num_category_classes", num_category_classes))
        else:
            base_model = "distilbert-base-uncased"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model using from_pretrained (handles both .bin and safetensors)
        try:
            # Try loading as a full model first (if saved with save_pretrained)
            self.model = MultiHeadClassifier(
                base_model,
                num_intent_classes=num_intent_classes,
                num_category_classes=num_category_classes,
            )
            
            # Try to load state dict
            model_file = Path(model_path) / "pytorch_model.bin"
            if not model_file.exists():
                # Check for safetensors
                import glob
                safetensors_files = glob.glob(str(Path(model_path) / "*.safetensors"))
                if safetensors_files:
                    # Would need safetensors library - for now, try loading from checkpoint
                    checkpoint_dir = Path(model_path)
                    # Try loading from training checkpoint
                    checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"), reverse=True)
                    if checkpoints:
                        model_file = checkpoints[0] / "pytorch_model.bin"
            
            if model_file.exists():
                state_dict = torch.load(model_file, map_location=device)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                print(f"Warning: Model weights not found, using randomly initialized model")
            
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("  Using randomly initialized model")
        
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ Loaded classifier from {model_path} on {device}")
    
    def classify(self, text: str) -> Dict:
        """
        Classify a single text using both intent and category heads.
        
        Intent is predicted as a 4-way severity/level classification:
        - Level 0, 1 = safer
        - Level 2, 3 = more harmful
        Category: One of 5 harm categories
        
        Returns:
            {
                'intent_level': 0|1|2|3,
                'intent_level_confidence': float,
                'intent': 'harmful' or 'safe',  # derived (level >= 2)
                'intent_confidence': float,     # derived prob (harmful vs safe)
                'category': str (category name),
                'category_confidence': float
            }
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Intent prediction (separate head)
            intent_logits = outputs['intent_logits']
            intent_probs = torch.softmax(intent_logits, dim=-1)
            intent_level_pred = int(intent_logits.argmax(dim=-1).item())
            intent_level_confidence = float(intent_probs[0][intent_level_pred].item())

            # Derived binary intent (safe vs harmful) for backward compatibility
            safe_prob = float(intent_probs[0][:2].sum().item()) if intent_probs.shape[-1] >= 2 else 0.0
            harmful_prob = float(intent_probs[0][2:].sum().item()) if intent_probs.shape[-1] > 2 else 0.0
            intent_binary = "harmful" if intent_level_pred >= 2 else "safe"
            intent_binary_conf = harmful_prob if intent_binary == "harmful" else safe_prob
            
            # Category prediction (separate head)
            category_logits = outputs['category_logits']
            category_probs = torch.softmax(category_logits, dim=-1)
            category_pred = category_logits.argmax(dim=-1).item()
            category_confidence = category_probs[0][category_pred].item()
        
        return {
            'intent_level': intent_level_pred,
            'intent_level_confidence': intent_level_confidence,
            'intent': intent_binary,
            'intent_confidence': intent_binary_conf,
            'category': REVERSE_CATEGORY_MAP.get(category_pred, 'Unknown'),
            'category_confidence': category_confidence
        }
    
    def classify_batch(self, texts: list) -> list:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]


def get_classifier(model_path: str = "models/multihead_classifier", 
                   device: Optional[str] = None) -> ClassifierInference:
    """
    Get or create classifier instance.
    
    Args:
        model_path: Path to saved model
        device: Device to use
    
    Returns:
        ClassifierInference instance
    """
    return ClassifierInference(model_path, device)

