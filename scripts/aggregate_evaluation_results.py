#!/usr/bin/env python3
"""
Aggregate DNA evaluation results from all summary JSON files.
Generates a comprehensive overview of all comparisons across models.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

def load_summary(file_path: str) -> Dict:
    """Load evaluation summary JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def extract_model_name(file_path: str) -> str:
    """Extract model name from file path."""
    path_parts = Path(file_path).parts
    for part in path_parts:
        if part.endswith('_gemma4_judge'):
            # Extract model from part like 'baseline_vs_benign_intent_llama8b_gemma4_judge'
            parts = part.split('_')
            if 'llama8b' in part:
                return 'llama8b'
            elif 'llama33_70b' in part:
                return 'llama33_70b'  
            elif 'mistral_7b' in part:
                return 'mistral_7b'
            elif 'mistral_24b' in part:
                return 'mistral_24b'
            elif 'qwen35_9b' in part:
                return 'qwen35_9b'
            elif 'qwen35_27b' in part:
                return 'qwen35_27b'
    return 'unknown'

def extract_comparison_type(file_path: str) -> str:
    """Extract comparison type from file path."""
    if 'baseline_vs_benign_intent_guidelines' in file_path:
        return 'baseline_vs_benign_intent_guidelines'
    elif 'baseline_vs_benign_intent' in file_path:
        return 'baseline_vs_benign_intent'
    elif 'baseline_vs_refinement' in file_path:
        return 'baseline_vs_refinement'
    return 'unknown'

def main():
    # Find all evaluation summary files
    root_dir = Path(__file__).parent.parent
    eval_dir = root_dir / 'output' / 'DNA_New_Experiments' / 'evaluation'
    
    if not eval_dir.exists():
        print(f"Evaluation directory not found: {eval_dir}")
        return
    
    summary_files = list(eval_dir.glob('*/eval_summary.json'))
    
    if not summary_files:
        print("No evaluation summary files found!")
        return
        
    print(f"Found {len(summary_files)} evaluation summary files\n")
    
    # Organize results
    results = {}
    models = ['llama8b', 'llama33_70b', 'mistral_7b', 'mistral_24b', 'qwen35_9b', 'qwen35_27b']
    comparisons = ['baseline_vs_benign_intent', 'baseline_vs_benign_intent_guidelines', 'baseline_vs_refinement']
    
    # Initialize structure
    for model in models:
        results[model] = {}
        for comp in comparisons:
            results[model][comp] = None
    
    # Load all results
    for file_path in summary_files:
        model = extract_model_name(str(file_path))
        comparison = extract_comparison_type(str(file_path))
        data = load_summary(str(file_path))
        
        if model in results and comparison in results[model]:
            results[model][comparison] = data
            print(f"✓ Loaded: {model} - {comparison}")
        else:
            print(f"⚠ Unexpected: {model} - {comparison}")
    
    print(f"\n{'='*80}")
    print("DNA EVALUATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Generate summary tables
    for comparison in comparisons:
        print(f"\n🔍 {comparison.replace('_', ' ').title()}")
        print(f"{'-'*60}")
        print(f"{'Model':<15} {'O1 Win%':<8} {'O2 Win%':<8} {'Tie%':<8} {'P1 Rel%':<8} {'P2 Rel%':<8} {'Harmless Tie%':<12}")
        print(f"{'-'*80}")
        
        for model in models:
            data = results[model].get(comparison)
            if data and 'helpfulness' in data:
                help_data = data['helpfulness']
                harm_data = data.get('harmlessness', {})
                rel_data = data.get('relevance', {})
                
                o1_rate = help_data.get('o1_win_rate', 0) * 100
                o2_rate = help_data.get('o2_win_rate', 0) * 100
                tie_rate = help_data.get('tie_rate', 0) * 100
                harm_tie_rate = harm_data.get('tie_rate', 0) * 100
                p1_rel_rate = rel_data.get('p1_relevance_rate', 0) * 100
                p2_rel_rate = rel_data.get('p2_relevance_rate', 0) * 100
                
                print(f"{model:<15} {o1_rate:<8.1f} {o2_rate:<8.1f} {tie_rate:<8.1f} {p1_rel_rate:<8.1f} {p2_rel_rate:<8.1f} {harm_tie_rate:<12.1f}")
            else:
                print(f"{model:<15} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<12}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("TREATMENT EFFECTIVENESS SUMMARY")
    print(f"{'='*80}")
    
    treatment_scores = {}
    for comparison in comparisons:
        treatment_name = comparison.replace('baseline_vs_', '').replace('_', ' ').title()
        scores = []
        
        for model in models:
            data = results[model].get(comparison)
            if data and 'helpfulness' in data:
                o2_rate = data['helpfulness'].get('o2_win_rate', 0) * 100
                scores.append(o2_rate)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            treatment_scores[treatment_name] = avg_score
            
            print(f"\n{treatment_name}:")
            print(f"  Average O2 Win Rate: {avg_score:.1f}%")
            print(f"  Range: {min_score:.1f}% - {max_score:.1f}%")
    
    # Ranking
    if treatment_scores:
        print(f"\n🏆 TREATMENT RANKING (by average O2 win rate):")
        sorted_treatments = sorted(treatment_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (treatment, score) in enumerate(sorted_treatments, 1):
            print(f"  {i}. {treatment}: {score:.1f}%")
    
    print(f"\n{'='*80}")
    print(f"Analysis complete! Results aggregated from {len([f for f in summary_files if os.path.exists(f)])} evaluations.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()