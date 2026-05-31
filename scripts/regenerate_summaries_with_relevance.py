#!/usr/bin/env python3
"""
Regenerate evaluation summaries to include relevance scores.
This script updates all eval_summary.json files to include relevance statistics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    results = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return results

def compute_summary_with_relevance(results: List[Dict], harmfulness_mode: bool = False) -> Dict:
    """Compute evaluation summary statistics including relevance."""
    summary = {"total_comparisons": len(results)}
    
    # Relevance stats
    relevance_results = [r.get("relevance", {}) for r in results if r.get("relevance")]
    if relevance_results:
        # Count relevance outcomes: p1, p2, equal
        p1_more_relevant = sum(1 for r in relevance_results if r.get("reasoning", "").find("<more_relevant_to>p1</more_relevant_to>") != -1)
        p2_more_relevant = sum(1 for r in relevance_results if r.get("reasoning", "").find("<more_relevant_to>p2</more_relevant_to>") != -1)
        equal_relevance = sum(1 for r in relevance_results if r.get("reasoning", "").find("<more_relevant_to>equal</more_relevant_to>") != -1)
        
        summary["relevance"] = {
            "total": len(relevance_results),
            "p1_more_relevant": p1_more_relevant,
            "p2_more_relevant": p2_more_relevant,
            "equal_relevance": equal_relevance,
            "p1_relevance_rate": p1_more_relevant / len(relevance_results) if relevance_results else 0,
            "p2_relevance_rate": p2_more_relevant / len(relevance_results) if relevance_results else 0,
            "equal_relevance_rate": equal_relevance / len(relevance_results) if relevance_results else 0,
        }
    
    # Helpfulness stats
    helpfulness_results = [r.get("helpfulness", {}) for r in results if r.get("helpfulness")]
    if helpfulness_results:
        o1_wins = sum(1 for r in helpfulness_results if r.get("winner") == "O1")
        o2_wins = sum(1 for r in helpfulness_results if r.get("winner") == "O2") 
        ties = sum(1 for r in helpfulness_results if r.get("winner") == "Tie")
        
        summary["helpfulness"] = {
            "total": len(helpfulness_results),
            "o1_wins": o1_wins,
            "o2_wins": o2_wins,
            "ties": ties,
            "o1_win_rate": o1_wins / len(helpfulness_results) if helpfulness_results else 0,
            "o2_win_rate": o2_wins / len(helpfulness_results) if helpfulness_results else 0,
            "tie_rate": ties / len(helpfulness_results) if helpfulness_results else 0,
        }
    
    # Harmfulness/harmlessness stats
    harm_key = "harmfulness" if harmfulness_mode else "harmlessness"
    harm_results = [r.get(harm_key, {}) for r in results if r.get(harm_key)]
    if harm_results:
        o1_wins = sum(1 for r in harm_results if r.get("winner") == "O1")
        o2_wins = sum(1 for r in harm_results if r.get("winner") == "O2")
        ties = sum(1 for r in harm_results if r.get("winner") == "Tie")
        
        summary[harm_key] = {
            "total": len(harm_results),
            "o1_wins": o1_wins,
            "o2_wins": o2_wins, 
            "ties": ties,
            "o1_win_rate": o1_wins / len(harm_results) if harm_results else 0,
            "o2_win_rate": o2_wins / len(harm_results) if harm_results else 0,
            "tie_rate": ties / len(harm_results) if harm_results else 0,
        }
        
        if harmfulness_mode:
            summary[harm_key]["o1_less_harmful"] = o2_wins
            summary[harm_key]["o2_less_harmful"] = o1_wins
        else:
            summary[harm_key]["o1_more_harmless"] = o1_wins
            summary[harm_key]["o2_more_harmless"] = o2_wins
    
    return summary

def main():
    # Find all evaluation directories
    root_dir = Path(__file__).parent.parent
    eval_dir = root_dir / 'output' / 'DNA_New_Experiments' / 'evaluation'
    
    if not eval_dir.exists():
        print(f"Evaluation directory not found: {eval_dir}")
        return
    
    # Find all directories with eval_results.jsonl
    eval_dirs = []
    for subdir in eval_dir.iterdir():
        if subdir.is_dir():
            results_file = subdir / "eval_results.jsonl"
            summary_file = subdir / "eval_summary.json"
            if results_file.exists():
                eval_dirs.append((subdir, results_file, summary_file))
    
    if not eval_dirs:
        print("No evaluation result files found!")
        return
    
    print(f"Found {len(eval_dirs)} evaluation directories to update\n")
    
    # Update each summary
    updated_count = 0
    for eval_dir_path, results_file, summary_file in eval_dirs:
        print(f"Processing: {eval_dir_path.name}")
        
        # Load results
        results = load_jsonl(str(results_file))
        if not results:
            print(f"  ⚠ No results found in {results_file}")
            continue
        
        # Check if relevance data exists
        relevance_count = sum(1 for r in results if r.get("relevance"))
        if relevance_count == 0:
            print(f"  ⚠ No relevance data found, skipping")
            continue
        
        # Generate new summary with relevance
        summary = compute_summary_with_relevance(results, harmfulness_mode=False)
        
        # Write updated summary
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Updated summary with relevance stats ({relevance_count} relevance evaluations)")
        updated_count += 1
    
    print(f"\n{'='*60}")
    print(f"Summary regeneration complete!")
    print(f"Updated {updated_count} evaluation summaries with relevance scores.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()