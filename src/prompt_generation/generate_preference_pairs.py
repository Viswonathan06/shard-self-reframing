#!/usr/bin/env python3
"""
Generate preference pairs from P1 and outputs for rating.
Each pair contains P1, output1, output2, and a preference field.
Supports multiple pairing strategies: tournament (balanced), round-robin (all pairs), or random sampling.
"""

import json
import sys
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def load_outputs(output_file):
    """Load all outputs from JSONL file and group by p1_id."""
    outputs_by_p1 = defaultdict(list)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            p1_id = rec.get('p1_id')
            if p1_id:
                outputs_by_p1[p1_id].append(rec)
    
    return outputs_by_p1


def generate_tournament_pairs(outputs_sorted, max_pairs_per_p1=None):
    """
    Generate pairs using a tournament-style approach (like chess Swiss system).
    Ensures each output appears roughly the same number of times.
    
    Strategy:
    1. Pair outputs in rounds (similar to tournament brackets)
    2. Each output appears in roughly equal number of comparisons
    3. Reduces from n*(n-1)/2 pairs to approximately n*rounds/2 pairs per round
    """
    n = len(outputs_sorted)
    if n < 2:
        return []
    
    pairs = []
    # Track which pairs have already been created
    seen_pairs = set()
    # Track how many times each output has been paired
    pair_counts = [0] * n
    
    # Calculate target number of rounds
    if max_pairs_per_p1 is None:
        # Default: aim for ~5-7 comparisons per output (good coverage without being exhaustive)
        target_comparisons_per_output = min(7, n - 1)
        num_rounds = target_comparisons_per_output
    else:
        # Calculate rounds needed to reach max_pairs
        pairs_per_round = n // 2
        num_rounds = max(1, (max_pairs_per_p1 + pairs_per_round - 1) // pairs_per_round)
    
    for round_num in range(num_rounds):
        # Create pairs ensuring balanced coverage
        used = set()
        round_pairs = []
        
        # Sort indices by pair count (ascending) to balance comparisons
        indices = list(range(n))
        indices.sort(key=lambda i: pair_counts[i])
        
        for i in indices:
            if i in used:
                continue
            
            # Find best partner: lowest pair count, not used, and pair not seen before
            best_j = None
            best_score = float('inf')
            
            for j in range(n):
                if j == i or j in used:
                    continue
                if (i, j) in seen_pairs or (j, i) in seen_pairs:
                    continue
                
                # Score: prioritize lower pair counts
                score = pair_counts[j]
                if score < best_score:
                    best_score = score
                    best_j = j
            
            if best_j is not None:
                round_pairs.append((i, best_j))
                used.add(i)
                used.add(best_j)
                seen_pairs.add((i, best_j))
                pair_counts[i] += 1
                pair_counts[best_j] += 1
        
        pairs.extend(round_pairs)
        
        # If we've exhausted all possible pairs, break
        if len(seen_pairs) >= n * (n - 1) // 2:
            break
    
    # If max_pairs specified and we have more, randomly sample
    if max_pairs_per_p1 is not None and len(pairs) > max_pairs_per_p1:
        pairs = random.sample(pairs, max_pairs_per_p1)
    
    return pairs


def generate_preference_pairs(outputs_by_p1, strategy='tournament', max_pairs_per_p1=None, random_seed=42):
    """
    Generate preference pairs: (P1, output1, output2) with preference field.
    
    Args:
        outputs_by_p1: Dictionary mapping p1_id to list of outputs
        strategy: 'tournament' (balanced, like chess), 'round-robin' (all pairs), or 'random' (sample)
        max_pairs_per_p1: Maximum pairs per P1 (for random strategy or to cap tournament)
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        random.seed(random_seed)
    
    all_pairs = []
    
    for p1_id, outputs in outputs_by_p1.items():
        # Sort by pprime_idx to ensure consistent ordering
        outputs_sorted = sorted(outputs, key=lambda x: x.get('pprime_idx', 0))
        n = len(outputs_sorted)
        
        if n < 2:
            continue
        
        # Generate pair indices based on strategy
        if strategy == 'round-robin':
            # All possible pairs
            pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        elif strategy == 'random':
            # Random sampling
            all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
            max_pairs = max_pairs_per_p1 if max_pairs_per_p1 else min(10, len(all_possible))
            pair_indices = random.sample(all_possible, min(max_pairs, len(all_possible)))
        else:  # tournament (default)
            pair_indices = generate_tournament_pairs(outputs_sorted, max_pairs_per_p1)
        
        # Create pair records
        for i, j in pair_indices:
            output1 = outputs_sorted[i]
            output2 = outputs_sorted[j]
            
            pair = {
                'p1_id': p1_id,
                'p1': output1.get('p1', ''),
                'p2': output1.get('p2', ''),  # May be None for safe prompts
                'output1': {
                    'pprime_idx': output1.get('pprime_idx'),
                    'output': output1.get('output', ''),
                    'conversion_skipped': output1.get('conversion_skipped', False)
                },
                'output2': {
                    'pprime_idx': output2.get('pprime_idx'),
                    'output': output2.get('output', ''),
                    'conversion_skipped': output2.get('conversion_skipped', False)
                },
                'preference': None,  # Field for rating: 1 (prefer output1) or 2 (prefer output2)
                'metadata': {
                    'id': output1.get('id'),
                    'lang': output1.get('lang', 'en'),
                    'level': output1.get('level'),
                    'subtype': output1.get('subtype', ''),
                    'type': output1.get('type', ''),
                    'source': output1.get('source', '')
                }
            }
            all_pairs.append(pair)
    
    return all_pairs


def save_pairs(pairs, output_file, format='auto'):
    """
    Save pairs to file.
    
    Args:
        pairs: List of pair dictionaries
        output_file: Output file path
        format: 'json', 'jsonl', or 'auto' (detect from file extension)
    """
    if format == 'auto':
        # Detect format from file extension
        if output_file.endswith('.json'):
            format = 'json'
        else:
            format = 'jsonl'
    
    if format == 'json':
        # Save as JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)
    else:
        # Save as JSONL (one JSON object per line)
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate preference pairs for rating')
    parser.add_argument('input_file', help='Input JSONL file with outputs')
    parser.add_argument('output_file', help='Output JSONL file for preference pairs')
    parser.add_argument('--strategy', choices=['tournament', 'round-robin', 'random'], 
                       default='tournament',
                       help='Pairing strategy: tournament (balanced, like chess), round-robin (all pairs), or random (sample)')
    parser.add_argument('--max-pairs-per-p1', type=int, default=None,
                       help='Maximum pairs per P1 (for random strategy or to cap tournament)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--format', choices=['json', 'jsonl', 'auto'], default='auto',
                       help='Output format: json (array), jsonl (lines), or auto (detect from extension)')
    
    args = parser.parse_args()
    
    print(f"Loading outputs from: {args.input_file}")
    outputs_by_p1 = load_outputs(args.input_file)
    
    print(f"Found {len(outputs_by_p1)} unique P1 prompts")
    total_outputs = sum(len(outputs) for outputs in outputs_by_p1.values())
    print(f"Total outputs: {total_outputs}")
    
    print(f"Generating preference pairs using '{args.strategy}' strategy...")
    pairs = generate_preference_pairs(
        outputs_by_p1, 
        strategy=args.strategy,
        max_pairs_per_p1=args.max_pairs_per_p1,
        random_seed=args.seed
    )
    
    print(f"Generated {len(pairs)} preference pairs")
    
    # Calculate statistics
    pairs_per_p1 = len(pairs) / len(outputs_by_p1) if outputs_by_p1 else 0
    print(f"Average pairs per P1: {pairs_per_p1:.1f}")
    
    # Compare to round-robin
    if args.strategy != 'round-robin':
        total_possible = sum(len(outputs) * (len(outputs) - 1) // 2 for outputs in outputs_by_p1.values())
        reduction = (1 - len(pairs) / total_possible) * 100 if total_possible > 0 else 0
        print(f"Reduction from round-robin: {reduction:.1f}% ({len(pairs)} vs {total_possible} possible pairs)")
    
    print(f"Saving preference pairs to: {args.output_file}")
    save_pairs(pairs, args.output_file, format=args.format)
    
    print("Done!")
    print(f"\nPreference field can be set to:")
    print("  - 1: Prefer output1")
    print("  - 2: Prefer output2")
    print("  - None: Not yet rated")


if __name__ == '__main__':
    main()

