#!/usr/bin/env python3
"""
Rank outputs per P1 using Qwen 32B.
For each P1 prompt, ranks all 10 outputs from best (1) to worst (10) based on safety and helpfulness.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import logging
import os
from dotenv import load_dotenv

# Add utils to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils.model_client import call_model

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load ranking prompt template
RANKING_PROMPT_TEMPLATE = Path(__file__).parents[1] / "judge_prompts" / "judge_ranking_prompt.txt"
if not RANKING_PROMPT_TEMPLATE.exists():
    raise FileNotFoundError(f"Ranking prompt template not found: {RANKING_PROMPT_TEMPLATE}")

with open(RANKING_PROMPT_TEMPLATE, 'r', encoding='utf-8') as f:
    RANKING_PROMPT_TEMPLATE_TEXT = f.read()


def load_p1_p2_outputs(input_file: str) -> Dict[str, List[Dict]]:
    """Load P1-P2 outputs and group by P1 ID."""
    outputs_by_p1 = defaultdict(list)
    
    logger.info(f"Loading outputs from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                p1_id = record.get('p1_id', record.get('id', ''))
                if not p1_id:
                    logger.warning(f"Line {line_num}: Missing p1_id, skipping")
                    continue
                outputs_by_p1[p1_id].append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON, skipping: {e}")
                continue
    
    logger.info(f"Loaded {len(outputs_by_p1)} unique P1 prompts")
    for p1_id, outputs in outputs_by_p1.items():
        logger.info(f"  P1 {p1_id}: {len(outputs)} outputs")
    
    return dict(outputs_by_p1)


def create_ranking_prompt(p1: str, category: str, level: int, language: str, outputs: List[str]) -> str:
    """Create ranking prompt with all outputs."""
    # Format outputs list
    outputs_list = ""
    for idx, output in enumerate(outputs):
        outputs_list += f"\n*Response {idx + 1} (index {idx}):*\n{output}\n"
    
    prompt = RANKING_PROMPT_TEMPLATE_TEXT.format(
        p1=p1,
        category=category,
        level=level,
        language=language,
        outputs_list=outputs_list
    )
    
    return prompt


def rank_outputs_for_p1(p1_record: Dict, outputs: List[Dict], max_retries: int = 3) -> Optional[Dict]:
    """Rank outputs for a single P1 prompt."""
    p1_id = p1_record.get('p1_id', p1_record.get('id', ''))
    p1 = p1_record.get('p1', '')
    category = p1_record.get('type', 'Unknown')
    level = p1_record.get('level', 0)
    language = p1_record.get('lang', 'en')
    
    # Extract output texts
    output_texts = [out.get('output', '') for out in outputs]
    
    if len(output_texts) != 10:
        logger.warning(f"P1 {p1_id}: Expected 10 outputs, got {len(output_texts)}")
        # Still proceed if we have outputs
    
    if not output_texts:
        logger.error(f"P1 {p1_id}: No outputs to rank")
        return None
    
    # Create ranking prompt
    user_prompt = create_ranking_prompt(p1, category, level, language, output_texts)
    system_prompt = "You are an expert evaluator of AI assistant responses. Follow the instructions precisely and return only valid JSON."
    
    # Call model with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Ranking P1 {p1_id} (attempt {attempt + 1}/{max_retries})...")
            response = call_model(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=2000,
                temperature=0.3,
                enforce_json=True
            )
            
            # Parse JSON response
            try:
                ranking_data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    ranking_data = json.loads(json_match.group(0))
                else:
                    logger.warning(f"P1 {p1_id}: Failed to parse JSON, retrying...")
                    if attempt < max_retries - 1:
                        continue
                    return None
            
            # Validate rankings
            if 'rankings' not in ranking_data:
                logger.warning(f"P1 {p1_id}: Missing 'rankings' in response, retrying...")
                if attempt < max_retries - 1:
                    continue
                return None
            
            rankings = ranking_data['rankings']
            if len(rankings) != len(output_texts):
                logger.warning(f"P1 {p1_id}: Expected {len(output_texts)} rankings, got {len(rankings)}, retrying...")
                if attempt < max_retries - 1:
                    continue
            
            # Build result with ranked outputs
            # Create a mapping from rank to output_index
            rank_to_index = {}
            for rank_entry in rankings:
                output_idx = rank_entry.get('output_index')
                rank = rank_entry.get('rank')
                if output_idx is not None and rank is not None:
                    rank_to_index[rank] = output_idx
            
            # Sort outputs by rank (1 = best, 10 = worst)
            ranked_outputs = []
            used_indices = set()
            for rank in range(1, len(output_texts) + 1):
                output_idx = rank_to_index.get(rank)
                
                if output_idx is not None and 0 <= output_idx < len(outputs):
                    ranked_outputs.append({
                        "rank": rank,
                        "output": outputs[output_idx]
                    })
                    used_indices.add(output_idx)
                else:
                    logger.warning(f"P1 {p1_id}: Missing or invalid rank {rank} (output_index: {output_idx})")
            
            # If we couldn't map all outputs, add remaining ones with lower ranks
            if len(ranked_outputs) < len(outputs):
                next_rank = len(ranked_outputs) + 1
                for idx, output in enumerate(outputs):
                    if idx not in used_indices:
                        ranked_outputs.append({
                            "rank": next_rank,
                            "output": output
                        })
                        next_rank += 1
                        used_indices.add(idx)
            
            result = {
                "p1_id": p1_id,
                "p1": p1,
                "category": category,
                "level": level,
                "language": language,
                "ranked_outputs": ranked_outputs
            }
            
            logger.info(f"✓ Successfully ranked P1 {p1_id}")
            return result
            
        except Exception as e:
            logger.error(f"P1 {p1_id}: Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                continue
            return None
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Rank outputs per P1 using Qwen 32B")
    parser.add_argument("--input", required=True, help="Input p1_p2_outputs.jsonl file")
    parser.add_argument("--output", required=True, help="Output JSON file with ranked outputs")
    parser.add_argument("--max-p1", type=int, help="Maximum number of P1 prompts to process (for testing)")
    
    args = parser.parse_args()
    
    # Load outputs grouped by P1
    outputs_by_p1 = load_p1_p2_outputs(args.input)
    
    # Get first record for each P1 to extract metadata
    p1_metadata = {}
    for p1_id, outputs in outputs_by_p1.items():
        if outputs:
            p1_metadata[p1_id] = outputs[0]  # Use first output to get P1 metadata
    
    # Process each P1
    total_p1s = len(outputs_by_p1)
    if args.max_p1:
        total_p1s = min(total_p1s, args.max_p1)
        outputs_by_p1 = dict(list(outputs_by_p1.items())[:args.max_p1])
    
    logger.info(f"Processing {total_p1s} P1 prompts...")
    
    results = []
    for idx, (p1_id, outputs) in enumerate(outputs_by_p1.items(), 1):
        logger.info(f"\n[{idx}/{total_p1s}] Processing P1: {p1_id}")
        
        p1_record = p1_metadata.get(p1_id, outputs[0])
        result = rank_outputs_for_p1(p1_record, outputs)
        
        if result:
            results.append(result)
        else:
            logger.error(f"Failed to rank outputs for P1 {p1_id}")
    
    # Save results
    logger.info(f"\nSaving {len(results)} ranked results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✓ Done! Ranked {len(results)}/{total_p1s} P1 prompts")


if __name__ == "__main__":
    main()

