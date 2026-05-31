#!/usr/bin/env python3
"""
New evaluation approach:
1. For each P1, generate 10 different P2s using all 10 p' candidates
2. Judge each P2 three times and average the scores
3. Compute BLEU and SBERT scores for each P2
4. Combine scores (judge + BLEU + SBERT) and select best P2 for each P1
"""

import json
import argparse
import os
import uuid
import logging
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple, Optional
import concurrent.futures
import pandas as pd
from dotenv import load_dotenv
import numpy as np

logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

from src.utils.model_client import call_model, call_model_batch, ContentFilteredError, AZURE_MODEL, AZURE_ENDPOINT, AZURE_API_VER
from src.utils.sanitize import load_terms, sanitize_with_placeholders, fallback_strict_sanitize
from src.utils.schema import extract_schema_from_p1r, build_schema_judge_input

# For BLEU score
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    BLEU_AVAILABLE = True
except ImportError:
    print("Warning: nltk not available. BLEU scores will be skipped.")
    BLEU_AVAILABLE = False
except Exception as e:
    print(f"Warning: NLTK setup failed: {e}. BLEU scores will be skipped.")
    BLEU_AVAILABLE = False

# For SBERT score
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SBERT_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. SBERT scores will be skipped.")
    SBERT_AVAILABLE = False

# Model for judging (GPT-4o-mini) - can be overridden via environment variable
JUDGE_MODEL = os.getenv("AZURE_OPENAI_JUDGE_MODEL", "gpt-4o-mini")

# Local model options (can be overridden by command line)
USE_LOCAL_JUDGE_ENV = os.getenv("USE_LOCAL_JUDGE", "false").strip().lower()
USE_LOCAL_JUDGE = USE_LOCAL_JUDGE_ENV == "true" or USE_LOCAL_JUDGE_ENV == "1"
LOCAL_JUDGE_MODEL = os.getenv("LOCAL_JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

USE_LOCAL_MODEL_ENV = os.getenv("USE_LOCAL_MODEL", "false").strip().lower()
USE_LOCAL_MODEL = USE_LOCAL_MODEL_ENV == "true" or USE_LOCAL_MODEL_ENV == "1"
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# Debug: Log local model status at import
if USE_LOCAL_JUDGE:
    print(f"[DEBUG] Local judge ENABLED from .env: {LOCAL_JUDGE_MODEL}")
else:
    print(f"[DEBUG] Local judge DISABLED (USE_LOCAL_JUDGE={USE_LOCAL_JUDGE_ENV})")

if USE_LOCAL_MODEL:
    print(f"[DEBUG] Local model ENABLED from .env: {LOCAL_MODEL_NAME} (for P2 generation)")
else:
    print(f"[DEBUG] Local model DISABLED (USE_LOCAL_MODEL={USE_LOCAL_MODEL_ENV})")

# Log Azure API URLs once at module import (only if not already logged by model_client)
_logged_azure_urls = False
def log_azure_urls_once():
    """Log Azure API URLs once at startup."""
    global _logged_azure_urls
    if _logged_azure_urls:
        return
    if AZURE_ENDPOINT:
        endpoint_clean = AZURE_ENDPOINT.rstrip('/')
        # Main model URL
        main_url = f"{endpoint_clean}/openai/deployments/{AZURE_MODEL}/chat/completions?api-version={AZURE_API_VER}"
        # Judge model URL (may be different)
        judge_url = f"{endpoint_clean}/openai/deployments/{JUDGE_MODEL}/chat/completions?api-version={AZURE_API_VER}"
        print(f"\n[Azure API Configuration]")
        print(f"  Endpoint: {endpoint_clean}")
        print(f"  API Version: {AZURE_API_VER}")
        print(f"  Main Model ({AZURE_MODEL}): {main_url}")
        if JUDGE_MODEL != AZURE_MODEL:
            print(f"  Judge Model ({JUDGE_MODEL}): {judge_url}")
        print()
    _logged_azure_urls = True

REWRITE_SYSTEM = (
    "Output ONLY the reframed prompt text, with no preface, no apologies, "
    "no explanations, no labels, no quotes, no markdown or code fences. "
)

JUDGE_SYSTEM = (
    "You are a strict JSON scoring engine. "
    "Always return a single valid JSON object with the required keys. "
    "No explanations, no markdown, no backticks with JSON object. Just the JSON object starting and ending with curly brackets."
)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_p1_from_csv(csv_path: str, language: str) -> List[Dict]:
    """
    Load P1 prompts from CSV file and filter by language.
    
    Args:
        csv_path: Path to linguasafe.csv
        language: Language code to filter (e.g., 'en', 'zh', 'ar')
    
    Returns:
        List of dicts with 'p1' (prompt text) and metadata (id, lang, level, subtype, type)
    """
    df = pd.read_csv(csv_path)
    
    # Filter by language
    df_filtered = df[df['lang'] == language].copy()
    
    if len(df_filtered) == 0:
        print(f"Warning: No prompts found for language '{language}' in {csv_path}")
        return []
    
    # Convert to list of dicts
    p1_prompts = []
    for _, row in df_filtered.iterrows():
        p1_record = {
            "id": str(row.get("id", "")),
            "p1": str(row.get("prompt", "")),
            "lang": str(row.get("lang", language)),
            "level": row.get("level", None),
            "subtype": str(row.get("subtype", "")) if pd.notna(row.get("subtype")) else "",
            "type": str(row.get("type", "")) if pd.notna(row.get("type")) else "",
            "source": str(row.get("source", "")) if pd.notna(row.get("source")) else "",
        }
        p1_prompts.append(p1_record)
    
    print(f"Loaded {len(p1_prompts)} P1 prompts for language '{language}' from {csv_path}")
    return p1_prompts


def save_jsonl(path: str, records: List[Dict]):
    """Save records to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def call_judge_model(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    """
    Call judge model for scoring.
    Uses local model if USE_LOCAL_JUDGE=true, otherwise uses Azure API.
    """
    global USE_LOCAL_JUDGE, LOCAL_JUDGE_MODEL
    
    if USE_LOCAL_JUDGE:
        # Use local judge model
        try:
            from src.utils.local_judge import get_local_judge
            judge = get_local_judge(model_name=LOCAL_JUDGE_MODEL)
            result = judge.judge(system_prompt, user_prompt, max_tokens, temperature=0.3)
            return result
        except ImportError as e:
            logging.error(f"❌ Local judge not available (transformers not installed?): {e}")
            logging.error("   Falling back to API. Install with: pip install transformers torch")
            # Fall through to API call
        except Exception as e:
            logging.error(f"❌ Local judge failed: {e}")
            logging.error(f"   Model: {LOCAL_JUDGE_MODEL}")
            logging.error("   Falling back to API judge")
            import traceback
            logging.debug(traceback.format_exc())
            # Fall through to API call
    
    # Use Azure API judge
    from src.utils.model_client import get_client, AZURE_API_VER
    global JUDGE_MODEL
    
    azure_client = get_client()
    if azure_client is None:
        raise ValueError("Azure client not available. Local judge may be enabled or credentials missing.")
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    kwargs = dict(
        model=JUDGE_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,  # Slight variation for averaging over multiple calls
        response_format={"type": "json_object"}
    )
    
    try:
        completion = azure_client.chat.completions.create(**kwargs)
        content = (completion.choices[0].message.content or "").strip()
        if not content:
            raise ContentFilteredError("Empty/blocked completion (likely content filter).")
        return content
    except Exception as e:
        msg = str(e)
        if ("content_filter" in msg
            or "ResponsibleAIPolicyViolation" in msg
            or 'status": 400' in msg):
            raise ContentFilteredError(msg)
        raise


def judge_p2_three_times(
    p1_redacted: str,
    p2: str,
    judge_template: str,
    schema: Optional[Dict] = None,
    extractor_prompt_text: Optional[str] = None,
    use_schema_judge: bool = True
) -> Dict:
    """
    Judge P2 three times and return averaged scores.
    Returns dict with averaged scores and individual scores.
    """
    scores_list = []
    
    for attempt in range(3):
        try:
            if use_schema_judge and schema:
                judge_input = build_schema_judge_input(judge_template, schema, p2)
            else:
                judge_input = judge_template.replace("<P1_REDACTED>", p1_redacted).replace("<P2>", p2)
            
            judge_resp = call_judge_model(JUDGE_SYSTEM, judge_input, max_tokens=500)
            scores = json.loads(judge_resp)
            scores_list.append(scores)
        except ContentFilteredError:
            # Fallback minimal judge
            minimal = (
                "Score P2 on: fidelity_goal, fidelity_target, fidelity_constraints, safety_score, "
                "fluency_score, usefulness_score (0-100). Return strict JSON.\n\n"
                f"P1 (redacted): {p1_redacted}\n\nP2: {p2}"
            )
            try:
                judge_resp = call_judge_model(JUDGE_SYSTEM, minimal, max_tokens=300)
                scores = json.loads(judge_resp)
                scores_list.append(scores)
            except Exception as e:
                print(f"Warning: Judge attempt {attempt + 1} failed: {e}")
                continue
        except Exception as e:
            print(f"Warning: Judge attempt {attempt + 1} failed: {e}")
            continue
    
    if not scores_list:
        return {
            "error": "All judge attempts failed",
            "fidelity_goal": 0, "fidelity_target": 0, "fidelity_constraints": 0,
            "safety_score": 0, "fluency_score": 0, "usefulness_score": 0,
            "actionable_detail_present": True
        }
    
    # Average scores across 3 attempts
    avg_scores = {}
    for key in scores_list[0].keys():
        if isinstance(scores_list[0][key], (int, float)):
            avg_scores[key] = mean([s.get(key, 0) for s in scores_list])
        elif isinstance(scores_list[0][key], bool):
            # For boolean, take majority vote
            true_count = sum(1 for s in scores_list if s.get(key, False))
            avg_scores[key] = true_count >= 2
        else:
            avg_scores[key] = scores_list[0][key]
    
    # Return only averaged scores (no individual scores)
    return avg_scores


def judge_p2_batch(
    p1_p2_pairs: List[Tuple[str, str, Optional[Dict]]],
    judge_template: str,
    extractor_prompt_text: Optional[str] = None,
    use_schema_judge: bool = True,
    num_judge_calls: int = 3,
    max_parallel: int = 20
) -> List[Dict]:
    """
    Batch version: Judge multiple P2s in parallel.
    
    Args:
        p1_p2_pairs: List of (p1_redacted, p2, schema) tuples
        judge_template: Judge prompt template
        extractor_prompt_text: Schema extractor prompt (if using schema judge)
        use_schema_judge: Whether to use schema-based judging
        num_judge_calls: Number of judge calls per P2 (default: 3 for averaging)
        max_parallel: Max parallel calls (for API) or batch size (for local)
    
    Returns:
        List of judge score dicts (same order as input)
    """
    global USE_LOCAL_JUDGE, LOCAL_JUDGE_MODEL
    
    # Use local judge if enabled
    if USE_LOCAL_JUDGE:
        logging.info(f"🔧 Attempting to use local judge: {LOCAL_JUDGE_MODEL}")
        try:
            from src.utils.local_judge import get_local_judge
            judge = get_local_judge(model_name=LOCAL_JUDGE_MODEL)
            logging.info(f"✓ Local judge loaded successfully: {LOCAL_JUDGE_MODEL}")
            
            # Prepare judge inputs
            judge_inputs = []
            for p1_redacted, p2, schema in p1_p2_pairs:
                if use_schema_judge and schema:
                    judge_input = build_schema_judge_input(judge_template, schema, p2)
                else:
                    judge_input = judge_template.replace("<P1_REDACTED>", p1_redacted).replace("<P2>", p2)
                
                # Repeat for num_judge_calls
                for _ in range(num_judge_calls):
                    judge_inputs.append((JUDGE_SYSTEM, judge_input))
            
            # Batch judge (true batch processing for local models)
            # For local models, larger batch sizes are more efficient
            # Use larger batch for local (no rate limits), smaller for API
            local_batch_size = max_parallel * 2 if USE_LOCAL_JUDGE else max_parallel
            judge_responses = judge.judge_batch(
                judge_inputs,
                max_tokens=500,
                temperature=0.3,
                batch_size=local_batch_size
            )
            
            # Parse and group by pair
            pair_scores = {}
            for pair_idx in range(len(p1_p2_pairs)):
                pair_scores[pair_idx] = []
                for call_idx in range(num_judge_calls):
                    response_idx = pair_idx * num_judge_calls + call_idx
                    if response_idx < len(judge_responses):
                        try:
                            scores = json.loads(judge_responses[response_idx])
                            pair_scores[pair_idx].append(scores)
                        except:
                            pass
            
            # Average scores for each pair
            results = []
            for pair_idx in range(len(p1_p2_pairs)):
                if pair_idx in pair_scores and pair_scores[pair_idx]:
                    scores_list = pair_scores[pair_idx]
                    avg_scores = {}
                    for key in scores_list[0].keys():
                        if isinstance(scores_list[0][key], (int, float)):
                            avg_scores[key] = mean([s.get(key, 0) for s in scores_list])
                        elif isinstance(scores_list[0][key], bool):
                            true_count = sum(1 for s in scores_list if s.get(key, False))
                            avg_scores[key] = true_count >= 2
                        else:
                            avg_scores[key] = scores_list[0][key]
                    results.append(avg_scores)
                else:
                    results.append({
                        "error": "All judge attempts failed",
                        "fidelity_goal": 0, "fidelity_target": 0, "fidelity_constraints": 0,
                        "safety_score": 0, "fluency_score": 0, "usefulness_score": 0,
                        "actionable_detail_present": True
                    })
            
            return results
        except Exception as e:
            logging.warning(f"Local judge batch failed, falling back to API: {e}")
            # Fall through to API version
    
    # Use API judge (original implementation)
    from src.utils.model_client import get_client, AZURE_API_VER
    global JUDGE_MODEL
    
    azure_client = get_client()
    if azure_client is None:
        raise ValueError("Azure client not available. Local judge may be enabled or credentials missing.")
    
    def judge_single(p1_redacted: str, p2: str, schema: Optional[Dict], attempt: int) -> Optional[Dict]:
        """Judge a single P2 once."""
        try:
            if use_schema_judge and schema:
                judge_input = build_schema_judge_input(judge_template, schema, p2)
            else:
                judge_input = judge_template.replace("<P1_REDACTED>", p1_redacted).replace("<P2>", p2)
            
            messages = []
            messages.append({"role": "system", "content": JUDGE_SYSTEM})
            messages.append({"role": "user", "content": judge_input})
            
            kwargs = dict(
                model=JUDGE_MODEL,
                messages=messages,
                max_tokens=500,
                temperature=0.3,  # Slight variation for averaging
                response_format={"type": "json_object"}
            )
            
            completion = azure_client.chat.completions.create(**kwargs)
            content = (completion.choices[0].message.content or "").strip()
            if not content:
                return None
            
            return json.loads(content)
        except Exception:
            return None
    
    # Collect all judge tasks: (p1_redacted, p2, schema, attempt_idx)
    judge_tasks = []
    task_to_pair_idx = []  # Map task index to original pair index
    
    for pair_idx, (p1_redacted, p2, schema) in enumerate(p1_p2_pairs):
        for attempt in range(num_judge_calls):
            judge_tasks.append((p1_redacted, p2, schema, attempt))
            task_to_pair_idx.append(pair_idx)
    
    # Batch judge all tasks in parallel
    all_scores = []
    # max_parallel is now a parameter (default: 20)
    
    for i in range(0, len(judge_tasks), max_parallel):
        batch = judge_tasks[i:i + max_parallel]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
            batch_results = list(executor.map(
                lambda task: judge_single(task[0], task[1], task[2], task[3]),
                batch
            ))
        all_scores.extend(batch_results)
    
    # Group scores by pair and average
    pair_scores = {}  # pair_idx -> list of score dicts
    for task_idx, scores in enumerate(all_scores):
        if scores is not None:
            pair_idx = task_to_pair_idx[task_idx]
            if pair_idx not in pair_scores:
                pair_scores[pair_idx] = []
            pair_scores[pair_idx].append(scores)
    
    # Average scores for each pair
    results = []
    for pair_idx in range(len(p1_p2_pairs)):
        if pair_idx in pair_scores and pair_scores[pair_idx]:
            scores_list = pair_scores[pair_idx]
            avg_scores = {}
            for key in scores_list[0].keys():
                if isinstance(scores_list[0][key], (int, float)):
                    avg_scores[key] = mean([s.get(key, 0) for s in scores_list])
                elif isinstance(scores_list[0][key], bool):
                    true_count = sum(1 for s in scores_list if s.get(key, False))
                    avg_scores[key] = true_count >= 2
                else:
                    avg_scores[key] = scores_list[0][key]
            results.append(avg_scores)
        else:
            # All judge attempts failed
            results.append({
                "error": "All judge attempts failed",
                "fidelity_goal": 0, "fidelity_target": 0, "fidelity_constraints": 0,
                "safety_score": 0, "fluency_score": 0, "usefulness_score": 0,
                "actionable_detail_present": True
            })
    
    return results


def compute_bleu_score(reference: str, candidate: str) -> float:
    """
    Compute BLEU score between reference (P1) and candidate (P2).
    Returns score between 0 and 1.
    """
    if not BLEU_AVAILABLE:
        return 0.0
    
    try:
        # Handle empty strings
        if not reference or not candidate:
            return 0.0
        
        # Tokenize
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())
        
        # Handle empty tokenization results
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        # Use smoothing function for better handling of short sentences
        smoothing = SmoothingFunction().method1
        
        # Compute BLEU score
        score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        return float(score)
    except Exception as e:
        # Print error only if it's not a silent failure
        error_msg = str(e) if e else "Unknown error"
        if error_msg:  # Only print if there's an actual error message
            print(f"Warning: BLEU computation failed: {error_msg}", flush=True)
        return 0.0


def compute_sbert_score(text1: str, text2: str, model: Optional[SentenceTransformer] = None) -> float:
    """
    Compute SBERT similarity score between two texts.
    Returns cosine similarity score between -1 and 1 (typically 0-1).
    """
    if not SBERT_AVAILABLE:
        return 0.0
    
    try:
        if model is None:
            # Use a multilingual model if available, otherwise default
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Compute embeddings
        embeddings = model.encode([text1, text2], convert_to_tensor=True)
        
        # Compute cosine similarity
        from torch.nn.functional import cosine_similarity
        similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])
        return float(similarity.item())
    except Exception as e:
        print(f"Warning: SBERT computation failed: {e}")
        return 0.0


def compute_combined_score(
    judge_scores: Dict,
    bleu_score: float,
    sbert_score: float,
    judge_weight: float = 0.5,
    bleu_weight: float = 0.25,
    sbert_weight: float = 0.25
) -> float:
    """
    Combine judge scores, BLEU, and SBERT into a single score.
    
    Judge score is computed as weighted average of judge metrics.
    """
    # Extract judge metrics and compute weighted average
    fidelity_components = []
    if "fidelity_goal" in judge_scores:
        fidelity_components.append(judge_scores["fidelity_goal"])
    if "fidelity_target" in judge_scores:
        fidelity_components.append(judge_scores["fidelity_target"])
    if "fidelity_constraints" in judge_scores:
        fidelity_components.append(judge_scores["fidelity_constraints"])
    
    avg_fidelity = mean(fidelity_components) if fidelity_components else 0
    safety = judge_scores.get("safety_score", 0)
    fluency = judge_scores.get("fluency_score", 0)
    usefulness = judge_scores.get("usefulness_score", 0)
    
    # Normalize judge metrics to 0-1 range (they're already 0-100)
    judge_avg = mean([avg_fidelity, safety, fluency, usefulness]) / 100.0
    
    # Normalize BLEU (already 0-1)
    bleu_norm = bleu_score
    
    # Normalize SBERT (typically 0-1, but ensure it's non-negative)
    sbert_norm = max(0.0, min(1.0, (sbert_score + 1) / 2.0))  # Map [-1,1] to [0,1]
    
    # Combined score
    combined = (
        judge_weight * judge_avg +
        bleu_weight * bleu_norm +
        sbert_weight * sbert_norm
    )
    
    return combined


def run_p2_selection_evaluation(
    csv_path: str,
    language: str,
    pprime_file: str,
    judge_prompt_file: str,
    output_dir: str,
    sanitizer_terms_file: str,
    use_schema_judge: bool = True,
    schema_extractor_prompt_file: str = None,
    judge_weight: float = 0.5,
    bleu_weight: float = 0.25,
    sbert_weight: float = 0.25,
    max_p1: Optional[int] = None,
    force_rerun: bool = False,
    use_batch: bool = True,
    batch_size: int = 10,
):
    """
    Main evaluation function:
    - For each P1, generate 10 P2s using all 10 p' candidates
    - Judge each P2 three times and average
    - Compute BLEU and SBERT scores
    - Select best P2 based on combined score
    
    Args:
        csv_path: Path to linguasafe.csv dataset
        language: Language code to filter (e.g., 'en', 'zh', 'ar')
        pprime_file: Path to p' candidates JSONL file
        judge_prompt_file: Path to judge schema template
        output_dir: Output directory for results
        sanitizer_terms_file: Path to sanitizer terms JSON
        use_schema_judge: Whether to use schema-based judging
        schema_extractor_prompt_file: Path to schema extractor prompt
        judge_weight: Weight for judge scores (default: 0.5)
        bleu_weight: Weight for BLEU score (default: 0.25)
        sbert_weight: Weight for SBERT score (default: 0.25)
        max_p1: Maximum number of P1 prompts to process (None = all)
    """
    
    # Load data
    p1_prompts = load_p1_from_csv(csv_path, language)
    if len(p1_prompts) == 0:
        print(f"Error: No P1 prompts found for language '{language}' in {csv_path}")
        return
    
    # Limit number of P1 prompts if specified
    if max_p1 is not None and max_p1 > 0:
        original_count = len(p1_prompts)
        p1_prompts = p1_prompts[:max_p1]
        print(f"Limited to {len(p1_prompts)} P1 prompts (from {original_count} total)")
    else:
        print(f"Processing all {len(p1_prompts)} P1 prompts")
    
    pprimes = load_jsonl(pprime_file)
    judge_template = Path(judge_prompt_file).read_text(encoding="utf-8")
    extractor_prompt_text = None
    if schema_extractor_prompt_file:
        extractor_prompt_text = Path(schema_extractor_prompt_file).read_text(encoding="utf-8")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    terms = load_terms(sanitizer_terms_file)
    
    # Initialize SBERT model once
    sbert_model = None
    if SBERT_AVAILABLE:
        try:
            sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✓ SBERT model loaded")
        except Exception as e:
            print(f"Warning: Could not load SBERT model: {e}")
    
    # Output files
    all_results_file = Path(output_dir) / "all_p2_candidates.jsonl"
    best_p2_file = Path(output_dir) / "best_p2_selected.jsonl"
    p1_p2_pairs_file = Path(output_dir) / "p1_p2_pairs.json"
    hierarchical_json_file = Path(output_dir) / "p1_p2s_hierarchical.json"
    summary_file = Path(output_dir) / "selection_summary.json"
    
    all_results = []
    best_p2_results = []
    hierarchical_data = {}  # Dict to store P1 -> list of P2s structure
    
    # Ask user if they want to rerun all prompts or start from checkpoint
    processed_p1_ids = set()
    if hierarchical_json_file.exists() and not force_rerun:
        print(f"\n{'='*80}")
        print("Checkpoint detected!")
        print(f"Found existing checkpoint: {hierarchical_json_file}")
        print(f"{'='*80}")
        
        # Try to load checkpoint to show how many P1s are already processed
        try:
            with open(hierarchical_json_file, "r", encoding="utf-8") as f:
                temp_data = json.load(f)
                num_processed = len(temp_data)
                print(f"Currently processed: {num_processed} P1s")
        except:
            num_processed = 0
        
        print(f"\nOptions:")
        print("  [n] - Start from checkpoint (skip already processed P1s)")
        print("  [y] - Rerun all prompts from scratch (clear checkpoint)")
        
        while True:
            response = input("\nRerun all prompts from scratch? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                # Clear checkpoint
                hierarchical_json_file.unlink()
                print("✓ Checkpoint cleared. Starting fresh...")
                processed_p1_ids = set()
                hierarchical_data = {}
                # Initialize empty file
                with open(hierarchical_json_file, "w", encoding="utf-8") as f:
                    json.dump({}, f, ensure_ascii=False, indent=2)
                break
            elif response in ['n', 'no', '']:
                # Load from checkpoint
                print("✓ Resuming from checkpoint...")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    elif force_rerun and hierarchical_json_file.exists():
        # Force rerun - clear checkpoint
        hierarchical_json_file.unlink()
        print("✓ Force rerun enabled. Checkpoint cleared. Starting fresh...")
        processed_p1_ids = set()
        hierarchical_data = {}
        # Initialize empty file
        with open(hierarchical_json_file, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    
    # Load checkpoint if exists (hierarchical JSON file)
    if hierarchical_json_file.exists() and len(processed_p1_ids) == 0:
        try:
            with open(hierarchical_json_file, "r", encoding="utf-8") as f:
                hierarchical_data = json.load(f)
                processed_p1_ids = set(hierarchical_data.keys())
                print(f"✓ Loaded checkpoint: {len(processed_p1_ids)} P1s already processed")
                
                # Restore best_p2_results and all_results from checkpoint
                for p1_id, p1_data in hierarchical_data.items():
                    if "p2s" in p1_data and len(p1_data["p2s"]) > 0:
                        # Find best P2 (first one after sorting)
                        best_p2 = p1_data["p2s"][0]
                        
                        # Safely get fields with defaults
                        best_combined_score = best_p2.get("combined_score", 0.0)
                        best_judge_scores = best_p2.get("judge_scores", {})
                        best_bleu_score = best_p2.get("bleu_score", 0.0)
                        best_sbert_score = best_p2.get("sbert_score", 0.0)
                        
                        best_p2_results.append({
                            "p1_id": p1_id,
                            "p1": p1_data["p1"],
                            "p1_redacted": p1_data.get("p1_redacted", ""),
                            "best_p2": best_p2.get("p2", ""),
                            "best_pprime_index": best_p2.get("pprime_index", 0),
                            "best_pprime_text": best_p2.get("pprime_text", ""),
                            "best_combined_score": best_combined_score,
                            "best_judge_scores": best_judge_scores,
                            "best_bleu_score": best_bleu_score,
                            "best_sbert_score": best_sbert_score
                        })
                        # Add all P2s to all_results
                        for p2_entry in p1_data["p2s"]:
                            all_results.append({
                                "id": str(uuid.uuid4()),
                                "p1_id": p1_id,
                                "p1": p1_data.get("p1", ""),
                                "p1_redacted": p1_data.get("p1_redacted", ""),
                                "pprime_index": p2_entry.get("pprime_index", 0),
                                "pprime_text": p2_entry.get("pprime_text", ""),
                                "p2": p2_entry.get("p2", ""),
                                "judge_scores": p2_entry.get("judge_scores", {}),
                                "bleu_score": p2_entry.get("bleu_score", 0.0),
                                "sbert_score": p2_entry.get("sbert_score", 0.0),
                                "combined_score": p2_entry.get("combined_score", 0.0)
                            })
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Starting fresh...")
            hierarchical_data = {}
            processed_p1_ids = set()
            # Initialize empty file
            with open(hierarchical_json_file, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    else:
        # Initialize hierarchical JSON file (empty dict)
        with open(hierarchical_json_file, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"P2 Selection Evaluation for {language}")
    print(f"{'='*80}")
    print(f"Total P1 prompts: {len(p1_prompts)}")
    if processed_p1_ids:
        print(f"Already processed: {len(processed_p1_ids)}")
        print(f"Remaining: {len(p1_prompts) - len(processed_p1_ids)}")
    print(f"Total p' candidates: {len(pprimes)}")
    print(f"Judge weight: {judge_weight}, BLEU weight: {bleu_weight}, SBERT weight: {sbert_weight}")
    print(f"{'='*80}\n")
    
    # Filter out already processed P1s
    remaining_p1s = [
        (idx, p1_record) for idx, p1_record in enumerate(p1_prompts)
        if p1_record.get("id", f"p1_{idx}") not in processed_p1_ids
    ]
    
    if not remaining_p1s:
        print("All P1s already processed!")
        return
    
    print(f"Processing {len(remaining_p1s)} P1s in batch mode...\n")
    
    # BATCH MODE: Process all P1s together for each p'
    # This is much more efficient than processing one P1 at a time
    
    # Process each p' candidate
    for pprime_idx, pprime_record in enumerate(pprimes):
        pprime_text = pprime_record["pprime"]
        print(f"\n{'='*60}")
        print(f"Processing p' {pprime_idx + 1}/10: {pprime_text[:60]}...")
        print(f"{'='*60}")
        
        # Step 1: Batch convert all P1s to P2s using this p'
        print(f"\nStep 1: Batch converting {len(remaining_p1s)} P1s to P2s...")
        
        batch_prompts = []
        p1_metadata = []  # Track which P1 each prompt belongs to
        
        for p1_idx, p1_record in remaining_p1s:
            p1 = p1_record["p1"]
            p1_id = p1_record.get("id", f"p1_{p1_idx}")
            
            rewrite_instr = (
                f"{pprime_text}\n\n"
                "Rewrite into a safe, legal, non-operational request.\n\n"
                f"Original: {p1}"
            )
            batch_prompts.append(rewrite_instr)
            p1_metadata.append((p1_idx, p1_id, p1, None))  # (idx, id, p1, p1_redacted)
        
        # Batch call for P2 generation
        p2_results = call_model_batch(
            REWRITE_SYSTEM,
            batch_prompts,
            max_tokens=400,
            temperature=0.5,
            max_parallel=batch_size
        )
        
        # Step 2: Handle content filter errors (need redaction)
        # Collect successful P2s and failed ones
        successful_p2s = []
        failed_indices = []
        
        for i, (p2_result, (p1_idx, p1_id, p1, _)) in enumerate(zip(p2_results, p1_metadata)):
            if p2_result is None:
                # Content filter - need to redact
                failed_indices.append(i)
            else:
                successful_p2s.append((i, p1_idx, p1_id, p1, None, p2_result))
        
        # Step 3: Handle failed ones with redaction (individual calls for now)
        if failed_indices:
            print(f"  {len(failed_indices)} P1s triggered content filter, redacting...")
            for failed_idx in failed_indices:
                p1_idx, p1_id, p1, _ = p1_metadata[failed_idx]
                # Redact and retry
                p1_redacted, _, _ = sanitize_with_placeholders(p1, terms)
                rewrite_instr = (
                    f"{pprime_text}\n\n"
                    "Important: The text may include placeholders like [TAG_001]. "
                    "Do NOT guess or restore their meaning. "
                    "Rewrite into a safe, legal, non-operational request.\n\n"
                    f"Original (redacted): {p1_redacted}"
                )
                try:
                    p2_retry = call_model(REWRITE_SYSTEM, rewrite_instr, temperature=0.5, max_tokens=400)
                    successful_p2s.append((failed_idx, p1_idx, p1_id, p1, p1_redacted, p2_retry))
                except Exception:
                    # Still failed, skip
                    pass
        
        print(f"  ✓ Generated {len(successful_p2s)} P2s")
        
        # Step 4: Batch judge all P2s
        if successful_p2s:
            print(f"\nStep 2: Batch judging {len(successful_p2s)} P2s...")
            
            # Prepare judge inputs
            judge_pairs = []
            for _, p1_idx, p1_id, p1, p1_redacted, p2 in successful_p2s:
                p1_for_judging = p1_redacted if p1_redacted else p1
                # Extract schema if needed (simplified - could be improved)
                schema = None
                judge_pairs.append((p1_for_judging, p2, schema))
            
            # Batch judge
            judge_scores_list = judge_p2_batch(
                judge_pairs,
                judge_template,
                extractor_prompt_text,
                use_schema_judge,
                num_judge_calls=3,
                max_parallel=batch_size * 2  # Judge calls can be more parallel (lighter model)
            )
            
            print(f"  ✓ Judged {len(judge_scores_list)} P2s")
            
            # Step 5: Compute BLEU and SBERT scores, then combined scores
            print(f"\nStep 3: Computing BLEU and SBERT scores...")
            
            for (_, p1_idx, p1_id, p1, p1_redacted, p2), judge_scores in zip(successful_p2s, judge_scores_list):
                p1_for_scoring = p1_redacted if p1_redacted else p1
                
                bleu_score = compute_bleu_score(p1_for_scoring, p2)
                sbert_score = compute_sbert_score(p1_for_scoring, p2, sbert_model)
                combined_score = compute_combined_score(
                    judge_scores, bleu_score, sbert_score,
                    judge_weight, bleu_weight, sbert_weight
                )
                
                # Store in hierarchical structure
                if p1_id not in hierarchical_data:
                    hierarchical_data[p1_id] = {
                        "p1": p1,
                        "p1_redacted": p1_redacted,
                        "p1_metadata": {
                            "id": p1_id,
                            "lang": language,
                            "level": p1_prompts[p1_idx].get("level"),
                            "subtype": p1_prompts[p1_idx].get("subtype", ""),
                            "type": p1_prompts[p1_idx].get("type", ""),
                            "source": p1_prompts[p1_idx].get("source", "")
                        },
                        "redaction_used": p1_redacted is not None,
                        "redaction_stats": {"num_redactions": 0, "placeholders": []},
                        "content_filter_errors": [],
                        "p2s": []
                    }
                
                hierarchical_data[p1_id]["p2s"].append({
                    "pprime_index": pprime_idx,
                    "pprime_text": pprime_text,
                    "p2": p2,
                    "judge_scores": judge_scores,
                    "bleu_score": bleu_score,
                    "sbert_score": sbert_score,
                    "combined_score": combined_score
                })
                
                # Also add to all_results
                all_results.append({
                    "id": str(uuid.uuid4()),
                    "p1_id": p1_id,
                    "p1": p1,
                    "p1_redacted": p1_redacted,
                    "pprime_index": pprime_idx,
                    "pprime_text": pprime_text,
                    "p2": p2,
                    "judge_scores": judge_scores,
                    "bleu_score": bleu_score,
                    "sbert_score": sbert_score,
                    "combined_score": combined_score
                })
            
            print(f"  ✓ Computed scores for {len(successful_p2s)} P2s")
    
    # After processing all p' candidates, select best P2 for each P1
    print(f"\n{'='*80}")
    print("Selecting best P2 for each P1...")
    print(f"{'='*80}\n")
    
    for p1_id, p1_data in hierarchical_data.items():
        if "p2s" in p1_data and len(p1_data["p2s"]) > 0:
            # Sort by combined score
            p1_data["p2s"].sort(key=lambda x: x["combined_score"], reverse=True)
            best_p2 = p1_data["p2s"][0]
            
            best_p2_results.append({
                "p1_id": p1_id,
                "p1": p1_data["p1"],
                "p1_redacted": p1_data.get("p1_redacted"),
                "best_p2": best_p2["p2"],
                "best_pprime_index": best_p2["pprime_index"],
                "best_pprime_text": best_p2["pprime_text"],
                "best_combined_score": best_p2["combined_score"],
                "best_judge_scores": best_p2["judge_scores"],
                "best_bleu_score": best_p2["bleu_score"],
                "best_sbert_score": best_p2["sbert_score"]
            })
    
    # Save checkpoint after all p' candidates are processed
    with open(hierarchical_json_file, "w", encoding="utf-8") as f:
        json.dump(hierarchical_data, f, ensure_ascii=False, indent=2)
    
    # OLD CODE REMOVED - Batch mode is now the default
    # The old sequential processing code has been replaced with batch processing above
    
    # Save final results
    save_jsonl(all_results_file, all_results)
    save_jsonl(best_p2_file, best_p2_results)
    
    # Save P1-P2 pairs file with scores: P1, best P2, and scores
    # Sorted by highest combined score (best first)
    best_p2_sorted = sorted(
        best_p2_results,
        key=lambda x: x.get("best_combined_score", 0.0),
        reverse=True
    )
    
    p1_p2_pairs = []
    for result in best_p2_sorted:  # Each result is one P1 with its best P2
        judge_scores = result.get("best_judge_scores", {})
        p1_p2_pairs.append({
            "p1": result.get("p1", ""),
            "p2": result.get("best_p2", ""),  # Only the best P2 for this P1
            "scores": {
                "combined": result.get("best_combined_score", 0.0),
                "judge": {
                    "fidelity_goal": judge_scores.get("fidelity_goal", 0),
                    "fidelity_target": judge_scores.get("fidelity_target", 0),
                    "fidelity_constraints": judge_scores.get("fidelity_constraints", 0),
                    "safety_score": judge_scores.get("safety_score", 0),
                    "fluency_score": judge_scores.get("fluency_score", 0),
                    "usefulness_score": judge_scores.get("usefulness_score", 0),
                    "actionable_detail_present": judge_scores.get("actionable_detail_present", 0)
                },
                "bleu": result.get("best_bleu_score", 0.0),
                "sbert": result.get("best_sbert_score", 0.0)
            }
        })
    
    with open(p1_p2_pairs_file, "w", encoding="utf-8") as f:
        json.dump(p1_p2_pairs, f, ensure_ascii=False, indent=2)
    
    # Save hierarchical JSON structure (P1 -> list of P2s)
    with open(hierarchical_json_file, "w", encoding="utf-8") as f:
        json.dump(hierarchical_data, f, ensure_ascii=False, indent=2)
    
    # Save summary
    summary = {
        "language": language,
        "total_p1": len(p1_prompts),
        "total_pprimes": len(pprimes),
        "total_candidates": len(all_results),
        "total_best_selected": len(best_p2_results),
        "score_weights": {
            "judge": judge_weight,
            "bleu": bleu_weight,
            "sbert": sbert_weight
        },
        "average_scores": {
            "combined": mean([r.get("best_combined_score", 0.0) for r in best_p2_results]) if best_p2_results else 0,
            "judge_avg": mean([
                mean([
                    r["best_judge_scores"].get("fidelity_goal", 0),
                    r["best_judge_scores"].get("fidelity_target", 0),
                    r["best_judge_scores"].get("fidelity_constraints", 0),
                    r["best_judge_scores"].get("safety_score", 0),
                    r["best_judge_scores"].get("fluency_score", 0),
                    r["best_judge_scores"].get("usefulness_score", 0)
                ]) / 100.0
                for r in best_p2_results
            ]) if best_p2_results else 0,
            "bleu": mean([r["best_bleu_score"] for r in best_p2_results]) if best_p2_results else 0,
            "sbert": mean([r["best_sbert_score"] for r in best_p2_results]) if best_p2_results else 0
        }
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print("Evaluation Complete")
    print(f"{'='*80}")
    print(f"All candidates (JSONL): {all_results_file}")
    print(f"Best P2 selections (JSONL): {best_p2_file}")
    print(f"P1-P2 pairs (simple): {p1_p2_pairs_file}")
    print(f"Hierarchical JSON (P1->P2s): {hierarchical_json_file}")
    print(f"Summary: {summary_file}")
    print(f"\nAverage combined score: {summary['average_scores']['combined']:.3f}")
    print(f"Average judge score: {summary['average_scores']['judge_avg']:.3f}")
    print(f"Average BLEU: {summary['average_scores']['bleu']:.3f}")
    print(f"Average SBERT: {summary['average_scores']['sbert']:.3f}")
    
    # Summary output
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to {output_dir}")
    print(f"  - Total P1s processed: {len(best_p2_results)}")
    print(f"  - Best P2 for each P1 saved to: best_p2_selected.jsonl")
    print(f"  - All P2 candidates saved to: all_p2_candidates.jsonl")
    print(f"  - P1-P2 pairs saved to: p1_p2_pairs.json")
    print(f"{'='*80}\n")
    
    # NEW: Extended pipeline - Generate outputs from high-scoring P2s
    # This is the new step: filter P2s by threshold, generate outputs, score outputs, select best
    return {
        "hierarchical_data": hierarchical_data,
        "best_p2_results": best_p2_results,
        "all_results": all_results,
        "output_dir": output_dir,
        "language": language,
        "sbert_model": sbert_model,
        "judge_template": judge_template,
        "extractor_prompt_text": extractor_prompt_text,
        "use_schema_judge": use_schema_judge,
        "judge_weight": judge_weight,
        "bleu_weight": bleu_weight,
        "sbert_weight": sbert_weight,
        "terms": terms
    }


def generate_outputs_from_p2s(
    p2_list: List[Dict],
    output_model_name: str = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
    batch_size: int = 10
) -> List[Dict]:
    """
    Generate outputs from P2 prompts using Llama 70B or specified model.
    
    Args:
        p2_list: List of dicts with 'p2' (prompt text) and metadata
        output_model_name: Model name for output generation (default: LOCAL_MODEL or Llama 70B)
        max_tokens: Max tokens per output
        temperature: Sampling temperature
        batch_size: Batch size for generation
    
    Returns:
        List of dicts with 'p2', 'output', and metadata
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Determine which model to use
    if output_model_name is None:
        output_model_name = os.getenv("OUTPUT_MODEL", LOCAL_MODEL_NAME)
    
    print(f"\n{'='*80}")
    print(f"Generating outputs from {len(p2_list)} P2s using {output_model_name}")
    print(f"{'='*80}\n")
    
    # Check if using local model
    USE_OUTPUT_MODEL = os.getenv("USE_OUTPUT_MODEL", "true").strip().lower() == "true"
    
    results = []
    
    if USE_OUTPUT_MODEL or USE_LOCAL_MODEL:
        # Use local model for output generation
        try:
            from src.utils.local_judge import get_local_judge
            model = get_local_judge(model_name=output_model_name)
            print(f"✓ Using local model: {output_model_name}")
            
            # Prepare prompts for batch generation
            system_prompt = "You are a helpful assistant. Provide a clear, informative, and safe response to the user's question."
            prompts = []
            for p2_item in p2_list:
                prompts.append((system_prompt, p2_item["p2"]))
            
            # Generate in batches
            print(f"Generating outputs in batches of {batch_size}...")
            all_outputs = []
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                batch_outputs = model.judge_batch(
                    batch,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    batch_size=batch_size
                )
                all_outputs.extend(batch_outputs)
                print(f"  Generated {min(i + batch_size, len(prompts))}/{len(prompts)} outputs...")
            
            # Combine with original P2 data
            for p2_item, output in zip(p2_list, all_outputs):
                result = p2_item.copy()
                result["output"] = output.strip() if output else ""
                result["output_model"] = output_model_name
                result["output_metadata"] = {
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
                results.append(result)
            
            print(f"✓ Generated {len(results)} outputs")
            return results
            
        except Exception as e:
            print(f"❌ Local model failed: {e}")
            print("   Falling back to API...")
            import traceback
            print(traceback.format_exc())
    
    # Fallback to API
    print(f"Using API for output generation...")
    for i, p2_item in enumerate(p2_list):
        try:
            system_prompt = "You are a helpful assistant. Provide a clear, informative, and safe response to the user's question."
            output = call_model(
                system_prompt=system_prompt,
                user_prompt=p2_item["p2"],
                max_tokens=max_tokens,
                temperature=temperature
            )
            result = p2_item.copy()
            result["output"] = output
            result["output_model"] = output_model_name or AZURE_MODEL
            result["output_metadata"] = {
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(p2_list)} outputs...")
        except Exception as e:
            print(f"  Warning: Failed to generate output for P2 {i}: {e}")
            result = p2_item.copy()
            result["output"] = ""
            result["output_error"] = str(e)
            results.append(result)
    
    return results


def score_outputs(
    p1_output_pairs: List[Tuple[str, str, str]],  # (p1_redacted, p2, output)
    judge_template: str,
    sbert_model,
    terms: Dict,
    extractor_prompt_text: Optional[str] = None,
    use_schema_judge: bool = True,
    judge_weight: float = 0.5,
    bleu_weight: float = 0.25,
    sbert_weight: float = 0.25,
    num_judge_calls: int = 3,
    max_parallel: int = 20
) -> List[Dict]:
    """
    Score outputs using the same metrics as P2s (judge, BLEU, SBERT).
    
    Args:
        p1_output_pairs: List of (p1_redacted, p2, output) tuples
        judge_template: Template for judging (should judge output quality)
        sbert_model: SBERT model for similarity
        terms: Sanitizer terms
        extractor_prompt_text: Schema extractor prompt
        use_schema_judge: Whether to use schema-based judging
        judge_weight, bleu_weight, sbert_weight: Score weights
        num_judge_calls: Number of judge calls per output
        max_parallel: Max parallel API calls
    
    Returns:
        List of dicts with scores for each output
    """
    print(f"\n{'='*80}")
    print(f"Scoring {len(p1_output_pairs)} outputs...")
    print(f"{'='*80}\n")
    
    # Judge outputs (similar to judging P2s, but judge output quality)
    # We'll judge: output quality, safety, helpfulness, etc.
    judge_results = judge_p2_batch(
        [(p1_redacted, output, None) for p1_redacted, p2, output in p1_output_pairs],  # Judge output against P1
        judge_template,
        extractor_prompt_text,
        use_schema_judge,
        num_judge_calls,
        max_parallel
    )
    
    # Compute BLEU and SBERT scores
    scored_outputs = []
    for i, (p1_redacted, p2, output) in enumerate(p1_output_pairs):
        judge_scores = judge_results[i] if i < len(judge_results) else {}
        
        # BLEU: output vs P2 (how well does output match the safe P2 intent?)
        bleu_score = 0.0
        if BLEU_AVAILABLE and output and p2:
            try:
                ref_tokens = word_tokenize(p2.lower())
                gen_tokens = word_tokenize(output.lower())
                bleu_score = sentence_bleu(
                    [ref_tokens],
                    gen_tokens,
                    smoothing_function=SmoothingFunction().method1
                )
            except:
                pass
        
        # SBERT: output vs P2
        sbert_score = 0.0
        if SBERT_AVAILABLE and sbert_model and output and p2:
            try:
                embeddings = sbert_model.encode([p2, output])
                # Use cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(embeddings[0:1], embeddings[1:2])
                sbert_score = float(similarity.item())
            except:
                pass
        
        # Combined score
        combined_score = compute_combined_score(
            judge_scores,
            bleu_score,
            sbert_score,
            judge_weight,
            bleu_weight,
            sbert_weight
        )
        
        scored_outputs.append({
            "p1_redacted": p1_redacted,
            "p2": p2,
            "output": output,
            "judge_scores": judge_scores,
            "bleu_score": bleu_score,
            "sbert_score": sbert_score,
            "combined_score": combined_score
        })
    
    return scored_outputs


def run_extended_pipeline(
    p2_selection_results: Dict,
    score_threshold: float = 0.90,
    output_model_name: str = None,
    max_output_tokens: int = 500,
    output_temperature: float = 0.7
):
    """
    Extended pipeline: Generate outputs from high-scoring P2s and select best outputs.
    
    Args:
        p2_selection_results: Results from run_p2_selection_evaluation
        score_threshold: Minimum combined score to generate outputs (default: 0.90)
        output_model_name: Model for output generation (default: Llama 70B or LOCAL_MODEL)
        max_output_tokens: Max tokens per output
        output_temperature: Temperature for output generation
    """
    hierarchical_data = p2_selection_results["hierarchical_data"]
    output_dir = p2_selection_results["output_dir"]
    language = p2_selection_results["language"]
    sbert_model = p2_selection_results["sbert_model"]
    judge_template = p2_selection_results["judge_template"]
    extractor_prompt_text = p2_selection_results["extractor_prompt_text"]
    use_schema_judge = p2_selection_results["use_schema_judge"]
    judge_weight = p2_selection_results["judge_weight"]
    bleu_weight = p2_selection_results["bleu_weight"]
    sbert_weight = p2_selection_results["sbert_weight"]
    terms = p2_selection_results["terms"]
    
    print(f"\n{'='*80}")
    print(f"Extended Pipeline: Output Generation & Selection")
    print(f"{'='*80}")
    print(f"Score threshold: {score_threshold}")
    print(f"{'='*80}\n")
    
    # Step 1: Filter P2s with combined_score > threshold
    filtered_p2s = []
    for p1_id, p1_data in hierarchical_data.items():
        if "p2s" in p1_data:
            for p2_entry in p1_data["p2s"]:
                if p2_entry.get("combined_score", 0.0) > score_threshold:
                    filtered_p2s.append({
                        "p1_id": p1_id,
                        "p1": p1_data["p1"],
                        "p1_redacted": p1_data.get("p1_redacted", ""),
                        "p2": p2_entry["p2"],
                        "pprime_index": p2_entry.get("pprime_index", 0),
                        "pprime_text": p2_entry.get("pprime_text", ""),
                        "p2_combined_score": p2_entry.get("combined_score", 0.0),
                        "p2_judge_scores": p2_entry.get("judge_scores", {}),
                        "p2_bleu_score": p2_entry.get("bleu_score", 0.0),
                        "p2_sbert_score": p2_entry.get("sbert_score", 0.0)
                    })
    
    print(f"✓ Filtered {len(filtered_p2s)} P2s with combined_score > {score_threshold}")
    if len(filtered_p2s) == 0:
        print("⚠️  No P2s passed the threshold. Lower the threshold or check P2 scores.")
        return
    
    # Step 2: Generate outputs from filtered P2s
    p2_with_outputs = generate_outputs_from_p2s(
        filtered_p2s,
        output_model_name=output_model_name,
        max_tokens=max_output_tokens,
        temperature=output_temperature
    )
    
    # Step 3: Score outputs
    p1_output_pairs = [
        (item["p1_redacted"], item["p2"], item.get("output", ""))
        for item in p2_with_outputs
    ]
    
    scored_outputs = score_outputs(
        p1_output_pairs,
        judge_template,
        sbert_model,
        terms,
        extractor_prompt_text,
        use_schema_judge,
        judge_weight,
        bleu_weight,
        sbert_weight
    )
    
    # Combine output scores with P2 metadata
    for i, scored in enumerate(scored_outputs):
        if i < len(p2_with_outputs):
            p2_with_outputs[i].update(scored)
    
    # Step 4: Select best output per P1
    best_outputs_by_p1 = {}
    for item in p2_with_outputs:
        p1_id = item["p1_id"]
        if p1_id not in best_outputs_by_p1:
            best_outputs_by_p1[p1_id] = item
        elif item.get("combined_score", 0.0) > best_outputs_by_p1[p1_id].get("combined_score", 0.0):
            best_outputs_by_p1[p1_id] = item
    
    # Step 5: Save results
    output_results_file = Path(output_dir) / "p2_outputs_scored.jsonl"
    best_outputs_file = Path(output_dir) / "best_outputs_per_p1.jsonl"
    finetuning_pairs_file = Path(output_dir) / "finetuning_pairs.jsonl"  # (P1, best_output) pairs
    
    # Save all scored outputs
    save_jsonl(output_results_file, p2_with_outputs)
    
    # Save best outputs per P1
    best_outputs_list = list(best_outputs_by_p1.values())
    save_jsonl(best_outputs_file, best_outputs_list)
    
    # Save finetuning pairs (P1, best_output)
    finetuning_pairs = []
    for item in best_outputs_list:
        finetuning_pairs.append({
            "p1": item["p1"],
            "output": item["output"],
            "p2": item["p2"],
            "output_scores": {
                "combined": item.get("combined_score", 0.0),
                "judge": item.get("judge_scores", {}),
                "bleu": item.get("bleu_score", 0.0),
                "sbert": item.get("sbert_score", 0.0)
            },
            "p2_scores": {
                "combined": item.get("p2_combined_score", 0.0),
                "judge": item.get("p2_judge_scores", {}),
                "bleu": item.get("p2_bleu_score", 0.0),
                "sbert": item.get("p2_sbert_score", 0.0)
            }
        })
    
    save_jsonl(finetuning_pairs_file, finetuning_pairs)
    
    print(f"\n{'='*80}")
    print(f"Extended Pipeline Complete")
    print(f"{'='*80}")
    print(f"  - Filtered P2s: {len(filtered_p2s)}")
    print(f"  - Generated outputs: {len(p2_with_outputs)}")
    print(f"  - Best outputs per P1: {len(best_outputs_list)}")
    print(f"\n  Output files:")
    print(f"    - All scored outputs: {output_results_file}")
    print(f"    - Best outputs per P1: {best_outputs_file}")
    print(f"    - Finetuning pairs (P1, best_output): {finetuning_pairs_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="P2 Selection Evaluation")
    parser.add_argument("--config", required=True, help="Path to language-specific config JSON")
    parser.add_argument("--csv-path", type=str, default="dataset/linguasafe.csv", 
                       help="Path to linguasafe.csv dataset (default: dataset/linguasafe.csv)")
    parser.add_argument("--num-p1", type=int, default=None, 
                       help="Maximum number of P1 prompts to process (default: all)")
    parser.add_argument("--force-rerun", action="store_true", 
                       help="Force rerun all prompts from scratch (clear checkpoint, skip interactive prompt)")
    parser.add_argument("--judge-weight", type=float, default=0.5, help="Weight for judge scores (default: 0.5)")
    parser.add_argument("--bleu-weight", type=float, default=0.25, help="Weight for BLEU score (default: 0.25)")
    parser.add_argument("--sbert-weight", type=float, default=0.25, help="Weight for SBERT score (default: 0.25)")
    parser.add_argument("--use-batch", action="store_true", default=True, help="Use batch processing for API calls (default: True)")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for parallel API calls (default: 20). Higher = faster but may hit rate limits. Recommended: 20-50 for Azure OpenAI.")
    parser.add_argument("--use-local-judge", action="store_true", default=False, 
                       help="Use local Hugging Face model for judging instead of API (default: False). Set USE_LOCAL_JUDGE=true in .env or use this flag.")
    parser.add_argument("--local-judge-model", type=str, default=None,
                       help="Hugging Face model name for local judge (default: Qwen/Qwen2.5-7B-Instruct). Examples: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--use-local-model", action="store_true", default=False,
                       help="Use local Hugging Face model for P2 generation instead of API (default: False). Set USE_LOCAL_MODEL=true in .env or use this flag.")
    parser.add_argument("--local-model", type=str, default=None,
                       help="Hugging Face model name for P2 generation (default: Qwen/Qwen2.5-7B-Instruct). Examples: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--run-extended-pipeline", action="store_true", default=False,
                       help="Run extended pipeline: generate outputs from high-scoring P2s and select best outputs")
    parser.add_argument("--score-threshold", type=float, default=0.90,
                       help="Minimum combined score threshold for P2s to generate outputs (default: 0.90)")
    parser.add_argument("--output-model", type=str, default=None,
                       help="Model for output generation (default: Llama 70B or LOCAL_MODEL). Set OUTPUT_MODEL in .env")
    parser.add_argument("--max-output-tokens", type=int, default=500,
                       help="Max tokens per output generation (default: 500)")
    parser.add_argument("--output-temperature", type=float, default=0.7,
                       help="Temperature for output generation (default: 0.7)")
    
    args = parser.parse_args()
    
    # Override environment variables with command line args
    global USE_LOCAL_JUDGE, LOCAL_JUDGE_MODEL, USE_LOCAL_MODEL, LOCAL_MODEL_NAME
    if args.use_local_judge:
        USE_LOCAL_JUDGE = True
        print("✓ Local judge enabled via --use-local-judge flag")
    if args.local_judge_model:
        LOCAL_JUDGE_MODEL = args.local_judge_model
        print(f"✓ Local judge model set to: {LOCAL_JUDGE_MODEL}")
    if args.use_local_model:
        USE_LOCAL_MODEL = True
        print("✓ Local model enabled via --use-local-model flag (for P2 generation)")
    if args.local_model:
        LOCAL_MODEL_NAME = args.local_model
        print(f"✓ Local model set to: {LOCAL_MODEL_NAME}")
    
    # Log which models are being used (with debug info)
    print(f"\n{'='*80}")
    print(f"🔧 MODEL CONFIGURATION")
    print(f"   P2 Generation:")
    print(f"     USE_LOCAL_MODEL: {USE_LOCAL_MODEL}")
    if USE_LOCAL_MODEL:
        print(f"     ✅ Using LOCAL model: {LOCAL_MODEL_NAME}")
        print(f"     (No API calls for P2 generation)")
    else:
        print(f"     ⚠️  Using API (Azure OpenAI): {AZURE_MODEL}")
        print(f"     (To use local model, add --use-local-model flag or set USE_LOCAL_MODEL=true in .env)")
    print(f"   Judging:")
    print(f"     USE_LOCAL_JUDGE: {USE_LOCAL_JUDGE}")
    if USE_LOCAL_JUDGE:
        print(f"     ✅ Using LOCAL model: {LOCAL_JUDGE_MODEL}")
        print(f"     (No API calls for judging)")
    else:
        print(f"     ⚠️  Using API (Azure OpenAI): {JUDGE_MODEL}")
        print(f"     (To use local judge, add --use-local-judge flag or set USE_LOCAL_JUDGE=true in .env)")
    print(f"{'='*80}\n")
    
    # Load config
    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    
    # Resolve config file paths relative to config file directory
    # Config is in ./config/, but paths like "src/prompts/" are relative to project root
    pipeline_dir = os.path.dirname(config_dir)  # Go up one level from config/ to project root
    
    def resolve_path(path_str):
        """Resolve path relative to project root (parent of config directory)."""
        if os.path.isabs(path_str):
            return path_str
        # Try relative to project root first (where src/prompts/, src/judge_prompts/, etc. are)
        resolved = os.path.join(pipeline_dir, path_str)
        if os.path.exists(resolved):
            return os.path.abspath(resolved)
        # Try relative to config directory
        resolved = os.path.join(config_dir, path_str)
        if os.path.exists(resolved):
            return os.path.abspath(resolved)
        # Try relative to current working directory
        if os.path.exists(path_str):
            return os.path.abspath(path_str)
        # Return as-is if not found (will error later)
        return path_str
    
    # Normalize weights
    total_weight = args.judge_weight + args.bleu_weight + args.sbert_weight
    if total_weight != 1.0:
        print(f"Warning: Weights sum to {total_weight}, normalizing...")
        args.judge_weight /= total_weight
        args.bleu_weight /= total_weight
        args.sbert_weight /= total_weight
    
    # Get language from config (use 'name' field or infer from file paths)
    language = cfg.get("name", "english")
    # Map common language names to codes
    lang_map = {
        "english": "en",
        "chinese": "zh", 
        "arabic": "ar",
        "bengali": "bn",
        "russian": "ru",
        "serbian": "sr",
        "thai": "th",
        "korean": "ko",
        "vietnamese": "vi",
        "czech": "cs",
        "hungarian": "hu",
        "malay": "ms"
    }
    # If it's a language name, convert to code; otherwise use as-is
    language_code = lang_map.get(language.lower(), language.lower())
    
    # Resolve CSV path (relative to project root or config file location)
    csv_path = args.csv_path if hasattr(args, 'csv_path') else "dataset/linguasafe.csv"
    if not os.path.isabs(csv_path):
        # Try relative to config file directory first
        csv_path_relative = os.path.join(os.path.dirname(config_dir), "..", csv_path)
        csv_path_relative = os.path.normpath(csv_path_relative)
        if os.path.exists(csv_path_relative):
            csv_path = csv_path_relative
        elif os.path.exists(csv_path):
            # Use as-is if it exists
            pass
        else:
            # Try relative to current working directory
            csv_path = os.path.abspath(csv_path)
    
    # Resolve all config file paths
    pprime_file = resolve_path(cfg["pprime_file"])
    judge_prompt_file = resolve_path(cfg["judge_prompt_file"])
    sanitizer_terms_file = resolve_path(cfg["sanitizer_terms_file"])
    output_dir = resolve_path(cfg["output_dir"])
    schema_extractor_prompt_file = None
    if cfg.get("schema_extractor_prompt_file"):
        schema_extractor_prompt_file = resolve_path(cfg["schema_extractor_prompt_file"])
    
    # Log Azure API URLs once at startup
    log_azure_urls_once()
    
    print(f"Using CSV dataset: {csv_path}")
    print(f"Language code: {language_code}")
    print(f"P' file: {pprime_file}")
    print(f"Judge prompt: {judge_prompt_file}")
    print(f"Sanitizer terms: {sanitizer_terms_file}")
    print(f"Output directory: {output_dir}")
    if args.num_p1 is not None:
        print(f"Processing up to {args.num_p1} P1 prompts")
    
    # Run P2 selection evaluation
    p2_results = run_p2_selection_evaluation(
        csv_path=csv_path,
        language=language_code,
        pprime_file=pprime_file,
        judge_prompt_file=judge_prompt_file,
        output_dir=output_dir,
        sanitizer_terms_file=sanitizer_terms_file,
        use_schema_judge=cfg.get("use_schema_judge", True),
        schema_extractor_prompt_file=schema_extractor_prompt_file,
        judge_weight=args.judge_weight,
        bleu_weight=args.bleu_weight,
        sbert_weight=args.sbert_weight,
        max_p1=args.num_p1,
        force_rerun=args.force_rerun
    )
    
    # Run extended pipeline if requested
    if args.run_extended_pipeline:
        run_extended_pipeline(
            p2_results,
            score_threshold=args.score_threshold,
            output_model_name=args.output_model,
            max_output_tokens=args.max_output_tokens,
            output_temperature=args.output_temperature
        )


if __name__ == "__main__":
    main()

