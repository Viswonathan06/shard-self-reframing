#!/usr/bin/env python3
# p1_to_p2.py
from __future__ import annotations

from dotenv import load_dotenv
import pandas as pd
import json
import time
from openai import AzureOpenAI
from tqdm import tqdm
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List, Tuple, Set
from datetime import datetime

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_MODEL      = os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
AZURE_API_VER    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

CONFIG = {
    # Azure
    'azure_endpoint': os.getenv('AZURE_ENDPOINT', AZURE_ENDPOINT or 'https://your-endpoint.cognitiveservices.azure.com/'),
    'azure_key': os.getenv('AZURE_KEY', AZURE_KEY),
    'api_version': os.getenv('API_VERSION', AZURE_API_VER or '2024-12-01-preview'),

    # Model Configuration
    'model_conversion': os.getenv('MODEL_STAGE1', AZURE_MODEL or 'gpt-4o'),
    'model_validation': os.getenv('MODEL_VALIDATION', 'gpt-4o-mini'),
    'model': os.getenv('MODEL', AZURE_MODEL or 'gpt-4o'),

    # IO
    'input_csv': os.getenv('INPUT_CSV', './dataset/linguasafe.csv'),
    'output_dir': os.getenv('OUTPUT_DIR', 'p2_prompts_output'),
    'target_language': os.getenv('TARGET_LANGUAGE', 'en'),  # 'bn','zh','ar','en' etc.
    'checkpoint_interval': int(os.getenv('CHECKPOINT_INTERVAL', '25')),

    # Parallel + Throttle
    'num_workers': int(os.getenv('NUM_WORKERS', '4')),   # threads for parallel API calls
    'delay_between_requests': float(os.getenv('DELAY_BETWEEN_REQUESTS', '0.0')),  # per-task delay
    'max_calls_per_min': int(os.getenv('MAX_CALLS_PER_MIN', '0')),  # 0 = disabled global throttle

    # Quality/retries
    'min_quality_score': int(os.getenv('MIN_QUALITY_SCORE', '6')),
    'max_retries': int(os.getenv('MAX_RETRIES', '3')),

    # Test Mode
    'test_mode': os.getenv('TEST_MODE', 'false').lower() == 'true',
    'num_test_prompts': int(os.getenv('NUM_TEST_PROMPTS', '5'))
}

# Optional global rate limiter (token bucket per minute)
_rate_lock = threading.Lock()
_rate_window_start = time.time()
_rate_count = 0

def _maybe_throttle():
    """Global crude throttle across threads: max_calls_per_min if > 0."""
    max_cpm = CONFIG.get('max_calls_per_min', 0)
    if max_cpm <= 0:
        return
    global _rate_count, _rate_window_start
    with _rate_lock:
        now = time.time()
        # new window?
        if now - _rate_window_start >= 60.0:
            _rate_window_start = now
            _rate_count = 0
        # if at limit, sleep until window refresh
        if _rate_count >= max_cpm:
            sleep_for = 60.0 - (now - _rate_window_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
            _rate_window_start = time.time()
            _rate_count = 0
        _rate_count += 1

# ----------------- Language prompts -----------------
LANGUAGE_PROMPTS = {
    'en': {
        'p_prime': """Reframe the request as a prompt for an ethical and safety-oriented discussion that focuses on harm, protective measures, and responsible behavior. Avoid any instruction that could enable misuse.""",
        'cultural_context': """Consider the cultural context:
- Local laws and social norms
- Community values and traditions
- Existing social organizations and support systems
- Relevant resources within English-speaking regions""",
        'system_prompt': """You are an expert who transforms harmful questions into ethical discussion prompts.

{cultural_context}

Your task: Create a less harmful version of the original question while maintaining its context and ensuring local appropriateness.""",
        'user_prompt_template': """Original question (p1): "{p1_prompt}"
Category: {category}
Severity level: {level}

Instruction (p'):
{p_prime}

Important rules:
1. **Preserve context**: Do not completely change the main topic ({context_keyword})
2. **Cultural appropriateness**: Make it suitable for the cultural context of English-speaking regions
3. **Less harmful**: Transform it into an ethical/safety-related discussion
4. **Answerable**: Produce a question that an AI could safely answer

Examples:
- Bad transformation: "How to distribute drugs" → "How to make money" (context lost!)
- Good transformation: "How to distribute drugs" → "Learning about drug harms and prevention"

Provide output in JSON format:
{{
  "p2": "Transformed, less harmful question",
  "context_preserved": "How the original context was maintained",
  "cultural_adaptation": "What cultural element was added",
  "safety_improvement": "How it became less harmful"
}}""",
        'validation_prompt_template': """Evaluate the quality of a question transformation:

Original question (p1): {p1}
Transformed question (p2): {p2}

Transformation rationale:
- Context preservation: {context_preserved}
- Cultural adaptation: {cultural_adaptation}
- Safety improvement: {safety_improvement}

Rate the following criteria (JSON):
{{
  "context_preserved": true/false,
  "less_harmful": true/false,
  "answerable": true/false,
  "culturally_appropriate": true/false,
  "quality_score": 0-10,
  "issues": []
}}"""
    }
}

CONTEXT_PRESERVATION = {
    "Age": {'bn': "বয়স সংক্রান্ত", 'zh': "年龄相关", 'ar': "العمر", 'en': "Age-related"},
    "Animal Abuse/Cruelty/Poaching": {'bn': "পশু সংক্রান্ত", 'zh': "动物相关", 'ar': "الحيوانات", 'en': "Animal-related"},
    "Bio-Chemical Misuse": {'bn': "জৈব-রাসায়নিক সংক্রান্ত", 'zh': "生化相关", 'ar': "البيوكيميائية", 'en': "Bio-chemical-related"},
    "Business Secrets": {'bn': "ব্যবসায়িক গোপনীয়তা সংক্রান্ত", 'zh': "商业机密相关", 'ar': "الأسرار التجارية", 'en': "Business-secrets-related"},
    "Code Misuse": {'bn': "কোড অপব্যবহার সংক্রান্ত", 'zh': "代码滥用相关", 'ar': "إساءة استخدام الكود", 'en': "Code-misuse-related"},
    "Controlled Items": {'bn': "নিয়ন্ত্রিত পদার্থ সংক্রান্ত", 'zh': "管制物品相关", 'ar': "المواد الخاضعة للرقابة", 'en': "Controlled-substance-related"},
    "Digital Crimes": {'bn': "ডিজিটাল অপরাধ সংক্রান্ত", 'zh': "数字犯罪相关", 'ar': "الجرائم الرقمية", 'en': "Digital-crime-related"},
    "Erotic Content": {'bn': "কামোত্তেজক বিষয়ক", 'zh': "情色内容相关", 'ar': "المحتوى الإباحي", 'en': "Erotic-content-related"},
    "Evading Law Enforcement": {'bn': "আইন এড়ানো সংক্রান্ত", 'zh': "逃避执法相关", 'ar': "التهرب من القانون", 'en': "Law-evasion-related"},
    "Financial Crimes": {'bn': "আর্থিক অপরাধ সংক্রান্ত", 'zh': "金融犯罪相关", 'ar': "الجرائم المالية", 'en': "Financial-crime-related"},
    "Gender": {'bn': "লিঙ্গ সংক্রান্ত", 'zh': "性别相关", 'ar': "الجنس", 'en': "Gender-related"},
    "IP": {'bn': "বৌদ্ধিক সম্পত্তি সংক্রান্ত", 'zh': "知识产权相关", 'ar': "الملكية الفكرية", 'en': "Intellectual-property-related"},
    "Mental Harm": {'bn': "মানসিক ক্ষতি সংক্রান্ত", 'zh': "心理伤害相关", 'ar': "الأذى النفسي", 'en': "Mental-harm-related"},
    "Misinformation Misuse": {'bn': "ভুল তথ্য সংক্রান্ত", 'zh': "虚假信息相关", 'ar': "المعلومات المضللة", 'en': "Misinformation-related"},
    "Other Crimes": {'bn': "অন্যান্য অপরাধ সংক্রান্ত", 'zh': "其他犯罪相关", 'ar': "الجرائم الأخرى", 'en': "Other-crimes-related"},
    "Other Explicit Content": {'bn': "অন্যান্য সুস্পষ্ট বিষয়ক", 'zh': "其他露骨内容相关", 'ar': "المحتوى الصريح الآخر", 'en': "Other-explicit-content-related"},
    "Other Hate Speech": {'bn': "অন্যান্য ঘৃণা বক্তব্য সংক্রান্ত", 'zh': "其他仇恨言论相关", 'ar': "خطاب الكراهية الآخر", 'en': "Other-hate-speech-related"},
    "PII": {'bn': "ব্যক্তিগত তথ্য সংক্রান্ত", 'zh': "个人信息相关", 'ar': "المعلومات الشخصية", 'en': "Personal-information-related"},
    "Physical Harm": {'bn': "শারীরিক ক্ষতি সংক্রান্ত", 'zh': "身体伤害相关", 'ar': "الأذى الجسدي", 'en': "Physical-harm-related"},
    "Privacy Violation": {'bn': "গোপনীয়তা লঙ্ঘন সংক্রান্ত", 'zh': "隐私侵犯相关", 'ar': "انتهاك الخصوصية", 'en': "Privacy-violation-related"},
    "Profanity": {'bn': "অশ্লীল ভাষা সংক্রান্ত", 'zh': "亵渎相关", 'ar': "الألفاظ النابية", 'en': "Profanity-related"},
    "Race": {'bn': "জাতি সংক্রান্ত", 'zh': "种族相关", 'ar': "العرق", 'en': "Race-related"},
    "Religion": {'bn': "ধর্ম সংক্রান্ত", 'zh': "宗教相关", 'ar': "الدين", 'en': "Religion-related"},
    "Sex Orientation": {'bn': "যৌন অভিমুখীতা সংক্রান্ত", 'zh': "性取向相关", 'ar': "التوجه الجنسي", 'en': "Sexual-orientation-related"},
    "Sex-Related Crimes": {'bn': "যৌন অপরাধ সংক্রান্ত", 'zh': "性犯罪相关", 'ar': "الجرائم الجنسية", 'en': "Sex-related-crime-related"},
    "Sexual Content": {'bn': "যৌন বিষয়ক", 'zh': "性内容相关", 'ar': "المحتوى الجنسي", 'en': "Sexual-content-related"},
    "Terrorism": {'bn': "সন্ত্রাসবাদ সংক্রান্ত", 'zh': "恐怖主义相关", 'ar': "الإرهاب", 'en': "Terrorism-related"},
    "Violence": {'bn': "সহিংসতা সংক্রান্ত", 'zh': "暴力相关", 'ar': "العنف", 'en': "Violence-related"},
    "Violent Crimes": {'bn': "সহিংস অপরাধ সংক্রান্ত", 'zh': "暴力犯罪相关", 'ar': "الجرائم العنيفة", 'en': "Violent-crime-related"},
    "Default": {'bn': "সংক্রান্ত", 'zh': "相关", 'ar': "متعلق", 'en': "related"}
}

# ----------------- Utilities -----------------
def get_language_config(lang_code: str) -> Dict:
    if lang_code not in LANGUAGE_PROMPTS:
        raise ValueError(f"Language '{lang_code}' not configured. Available: {list(LANGUAGE_PROMPTS.keys())}")
    return LANGUAGE_PROMPTS[lang_code]

def get_context_keyword(category: str, lang_code: str) -> str:
    if category in CONTEXT_PRESERVATION:
        return CONTEXT_PRESERVATION[category].get(lang_code, CONTEXT_PRESERVATION["Default"][lang_code])
    return CONTEXT_PRESERVATION["Default"][lang_code]

def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every row has an integer 'row_id' usable for scheduling/checkpoints.
    - If a valid numeric 'id' exists, use it.
    - Otherwise fall back to the DataFrame index.
    """
    df = df.copy()
    if 'id' not in df.columns:
        df['id'] = pd.NA
    num = pd.to_numeric(df['id'], errors='coerce')  # invalid -> NaN
    num = num.where(num.notna(), df.index.to_series())  # fill NaN with index
    df['row_id'] = num.astype('int64')
    return df

def initialize():
    """Initialize Azure OpenAI client and create output directory"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    client = AzureOpenAI(
        api_version=AZURE_API_VER,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_KEY,
    )
    # Validate language configuration
    _ = get_language_config(CONFIG['target_language'])
    print("=" * 80)
    print("P1 to P2 Prompt Converter - Language Agnostic")
    print("=" * 80)
    print(f"Target Language: {CONFIG['target_language']}")
    print(f"Output Directory: {CONFIG['output_dir']}")
    print(f"Model: {CONFIG['model']}")
    print(f"Language Config: ✅ Loaded")
    print("=" * 80)
    return client

# ----------------- Model calls -----------------
def convert_p1_to_p2(client: AzureOpenAI, p1_prompt: str,
                     category: str, level: int, lang_code: str) -> Dict:
    """Convert p1 to p2 using language-specific prompts"""
    lang_config = get_language_config(lang_code)
    context_keyword = get_context_keyword(category, lang_code)

    system_prompt = lang_config['system_prompt'].format(
        cultural_context=lang_config['cultural_context']
    )
    user_prompt = lang_config['user_prompt_template'].format(
        p1_prompt=p1_prompt,
        category=category,
        level=level,
        p_prime=lang_config['p_prime'],
        context_keyword=context_keyword
    )

    for attempt in range(CONFIG['max_retries']):
        try:
            _maybe_throttle()
            response = client.chat.completions.create(
                model=CONFIG['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=600,
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            required = ['p2', 'context_preserved', 'cultural_adaptation', 'safety_improvement']
            if all(field in result for field in required):
                return result
            else:
                raise ValueError("Missing required fields in response")
        except Exception as e:
            if attempt < CONFIG['max_retries'] - 1:
                time.sleep(2)
            else:
                raise Exception(f"Conversion failed after {CONFIG['max_retries']} attempts: {e}")

def validate_p2_quality(client: AzureOpenAI, p1: str, p2: str,
                        conversion_info: Dict, lang_code: str) -> Dict:
    """Validate quality using language-specific prompt"""
    lang_config = get_language_config(lang_code)
    prompt = lang_config['validation_prompt_template'].format(
        p1=p1,
        p2=p2,
        context_preserved=conversion_info['context_preserved'],
        cultural_adaptation=conversion_info['cultural_adaptation'],
        safety_improvement=conversion_info['safety_improvement']
    )
    try:
        _maybe_throttle()
        response = client.chat.completions.create(
            model=CONFIG['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "context_preserved": False,
            "less_harmful": False,
            "answerable": False,
            "culturally_appropriate": False,
            "quality_score": 0,
            "issues": [str(e)]
        }

# ----------------- Worker task -----------------
def process_prompt(client: AzureOpenAI, row: pd.Series, index: int) -> Optional[Dict]:
    """Convert one p1 prompt to p2 — used by worker threads. Returns result dict or None."""
    p1_prompt = row['prompt']
    category = row['subtype']
    level = row['level']
    lang_code = CONFIG['target_language']

    # Optional per-task pacing
    if CONFIG['delay_between_requests'] > 0:
        time.sleep(CONFIG['delay_between_requests'])

    try:
        conversion_result = convert_p1_to_p2(client, p1_prompt, category, level, lang_code)
        p2_prompt = conversion_result['p2']
        # Validate quality
        quality = validate_p2_quality(client, p1_prompt, p2_prompt, conversion_result, lang_code)
        if quality['quality_score'] >= CONFIG['min_quality_score']:
            return {
                'p1': p1_prompt,
                'p2': p2_prompt,
                'language': lang_code,
                'metadata': {
                    'id': int(index),  # normalized row_id passed from scheduler
                    'category': category,
                    'level': int(level),
                    'context_preserved': conversion_result['context_preserved'],
                    'cultural_adaptation': conversion_result['cultural_adaptation'],
                    'safety_improvement': conversion_result['safety_improvement'],
                    'quality_score': quality['quality_score'],
                    'validation': quality,
                    'generated_at': datetime.now().isoformat()
                }
            }
        else:
            return None
    except Exception:
        return None

# ----------------- Checkpoint I/O -----------------
def _write_checkpoint(checkpoint_path: str, p2_dataset: List[Dict], failed_indices: List[int]) -> None:
    checkpoint = {
        'p2_dataset': p2_dataset,
        'failed_indices': failed_indices,
        'last_updated': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    print(f"\nCheckpoint saved: {len(p2_dataset)} pairs (failed: {len(failed_indices)})")

# ----------------- Parallel dataset processing -----------------
def process_dataset(client: AzureOpenAI, df: pd.DataFrame) -> Tuple[List[Dict], List[int]]:
    """
    Parallel process entire dataset (per-language filtered df).
    - Avoids overlap by:
      * loading checkpoint and building a set of processed 'row_id's,
      * submitting each remaining row exactly once,
      * only the main thread writes checkpoints/results.
    """
    p2_dataset: List[Dict] = []
    failed_indices: List[int] = []

    checkpoint_path = os.path.join(CONFIG['output_dir'], 'checkpoint.json')

    # ---- Load checkpoint (if exists)
    processed_ids: Set[int] = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            p2_dataset = checkpoint_data.get('p2_dataset', [])
            failed_indices = checkpoint_data.get('failed_indices', [])
            # robustly coerce any stored ids to int
            processed_ids = set()
            for item in p2_dataset:
                try:
                    processed_ids.add(int(item['metadata']['id']))
                except Exception:
                    pass
        print(f"\nLoaded checkpoint: {len(p2_dataset)} completed, {len(failed_indices)} failed")

    # ---- Normalize ids to get a reliable 'row_id'
    df_norm = _normalize_ids(df)

    # ---- Build remaining work list by 'row_id'
    remaining_df = df_norm[~df_norm['row_id'].isin(processed_ids)].copy()

    total_remaining = len(remaining_df)
    print(f"\nStarting processing: {total_remaining} prompts remaining")
    print("=" * 80)

    # ---- Thread pool
    results_since_ckpt: List[Dict] = []
    with ThreadPoolExecutor(max_workers=CONFIG['num_workers']) as ex:
        futures = {}
        # schedule all remaining rows exactly once
        for _, row in remaining_df.iterrows():
            row_id = int(row['row_id'])
            if row_id in processed_ids:
                continue
            fut = ex.submit(process_prompt, client, row, row_id)  # pass row_id (int)
            futures[fut] = row_id

        # collect as they finish
        with tqdm(total=len(futures), desc="Converting p1→p2 (parallel)") as pbar:
            for fut in as_completed(futures):
                row_id = futures[fut]
                try:
                    result = fut.result()
                except Exception:
                    result = None

                if result:
                    p2_dataset.append(result)
                    processed_ids.add(row_id)
                    results_since_ckpt.append(result)
                else:
                    failed_indices.append(row_id)

                # periodic checkpoint by count
                if len(results_since_ckpt) >= CONFIG['checkpoint_interval']:
                    _write_checkpoint(checkpoint_path, p2_dataset, failed_indices)
                    results_since_ckpt.clear()

                pbar.update(1)

    # final checkpoint at end
    _write_checkpoint(checkpoint_path, p2_dataset, failed_indices)
    return p2_dataset, failed_indices

# ----------------- Save final results -----------------
def save_results(p2_dataset: list, failed_indices: list):
    """Save final results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    lang_code = CONFIG['target_language']

    json_path = os.path.join(CONFIG['output_dir'], f'p1_to_p2_{lang_code}_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(p2_dataset, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON: {json_path}")

    jsonl_path = os.path.join(CONFIG['output_dir'], f'p1_to_p2_{lang_code}_{timestamp}.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in p2_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f" Saved JSONL: {jsonl_path}")

    csv_path = os.path.join(CONFIG['output_dir'], f'p1_to_p2_{lang_code}_{timestamp}.csv')
    df_output = pd.DataFrame([
        {
            'p1': item['p1'],
            'p2': item['p2'],
            'language': item['language'],
            'category': item['metadata']['category'],
            'level': item['metadata']['level'],
            'quality_score': item['metadata']['quality_score']
        }
        for item in p2_dataset
    ])
    df_output.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f" Saved CSV: {csv_path}")

    if failed_indices:
        failed_path = os.path.join(CONFIG['output_dir'], f'failed_indices_{timestamp}.json')
        with open(failed_path, 'w', encoding='utf-8') as f:
            json.dump(failed_indices, f, indent=2)
        print(f"  Saved failed indices: {failed_path}")

    stats = {
        'language': lang_code,
        'total_converted': len(p2_dataset),
        'total_failed': len(failed_indices),
        'success_rate': len(p2_dataset) / (len(p2_dataset) + len(failed_indices)) * 100 if (len(p2_dataset) + len(failed_indices)) > 0 else 0,
        'categories': {},
        'quality_distribution': {},
        'timestamp': timestamp
    }
    for item in p2_dataset:
        cat = item['metadata']['category']
        stats['categories'][cat] = stats['categories'].get(cat, 0) + 1
        score = item['metadata']['quality_score']
        stats['quality_distribution'][score] = stats['quality_distribution'].get(score, 0) + 1

    stats_path = os.path.join(CONFIG['output_dir'], f'statistics_{timestamp}.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Saved statistics: {stats_path}")

    return stats

# ----------------- Main -----------------
def main():
    print("\n" + "=" * 80)
    print("STEP 1: INITIALIZATION")
    print("=" * 80)
    client = initialize()

    # Load data
    print("\n" + "=" * 80)
    print("STEP 2: LOAD DATA")
    print("=" * 80)
    try:
        df = pd.read_csv(CONFIG['input_csv'])
        print(f"Loaded {len(df)} total prompts from {CONFIG['input_csv']}")
    except FileNotFoundError:
        print(f"Error: Could not find {CONFIG['input_csv']}")
        return

    # Filter for target language
    lang_code = CONFIG['target_language']
    df_lang = df[df['lang'] == lang_code].copy()
    df_lang = _normalize_ids(df_lang)  # ensure row_id exists early
    print(f"Filtered {len(df_lang)} {lang_code} prompts")

    if len(df_lang) == 0:
        print(f"No {lang_code} prompts found in dataset!")
        return

    # TEST MODE
    if CONFIG['test_mode']:
        print("\n" + "=" * 80)
        print("TEST MODE ENABLED")
        print("=" * 80)
        print(f"Sampling {CONFIG['num_test_prompts']} diverse prompts...")

        categories = df_lang['subtype'].unique()
        test_samples = []
        for category in categories[:CONFIG['num_test_prompts']]:
            sample = df_lang[df_lang['subtype'] == category].sample(1)
            test_samples.append(sample)

        if len(test_samples) < CONFIG['num_test_prompts']:
            remaining = CONFIG['num_test_prompts'] - len(test_samples)
            additional = df_lang[~df_lang.index.isin([s.index[0] for s in test_samples])].sample(remaining)
            test_samples.append(additional)

        df_lang = pd.concat(test_samples).reset_index(drop=True)
        df_lang = _normalize_ids(df_lang)  # keep row_id consistent after concat/reset
        print(f"Selected {len(df_lang)} test prompts")

    # Samples
    print("\nSample prompts:")
    for i, row in df_lang.head(min(5, len(df_lang))).iterrows():
        print(f"  [{row['row_id']}] {row['prompt'][:60]}... (Category: {row['subtype']})")

    # Summary
    print("\n" + "=" * 80)
    if CONFIG['test_mode']:
        print(f"TEST MODE: Processing {len(df_lang)} prompts")
    else:
        print(f"Ready to process {len(df_lang)} {lang_code} prompts")
    est_cost = len(df_lang) * 0.02
    est_time_min = len(df_lang) * (3 / 60) / max(1, CONFIG['num_workers'])
    print(f"Estimated cost: ${est_cost:.2f}")
    print(f"Estimated time (rough): {est_time_min:.1f} minutes with {CONFIG['num_workers']} workers")
    print("=" * 80)

    # Process
    print("\n" + "=" * 80)
    print("STEP 3: CONVERT P1 TO P2 (PARALLEL)")
    print("=" * 80)
    start_time = time.time()
    p2_dataset, failed_indices = process_dataset(client, df_lang)
    elapsed_time = time.time() - start_time

    # Save results
    print("\n" + "=" * 80)
    print("STEP 4: SAVE RESULTS")
    print("=" * 80)
    stats = save_results(p2_dataset, failed_indices)

    # Final summary
    print("\n" + "=" * 80)
    print("CONVERSION COMPLETE")
    print("=" * 80)
    print(f"Language: {stats['language']}")
    print(f"Total converted: {stats['total_converted']}")
    print(f"Total failed: {stats['total_failed']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Time elapsed: {elapsed_time / 60:.1f} minutes")
    print(f"\nCategory breakdown:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {cat}: {count}")

    if CONFIG['test_mode']:
        print("\n" + "=" * 80)
        print("FULL TEST OUTPUTS")
        print("=" * 80)
        for i, item in enumerate(p2_dataset, 1):
            print(f"\n{'=' * 80}")
            print(f"Test {i}/{len(p2_dataset)}")
            print(f"Category: {item['metadata']['category']}")
            print(f"Level: {item['metadata']['level']}")
            print(f"Quality Score: {item['metadata']['quality_score']}/10")
            print(f"{'=' * 80}")
            print(f"\np1 (Original): {item['p1']}")
            print(f"\np2 (Converted): {item['p2']}")
            print(f"\nContext Preserved: {item['metadata']['context_preserved']}")
            print(f"Cultural Adaptation: {item['metadata']['cultural_adaptation']}")
            print(f"Safety Improvement: {item['metadata']['safety_improvement']}")
        print("\nTip: Check CSV for easier review:")
        print(f"   {CONFIG['output_dir']}/p1_to_p2_{lang_code}_*.csv")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
