#!/usr/bin/env python3
"""
Run inference on the scratchpad SFT adapter (Llama 3.1 8B + LoRA).

How this relates to training (see scripts/sft_final.py + data/sft_scratchpad):
  - The adapter was trained so that after "### Human: {p1}" "### Assistant: " the
    model continues with structured analysis then the user-facing answer.
  - Training labels use <scratchpad>...</scratchpad> around the STEPs (harm,
    legitimate need, minimal restriction, caring engagement). You do **not**
    need a long instruction telling it to "do CoT": that format is already the
    learned habit **if** the prompt prefix matches training (### Human/### Assistant,
    no chat-template mismatch).

  - If you want the *XML delimiter* to read as <reasoning> instead of <scratchpad>,
    the reliable fix is to **change the training JSONL** (same STEPs, different
    tags) and **re-run SFT**. At inference-only, we still parse <scratchpad> first
    (what weights know), then <reasoning> if present.

  - SFT biases toward the Qwen teacher's helpful-safe O2 style; plain refusals can
    still appear if decoding/format is off-distribution, temperature is high, or
    the human turn differs a lot from the short P1-only training text.
"""

import json
import torch
import argparse
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def load_sft_model(base_model_path, sft_adapter_path):
    """Load the base model and apply SFT adapter."""
    
    print(f"🤖 Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"🔧 Loading SFT adapter from: {sft_adapter_path}")
    model = PeftModel.from_pretrained(base_model, sft_adapter_path)
    
    print(f"📝 Loading tokenizer from: {sft_adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter_path)
    
    return model, tokenizer


def load_p1_baseline_map(jsonl_path: Path) -> dict:
    """
    Llama 8B direct P1 → O1 lines from DNA baseline multimodel (`output` field).
    Keyed by str(p1_id) for joining onto SFT rows.
    """
    m = {}
    if not jsonl_path.is_file():
        return m
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = str(r.get("p1_id", ""))
            out = r.get("output")
            if pid and out is not None:
                m[pid] = out
    return m


def load_guidelines(guidelines_dir, category):
    """Load universal and category-specific guidelines."""
    guidelines_path = Path(guidelines_dir)
    
    # Load universal guidelines
    universal_file = guidelines_path / "universal_guidelines.txt"
    if universal_file.exists():
        with open(universal_file, 'r') as f:
            universal_guidelines = f.read().strip()
    else:
        universal_guidelines = "Apply general safety and helpfulness principles."
    
    # Convert category name to filename format
    category_filename = category.lower().replace(' & ', '_and_').replace(' ', '_') + '_guidelines.txt'
    category_file = guidelines_path / category_filename
    
    if category_file.exists():
        with open(category_file, 'r') as f:
            category_guidelines = f.read().strip()
    else:
        category_guidelines = f"Apply category-specific safety guidelines for {category}."
    
    return universal_guidelines, category_guidelines


def create_inference_prompt(
    p1,
    p2,
    universal_guidelines,
    category_guidelines,
    category,
    include_p2_and_guidelines: bool = False,
):
    """Build the *human* block text. Full training string is ### Human: ... ### Assistant:"""

    if not include_p2_and_guidelines:
        return p1.strip()

    # Context only — no long "how to think" instructions; the adapter already learned
    # the STEP / tag structure from SFT (distribution shift if this block is long).
    parts = [
        f"Original user request (P1):\n{p1.strip()}",
        f"\nSafer rewritten request (P2):\n{p2.strip()}",
        f"\nCategory: {category}",
        f"\nUniversal guidelines:\n{universal_guidelines.strip()}",
        f"\nCategory-specific guidelines:\n{category_guidelines.strip()}",
    ]
    return "\n".join(parts)


def build_sft_training_style_prefix(human_block: str) -> str:
    """Must match scripts/sft_final.py ScratchpadDataset: ### Human: ... ### Assistant:"""
    return f"### Human: {human_block}\n\n### Assistant: "


def generate_response(
    model,
    tokenizer,
    human_block: str,
    max_new_tokens=2048,
    temperature=0.3,
    use_chat_template: bool = False,
):
    """Generate continuation after '### Assistant: '. Default matches SFT training (no chat template)."""

    full_prompt = build_sft_training_style_prefix(human_block)

    if use_chat_template:
        messages = [{"role": "user", "content": full_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        formatted_prompt = full_prompt

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature and temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    # Decode response (only the new tokens)
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    return response


def _clean_reasoning_body(raw_inner: str, bare_tag: str) -> str:
    lines_out = []
    for line in raw_inner.split('\n'):
        line = line.strip()
        if line in (f'<{bare_tag}>', f'</{bare_tag}>'):
            continue
        if line.startswith('**Improved Response:**') or line.startswith(
            'RESPONSE:'
        ) or line.startswith('**Response:**'):
            break
        lines_out.append(line)
    return '\n'.join(lines_out).strip()


def _strip_response_prefix(response: str) -> str:
    for prefix in (
        '**Improved Response:**',
        '**Response:**',
        'RESPONSE:',
        'Response:',
    ):
        if response.startswith(prefix):
            response = response[len(prefix) :].strip()
            break
    return response


def _extract_closed_tag_block(text: str, tag: str):
    """If <tag>...</tag> exists, return (inner_reasoning, tail_after_close)."""
    open_t, close_t = f'<{tag}>', f'</{tag}>'
    if open_t not in text or close_t not in text:
        return None
    start = text.find(open_t) + len(open_t)
    end = text.find(close_t)
    if end <= start:
        return None
    inner = _clean_reasoning_body(text[start:end], tag)
    tail = text[end + len(close_t) :].strip()
    response_lines = [ln.strip() for ln in tail.split('\n') if ln.strip()]
    response = _strip_response_prefix('\n'.join(response_lines))
    return inner, response


def _extract_tag_then_improved(text: str, tag: str):
    open_t = f'<{tag}>'
    if open_t not in text or '**Improved Response:**' not in text:
        return None
    start = text.find(open_t) + len(open_t)
    cut = text.find('**Improved Response:**')
    if cut <= start:
        return None
    inner = _clean_reasoning_body(text[start:cut], tag)
    response = text[cut + len('**Improved Response:**') :].strip()
    return inner, response


def extract_reasoning_and_response(generated_text: str):
    """
    Split structured reasoning from the final user-facing answer.

    Tries, in order:
      1) <scratchpad>...</scratchpad> (matches current SFT training)
      2) <reasoning>...</reasoning> (use after you re-tag training data + re-SFT)
      3) Same tags but **Improved Response:** instead of closing tag (legacy)
      4) Untagged STEP 1–4 heuristics
    Returns (reasoning_text, final_response).
    """
    text = generated_text.strip()

    for tag in ('scratchpad', 'reasoning'):
        got = _extract_closed_tag_block(text, tag)
        if got is not None:
            return got

    for tag in ('scratchpad', 'reasoning'):
        got = _extract_tag_then_improved(text, tag)
        if got is not None:
            return got

    if 'STEP 1 - HARM ASSESSMENT' in text:
        lines = text.split('\n')
        scratchpad_lines = []
        response_lines = []
        in_scratchpad = True

        for line in lines:
            line = line.strip()

            if in_scratchpad:
                if (
                    line.startswith('**Improved Response:**')
                    or line.startswith('RESPONSE:')
                    or line.startswith('**Response:**')
                    or line.startswith("Since you're interested")
                    or line.startswith("If you're looking")
                    or line.startswith("If you're interested")
                    or line.startswith("While I can't")
                    or line.startswith('The topic of')
                ):
                    in_scratchpad = False
                    if not any(
                        line.startswith(p)
                        for p in (
                            '**Improved Response:**',
                            'RESPONSE:',
                            '**Response:**',
                        )
                    ):
                        response_lines.append(line)
                else:
                    scratchpad_lines.append(line)
            else:
                if not any(
                    line.startswith(p)
                    for p in (
                        '**Improved Response:**',
                        'RESPONSE:',
                        '**Response:**',
                    )
                ):
                    response_lines.append(line)

        return (
            '\n'.join(scratchpad_lines).strip(),
            '\n'.join(response_lines).strip(),
        )

    return 'No structured reasoning detected', text


def extract_scratchpad_and_response(generated_text):
    """Backward-compatible name for extract_reasoning_and_response."""
    return extract_reasoning_and_response(generated_text)


def main():
    if not HAS_TRANSFORMERS:
        print("❌ Required packages not available")
        return
    
    parser = argparse.ArgumentParser(description="Run inference with SFT model")
    parser.add_argument('--base-model', 
                       default="$HF_HOME/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
                       help='Path to base Llama model')
    parser.add_argument('--sft-adapter', 
                       default="models/llama31_full_sft",
                       help='Path to SFT adapter')
    parser.add_argument('--input-data',
                       default="output/DNA_New_Experiments/dna_refinement_from_baseline_multimodel/llama8b/baseline_outputs.jsonl",
                       help='Input JSONL with at least p1 (and usually p2, category) per line')
    parser.add_argument('--guidelines-dir',
                       default="src/guidelines",
                       help='Directory containing guideline files')
    parser.add_argument('--output',
                       default="output/sft_inference_results.jsonl",
                       help='Output file for results')
    parser.add_argument(
        '--max-records',
        type=int,
        default=10,
        help='Max rows to process from input JSONL. Use 0 for entire file.',
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Max new tokens per completion (scratchpad+answer can be long).',
    )
    parser.add_argument(
        '--include-p2-and-guidelines',
        action='store_true',
        help='Put P2 + guidelines in the human turn (OOD vs P1-only SFT; try if you need context).',
    )
    parser.add_argument(
        '--use-chat-template',
        action='store_true',
        help='Wrap in Llama chat template (usually WRONG for this LoRA; training used ### Human/### Assistant).',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Sampling temperature (0 = greedy). Default 0.3 to align loosely with refinement generation jobs.',
    )
    parser.add_argument(
        '--p1-baseline-jsonl',
        type=str,
        default="output/DNA_New_Experiments/dna_baseline_multimodel/llama8b/p1_baseline_outputs.jsonl",
        help='Llama 8B P1-only baseline JSONL (field `output`) to attach as p1_baseline_output for side-by-side comparison.',
    )

    args = parser.parse_args()

    p1_baseline_path = Path(args.p1_baseline_jsonl)
    p1_baseline_map = load_p1_baseline_map(p1_baseline_path)
    if p1_baseline_map:
        print(f"📎 Loaded {len(p1_baseline_map)} P1 baseline rows from {p1_baseline_path}")
    else:
        print(f"⚠️ No P1 baseline map loaded (missing or empty): {p1_baseline_path}")
    
    print("🚀 SFT Model Inference")
    print("=" * 40)
    print(f"📦 Base model: {args.base_model}")
    print(f"🔧 SFT adapter: {args.sft_adapter}")
    print(f"📚 Input data: {args.input_data}")
    print(f"📄 Output: {args.output}")
    print(
        f"🧩 Prompt: P1-only human turn (match SFT): {not args.include_p2_and_guidelines}"
    )
    print(f"🧩 max_new_tokens={args.max_new_tokens} temperature={args.temperature}")

    # Load model
    print("\n🤖 Loading SFT model...")
    model, tokenizer = load_sft_model(args.base_model, args.sft_adapter)
    
    input_path = Path(args.input_data)
    with open(input_path, "r", encoding="utf-8") as _lc:
        total_lines = sum(1 for ln in _lc if ln.strip())
    limit = args.max_records if args.max_records > 0 else total_lines
    limit = min(limit, total_lines)

    print(
        f"\n📊 Processing {limit}/{total_lines} rows from {input_path} "
        f"(streaming to {args.output})..."
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    success_count = 0
    error_count = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for i, line in enumerate(fin):
            if i >= limit:
                break
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            p1 = record['p1']
            p2 = record.get('p2', '')
            category = record.get('category', 'Unknown')

            verbose = i < 2 or (i + 1) % 50 == 0 or i + 1 == limit
            if verbose:
                print(f"\n📝 [{i + 1}/{limit}] {category} | P1: {p1[:70]}...")

            if args.include_p2_and_guidelines:
                universal_guidelines, category_guidelines = load_guidelines(
                    args.guidelines_dir, category
                )
            else:
                universal_guidelines, category_guidelines = '', ''

            human_block = create_inference_prompt(
                p1,
                p2,
                universal_guidelines,
                category_guidelines,
                category,
                include_p2_and_guidelines=args.include_p2_and_guidelines,
            )

            try:
                generated_text = generate_response(
                    model,
                    tokenizer,
                    human_block,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    use_chat_template=args.use_chat_template,
                )
                reasoning, final_response = extract_reasoning_and_response(
                    generated_text
                )

                if verbose:
                    print(
                        f"   ✅ raw {len(generated_text)} | "
                        f"reasoning {len(reasoning)} | answer {len(final_response)}"
                    )

                result = {
                    'p1_id': record.get('p1_id', str(i)),
                    'p1': p1,
                    'p2': p2,
                    'category': category,
                    'sft_generated_reasoning': reasoning,
                    'sft_generated_scratchpad': reasoning,
                    'sft_generated_response': final_response,
                    'raw_generation': generated_text,
                    'model': 'llama31_sft_trained',
                    'inference_p1_only': not args.include_p2_and_guidelines,
                }
                # O1 copied on refinement rows (same Llama 8B P1 answer, if pipeline stored it)
                if 'baseline_output' in record and record['baseline_output'] is not None:
                    result['baseline_output'] = record['baseline_output']
                # Canonical multimodel P1 → baseline `output` for apples-to-apples compare vs SFT
                _pid = str(record.get('p1_id', i))
                if _pid in p1_baseline_map:
                    result['p1_baseline_output'] = p1_baseline_map[_pid]
                if args.include_p2_and_guidelines:
                    result['universal_guidelines'] = (
                        universal_guidelines[:200] + '...'
                        if len(universal_guidelines) > 200
                        else universal_guidelines
                    )
                    result['category_guidelines'] = (
                        category_guidelines[:200] + '...'
                        if len(category_guidelines) > 200
                        else category_guidelines
                    )

                fout.write(json.dumps(result) + '\n')
                fout.flush()
                success_count += 1

            except Exception as e:
                print(f"   ❌ Error: {e}")
                err = {
                    'p1_id': record.get('p1_id', str(i)),
                    'p1': p1,
                    'p2': p2,
                    'category': category,
                    'error': str(e),
                }
                fout.write(json.dumps(err) + '\n')
                fout.flush()
                error_count += 1

    print(f"\n🎉 Inference completed!")
    print(f"✅ Successful: {success_count} | ❌ Errors: {error_count} | Wrote: {out_path}")


if __name__ == '__main__':
    main()