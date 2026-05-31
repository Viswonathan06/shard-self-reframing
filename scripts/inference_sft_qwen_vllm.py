#!/usr/bin/env python3
"""
vLLM inference for the Qwen3.5-9B SFT LoRA adapter.

Input JSONL must have a `p1` field (the user prompt).
Outputs a JSONL with `sft_raw_output` (full generation including <think> block)
and `sft_output` (final answer after </think>, or full text if no think block).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:
    sys.exit(f"vLLM not found: {e}")

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL_DEFAULT = (
    "/playpen/huggingface/hub/models--Qwen--Qwen3.5-9B"
    "/snapshots/c202236235762e1c871ad0ccb60c8ee5ba327b9a"
)
SFT_ADAPTER_DEFAULT = str(
    ROOT / "output/SFT/qwen35_122b_teacher/qwen35_9b/model"
)
INPUT_DEFAULT = str(
    ROOT / "output/SFT/qwen35_122b_teacher/sft_reasoning/test.jsonl"
)
OUTPUT_DEFAULT = str(
    ROOT / "output/SFT/qwen35_122b_teacher/qwen35_9b/inference.jsonl"
)


def load_records(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_THINKING_PREFIXES = (
    "thinking process:",
    "here's a thinking process",
    "here is a thinking process",
    "let me think through",
    "let me work through",
)

# Italic meta-markers the 122B teacher inserts between reasoning steps.
# Paragraphs matching this are skipped; content after ":" on the same line is kept.
_META_MARKER_RE = re.compile(
    r"^\s*\*+\s*("
    r"thinking process|draft(?:ing)?(?: response| the response)?|"
    r"okay[,.]?\s*final|final (?:decision|version|answer|response|plan)|"
    r"wait[,. ]|revised?\s*(?:version|response|draft|plan)?|"
    r"expanding for|version [ab]|option [ab]|my response|note:"
    r")[^*\n]*\*+[:\.]?\s*",
    re.IGNORECASE,
)


def _is_meta_only(para: str) -> bool:
    """True if the paragraph is entirely a meta-marker with no real content after it."""
    first_line = para.lstrip().split("\n")[0]
    m = _META_MARKER_RE.match(first_line)
    if not m:
        return False
    after = first_line[m.end():].strip()
    # If there's substantial content after the marker on the same line, keep it
    return len(after.split()) < 8


def _extract_after_meta(para: str) -> str:
    """If paragraph starts with a meta-marker followed by real content, return that content."""
    first_line = para.lstrip().split("\n")[0]
    m = _META_MARKER_RE.match(first_line)
    if not m:
        return ""
    after = first_line[m.end():].strip()
    rest_lines = para.split("\n")[1:]
    rest = "\n".join(rest_lines).strip()
    combined = (after + ("\n" + rest if rest else "")).strip()
    return combined if len(combined.split()) >= 8 else ""


def extract_final_answer(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # Hard separator used by rational-SFT structured prompt format
    if "### Response:" in t:
        return t[t.index("### Response:") + len("### Response:"):].strip()

    # XML-style thinking tags: <think>, <scratchpad>, <reasoning>
    for tag in ("think", "scratchpad", "reasoning"):
        end_tag = f"</{tag}>"
        idx = t.lower().find(end_tag)
        if idx != -1:
            return t[idx + len(end_tag):].strip()

    # Generic single XML block fallback
    m = re.match(r"^\s*<([a-zA-Z_][\w\-]*)>.*?</\1>\s*(.*)$", t, flags=re.DOTALL)
    if m:
        return (m.group(2) or "").strip()

    # "Thinking Process:" / "Here's a thinking process" style (no XML tags).
    lower_t = t.lower()
    if any(lower_t.startswith(p) for p in _THINKING_PREFIXES):
        paragraphs = re.split(r"\n{2,}", t)
        candidates = []
        for para in paragraphs:
            first = para.lstrip().split("\n")[0]
            # Skip thinking-prefix headers
            if any(first.lower().startswith(p) for p in _THINKING_PREFIXES):
                continue
            # Skip numbered/bulleted reasoning steps
            if re.match(r"^\s*(\d+[\.\)]\s|\*\s|-\s|[A-Z]\)\s)", first):
                continue
            # Meta-only markers: try to salvage content after the colon
            if _is_meta_only(para):
                salvaged = _extract_after_meta(para)
                if salvaged:
                    candidates.append(salvaged)
                continue
            candidates.append(para.strip())

        # Return the last candidate with >15 words (avoids late "wait" snippets)
        substantial = [c for c in candidates if len(c.split()) > 15]
        if substantial:
            return substantial[-1]
        if candidates:
            return candidates[-1]

    return t


def extract_turns(r: dict) -> tuple[str, str, str]:
    """Return (system, user_prompt, reference_answer) from a record.

    Supports two formats:
      - messages list: [system, user, assistant]
      - flat fields: p1 (+ optional p2 as reference)
    """
    msgs = r.get("messages")
    if msgs:
        system = next((m["content"] for m in msgs if m["role"] == "system"), "")
        user   = next((m["content"] for m in msgs if m["role"] == "user"),   "")
        ref    = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        return system.strip(), user.strip(), ref.strip()
    return "", (r.get("p1") or "").strip(), (r.get("p2") or "").strip()


def build_prompts(records: list[dict], tokenizer, system_prompt: str = "",
                  enable_thinking: bool = False) -> list[str]:
    prompts = []
    extra_kwargs = {"enable_thinking": True} if enable_thinking else {}
    for r in records:
        system, user, _ = extract_turns(r)
        effective_system = system_prompt or system
        msgs = []
        if effective_system:
            try:
                msgs.append({"role": "system", "content": effective_system})
                tokenizer.apply_chat_template(msgs + [{"role": "user", "content": "test"}],
                                              tokenize=False, add_generation_prompt=True)
            except Exception:
                user = effective_system + "\n\n" + user
                msgs = []
        msgs.append({"role": "user", "content": user})
        prompts.append(
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **extra_kwargs
            )
        )
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=BASE_MODEL_DEFAULT)
    ap.add_argument("--sft-adapter", default="",
                    help="Path to LoRA adapter. If empty, runs in baseline mode (no LoRA, outputs baseline_output field).")
    ap.add_argument("--input-data", default=INPUT_DEFAULT)
    ap.add_argument("--output", default=OUTPUT_DEFAULT)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tensor-parallel-size", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--system-prompt", default="",
                    help="Override system prompt at inference. Use the same prompt as training "
                         "to avoid distribution mismatch (especially important for Mistral).")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Pass enable_thinking=True to apply_chat_template (Qwen3.5 thinking models).")
    ap.add_argument("--tokenizer-mode", default="auto",
                    help="vLLM tokenizer mode: 'auto' (default) or 'mistral' for mistral_common models.")
    ap.add_argument("--output-key", default="",
                    help="If set, write the final response under this key instead of sft_output/response. "
                         "Use 'self_output' for self-model runs and 'large_model_output' for teacher runs.")
    args = ap.parse_args()

    baseline_mode = not args.sft_adapter

    records = load_records(Path(args.input_data))
    if args.max_records > 0:
        records = records[: args.max_records]
    records = [r for r in records if extract_turns(r)[1]]
    print(f"Prompts: {len(records)}")
    print(f"Mode:    {'baseline (no adapter)' if baseline_mode else f'SFT adapter: {args.sft_adapter}'}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    prompts = build_prompts(records, tokenizer, system_prompt=args.system_prompt,
                            enable_thinking=args.enable_thinking)
    print("Example prompt (first 200 chars):")
    print(prompts[0][:200].replace("\n", "\\n"))
    print(f"\nBase model:  {args.base_model}")

    llm_kwargs = dict(
        model=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_custom_all_reduce=True,
        tokenizer_mode=args.tokenizer_mode,
        limit_mm_per_prompt={"image": 0},
    )
    if not baseline_mode:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.lora_rank

    llm = LLM(**llm_kwargs)

    lora_request = None
    if not baseline_mode:
        lora_request = LoRARequest(
            lora_name="sft_adapter",
            lora_int_id=1,
            lora_path=str(Path(args.sft_adapter).resolve()),
        )

    stop_ids = [tokenizer.eos_token_id]
    for tok in ("<|im_end|>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid != tokenizer.unk_token_id:
            stop_ids.append(tid)
    stop_ids = list(set(filter(None, stop_ids)))

    sampling_params = SamplingParams(
        temperature=max(0.0, args.temperature),
        max_tokens=args.max_new_tokens,
        stop_token_ids=stop_ids,
    )

    print(f"\nGenerating {len(prompts)} responses...")
    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)
    print("Done.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for rec, out in zip(records, outputs):
            _, user, ref = extract_turns(rec)
            raw = (out.outputs[0].text or "").strip()
            # Strip Mistral BOS/EOS wrappers that vLLM includes in raw output
            # when tokenizer_mode=mistral; the XML fallback in extract_final_answer
            # matches <s>...</s> as a tag and returns the empty string after it.
            raw_clean = re.sub(r"^<s>", "", raw)
            if raw_clean.endswith("</s>"):
                raw_clean = raw_clean[:-4]
            raw_clean = raw_clean.strip()
            final = extract_final_answer(raw_clean)
            if baseline_mode:
                row = {
                    "p1_id":           str(rec.get("p1_id", "")),
                    "p1":              user,
                    "category":        rec.get("category", ""),
                    "level":           rec.get("level", ""),
                    "baseline_output": final,
                }
            elif args.output_key:
                row = {
                    "p1_id":             str(rec.get("p1_id", "")),
                    "p1":                user,
                    "category":          rec.get("category", ""),
                    "level":             rec.get("level", ""),
                    args.output_key:     final,
                }
                # self_output doubles as baseline_output so the judge can use this
                # file directly as --baseline; large_model_output doubles as response
                # so the judge can use it as --sft-responses.
                if args.output_key == "self_output":
                    row["baseline_output"] = final
                else:
                    row["response"] = final
            else:
                row = {
                    "p1_id":           str(rec.get("p1_id", "")),
                    "p1":              user,
                    "reference":       ref,
                    "category":        rec.get("category", ""),
                    "level":           rec.get("level", ""),
                    "sft_raw_output":  raw,
                    "sft_output":      final,
                    "response":        final,
                }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows -> {out_path}")


if __name__ == "__main__":
    main()
