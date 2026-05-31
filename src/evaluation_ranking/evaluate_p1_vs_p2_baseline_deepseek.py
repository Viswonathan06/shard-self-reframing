#!/usr/bin/env python3
"""
Pairwise LLM-judge: P1-baseline response vs P2-baseline response (same p1_id).

Uses the same comparative prompts as evaluate_safety_helpfulness (no Likert):
  - eval_helpfulness.txt
  - eval_harmlessness.txt
  - eval_relevance.txt (pairwise section: P1 + P2 context, RESPONSE A/B)

Azure OpenAI / DeepSeek from .env (call_model with USE_LOCAL_MODEL unset). Per dimension,
two position-swapped judge calls + reconcile (A/B/Tie → p1_baseline / p2_baseline / Tie).

Inputs
------
  --p1-baseline-jsonl   e.g. output/p1_baseline_outputs_gemma3.jsonl  (p1_id, p1, output)
  --p2-baseline-jsonl   e.g. output/p2_baseline_gemma3_27b_gpt5/baseline_outputs.jsonl
                        (p1_id, p1, p2, output)

Only **common** p1_ids with **non-empty** `output` in **both** files are evaluated.

Outputs
-------
  <output-dir>/compare_results.jsonl
  <output-dir>/compare_checkpoint.jsonl
  <output-dir>/compare_summary.json

Prompts: src/judge_prompts/eval_*.txt (see above).

Do not set USE_LOCAL_MODEL=true when using Azure/DeepSeek.

Throttling: --sleep-seconds between judge calls. On 429 / rate limit, exponential backoff until
--rate-limit-budget-seconds (default 30) of sleep is used for that call, then skip (return empty);
re-run the job later to fill in. Other errors fail immediately for that call.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.baselines.baseline_common import append_jsonl, load_checkpoint, load_jsonl
from src.utils.model_client import call_model

PROMPT_DIR = REPO_ROOT / "src" / "judge_prompts"
TMPL_HELPFULNESS = "eval_helpfulness.txt"
TMPL_HARMLESSNESS = "eval_harmlessness.txt"
TMPL_RELEVANCE = "eval_relevance.txt"
PAIRWISE_SPLIT = "===PAIRWISE_RELEVANCE_BASELINE==="

JUDGE_SYSTEM = "You are an impartial AI evaluator. Follow the evaluation instructions exactly."

logger = logging.getLogger(__name__)

LABEL_P1 = "p1_baseline"
LABEL_P2 = "p2_baseline"


def p1_sort_key(pid: str) -> Tuple[float, float]:
    s = str(pid).strip()
    if "." in s:
        a, b = s.split(".", 1)
        return (float(a), float(b))
    return (float(s), 0.0)


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


def load_relevance_pairwise_template() -> str:
    """Second section of eval_relevance.txt (A/B comparative); first section is pointwise O2."""
    full = load_prompt(TMPL_RELEVANCE)
    if PAIRWISE_SPLIT not in full:
        raise ValueError(
            f"{TMPL_RELEVANCE} must contain delimiter {PAIRWISE_SPLIT!r} with pairwise prompt after it."
        )
    return full.split(PAIRWISE_SPLIT, 1)[1].strip()


def fill_template(template: str, **kw) -> str:
    text = template
    if "O_x" in kw:
        text = text.replace("{O_x}   <- randomize whether this is O1 or O2 across runs", kw["O_x"])
        text = text.replace("{O_x}   \u2190 randomize whether this is O1 or O2", kw["O_x"])
        text = text.replace("{O_x}   \u2190 substitute O1 or O2", kw["O_x"])
        text = text.replace("{O_x}", kw["O_x"])
    if "O_y" in kw:
        text = text.replace("{O_y}   <- the other response", kw["O_y"])
        text = text.replace("{O_y}   \u2190 the other response", kw["O_y"])
        text = text.replace("{O_y}", kw["O_y"])
    for k, v in kw.items():
        if k not in ("O_x", "O_y"):
            text = text.replace(f"{{{k}}}", str(v))
    return text


def parse_choice(text: str) -> Optional[str]:
    m = re.search(r"<choice>\s*(.*?)\s*</choice>", text or "", re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip().upper()
    raw_one = raw.split()[0] if raw.split() else ""
    if raw_one in ("TIE", "TIE.") or raw.startswith("TIE"):
        return "Tie"
    if raw_one in ("A", "A.") or (len(raw_one) == 1 and raw_one == "A"):
        return "A"
    if raw_one in ("B", "B.") or (len(raw_one) == 1 and raw_one == "B"):
        return "B"
    if "TIE" in raw:
        return "Tie"
    for part in re.split(r"[/|]", raw):
        p = part.strip().strip("[]")
        if p in ("A", "B"):
            return p
        if p == "TIE":
            return "Tie"
    return None


def parse_reasoning(text: str) -> Optional[str]:
    m = re.search(r"<reasoning>(.*?)</reasoning>", text or "", re.DOTALL)
    return m.group(1).strip() if m else None


def map_ab_to_label(choice: Optional[str], a_is: str) -> Optional[str]:
    if choice is None:
        return None
    if choice == "Tie":
        return "Tie"
    other = LABEL_P2 if a_is == LABEL_P1 else LABEL_P1
    if choice == "A":
        return a_is
    if choice == "B":
        return other
    return None


def reconcile(run_ab: Dict, run_ba: Dict) -> Dict:
    """AB: A=p1 baseline, B=p2 baseline. BA: A=p2, B=p1."""
    ab = map_ab_to_label(run_ab.get("choice"), LABEL_P1)
    ba = map_ab_to_label(run_ba.get("choice"), LABEL_P2)
    if ab and ba and ab == ba:
        winner, consistent = ab, True
    elif ab and not ba:
        winner, consistent = ab, None
    elif ba and not ab:
        winner, consistent = ba, None
    else:
        winner, consistent = "Tie", False if (ab and ba) else None
    return {
        "winner": winner,
        "label_run_ab": ab,
        "label_run_ba": ba,
        "consistent": consistent,
    }


def _is_rate_limit_error(exc: BaseException) -> bool:
    s = str(exc).lower()
    return (
        "429" in str(exc)
        or "ratelimit" in s
        or "rate limit" in s
        or "quota" in s
        or "too many requests" in s
    )


def judge_call(
    prompt: str,
    temperature: float,
    max_tokens: int,
    *,
    sleep_after: float = 0.0,
    rate_limit_budget_seconds: float = 30.0,
) -> Optional[str]:
    """Call judge API; sleep_after after each finished call. Rate limits: backoff up to budget, then skip."""
    if os.getenv("USE_LOCAL_MODEL", "").strip().lower() in ("1", "true", "yes"):
        logger.warning("USE_LOCAL_MODEL is set; judge will use local model, not Azure.")

    rate_sleep_used = 0.0
    attempt = 0
    max_iterations = 64

    while attempt < max_iterations:
        try:
            resp = call_model(
                system_prompt=JUDGE_SYSTEM,
                user_prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                use_hf=False,
                model_name=None,
            )
            out = (resp or "").strip()
            if sleep_after > 0:
                time.sleep(sleep_after)
            return out
        except Exception as e:
            if not _is_rate_limit_error(e):
                logger.error("Judge call failed: %s", e)
                if sleep_after > 0:
                    time.sleep(sleep_after)
                return None
            remaining = rate_limit_budget_seconds - rate_sleep_used
            if remaining <= 0.01:
                logger.warning(
                    "Rate limited: used %.1fs backoff budget; skipping this judge call (re-run to retry)",
                    rate_limit_budget_seconds,
                )
                if sleep_after > 0:
                    time.sleep(sleep_after)
                return None
            backoff = min(2.0 ** (attempt + 1), remaining)
            if backoff < 0.05:
                logger.warning(
                    "Rate limited: backoff budget exhausted; skipping this judge call (re-run to retry)"
                )
                if sleep_after > 0:
                    time.sleep(sleep_after)
                return None
            logger.warning(
                "Rate limited; sleeping %.1fs (%.1fs / %.1fs backoff budget left)",
                backoff,
                remaining,
                rate_limit_budget_seconds,
            )
            time.sleep(backoff)
            rate_sleep_used += backoff
            attempt += 1
    logger.warning("Rate limited: too many attempts; skipping this judge call (re-run to retry)")
    if sleep_after > 0:
        time.sleep(sleep_after)
    return None


def eval_pair(
    tmpl: str,
    p1: str,
    p2: str,
    out_p1: str,
    out_p2: str,
    temperature: float,
    max_tokens: int,
    *,
    include_p2: bool,
    sleep_after: float = 0.0,
    rate_limit_budget_seconds: float = 30.0,
) -> Dict:
    def fill(o_x: str, o_y: str) -> str:
        kw = dict(
            p1=p1,
            p2=p2,
            question=p1,
            O_x=o_x,
            O_y=o_y,
            response_a=o_x,
            response_b=o_y,
            answer_a=o_x,
            answer_b=o_y,
        )
        if include_p2:
            return fill_template(tmpl, **kw)
        return fill_template(tmpl, **{k: v for k, v in kw.items() if k != "p2"})

    run_ab: Dict = {"raw": None, "choice": None, "reasoning": None}
    raw_ab = judge_call(
        fill(out_p1, out_p2),
        temperature,
        max_tokens,
        sleep_after=sleep_after,
        rate_limit_budget_seconds=rate_limit_budget_seconds,
    )
    run_ab["raw"] = raw_ab
    run_ab["choice"] = parse_choice(raw_ab or "")
    run_ab["reasoning"] = parse_reasoning(raw_ab or "")

    run_ba: Dict = {"raw": None, "choice": None, "reasoning": None}
    raw_ba = judge_call(
        fill(out_p2, out_p1),
        temperature,
        max_tokens,
        sleep_after=sleep_after,
        rate_limit_budget_seconds=rate_limit_budget_seconds,
    )
    run_ba["raw"] = raw_ba
    run_ba["choice"] = parse_choice(raw_ba or "")
    run_ba["reasoning"] = parse_reasoning(raw_ba or "")

    return {
        "compare_ab_p1A_p2B": run_ab,
        "compare_ba_p2A_p1B": run_ba,
        "compare": reconcile(run_ab, run_ba),
    }


def eval_all_three(
    tmpl_h: str,
    tmpl_hless: str,
    tmpl_r: str,
    p1: str,
    p2: str,
    out_p1: str,
    out_p2: str,
    temperature: float,
    max_tokens: int,
    *,
    sleep_after: float = 0.0,
    rate_limit_budget_seconds: float = 30.0,
) -> Dict:
    return {
        "helpfulness": eval_pair(
            tmpl_h,
            p1,
            p2,
            out_p1,
            out_p2,
            temperature,
            max_tokens,
            include_p2=False,
            sleep_after=sleep_after,
            rate_limit_budget_seconds=rate_limit_budget_seconds,
        ),
        "harmlessness": eval_pair(
            tmpl_hless,
            p1,
            p2,
            out_p1,
            out_p2,
            temperature,
            max_tokens,
            include_p2=False,
            sleep_after=sleep_after,
            rate_limit_budget_seconds=rate_limit_budget_seconds,
        ),
        "relevance": eval_pair(
            tmpl_r,
            p1,
            p2,
            out_p1,
            out_p2,
            temperature,
            max_tokens,
            include_p2=True,
            sleep_after=sleep_after,
            rate_limit_budget_seconds=rate_limit_budget_seconds,
        ),
    }


def _output_ok(r: Dict) -> bool:
    o = r.get("output")
    return o is not None and isinstance(o, str) and bool(o.strip())


def build_pairs(
    p1_path: Path,
    p2_path: Path,
    max_pairs: Optional[int],
) -> List[Dict]:
    m1: Dict[str, Dict] = {}
    for r in load_jsonl(str(p1_path)):
        pid = str(r.get("p1_id", r.get("id", "")))
        if pid and _output_ok(r):
            m1[pid] = r

    m2: Dict[str, Dict] = {}
    for r in load_jsonl(str(p2_path)):
        pid = str(r.get("p1_id", r.get("id", "")))
        if pid and _output_ok(r):
            m2[pid] = r

    common = sorted(set(m1.keys()) & set(m2.keys()), key=p1_sort_key)
    if max_pairs is not None:
        common = common[: max_pairs]

    rows: List[Dict] = []
    for pid in common:
        a, b = m1[pid], m2[pid]
        rows.append(
            {
                "p1_id": pid,
                "p1": a.get("p1", ""),
                "p2": b.get("p2", ""),
                "o_p1_baseline": a.get("output", ""),
                "o_p2_baseline": b.get("output", ""),
                "category": a.get("category", b.get("category", "")),
                "lang": a.get("lang", b.get("lang", "")),
            }
        )
    return rows


def _dim_counts(results: List[Dict], dim: str) -> Dict[str, int]:
    counts = {LABEL_P1: 0, LABEL_P2: 0, "Tie": 0}
    for r in results:
        block = r.get(dim) or {}
        w = (block.get("compare") or {}).get("winner")
        if w in counts:
            counts[w] += 1
    return counts


def compute_summary(results: List[Dict]) -> Dict:
    out: Dict = {"n_evaluated": len(results), "dimensions": {}}
    for dim in ("helpfulness", "harmlessness", "relevance"):
        counts = _dim_counts(results, dim)
        n = sum(counts.values())
        out["dimensions"][dim] = {
            "winner_counts": counts,
            "p1_baseline_win_rate": round(counts[LABEL_P1] / n, 4) if n else None,
            "p2_baseline_win_rate": round(counts[LABEL_P2] / n, 4) if n else None,
            "tie_rate": round(counts["Tie"] / n, 4) if n else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p1-baseline-jsonl",
        type=Path,
        default=REPO_ROOT / "output" / "p1_baseline_outputs_gemma3.jsonl",
        help="JSONL with p1_id, p1, output (P1-baseline generations)",
    )
    parser.add_argument(
        "--p2-baseline-jsonl",
        type=Path,
        default=REPO_ROOT / "output" / "p2_baseline_gemma3_27b_gpt5" / "baseline_outputs.jsonl",
        help="JSONL with p1_id, p1, p2, output (P2-baseline generations)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for compare_results.jsonl, compare_checkpoint.jsonl, compare_summary.json",
    )
    parser.add_argument("--max-pairs", type=int, default=None, help="Cap number of pairs (after intersection)")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.75,
        help="Seconds to sleep after each judge API call (6 per pair) to reduce 429 / TPM bursts (default: 0.75)",
    )
    parser.add_argument(
        "--rate-limit-budget-seconds",
        type=float,
        default=30.0,
        help="Max total seconds spent in backoff retries per judge call on 429; then skip (default: 30)",
    )
    args = parser.parse_args()

    if not args.p1_baseline_jsonl.is_file():
        sys.exit(f"Missing --p1-baseline-jsonl: {args.p1_baseline_jsonl}")
    if not args.p2_baseline_jsonl.is_file():
        sys.exit(f"Missing --p2-baseline-jsonl: {args.p2_baseline_jsonl}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tmpl_h = load_prompt(TMPL_HELPFULNESS)
    tmpl_hless = load_prompt(TMPL_HARMLESSNESS)
    tmpl_r = load_relevance_pairwise_template()

    pairs = build_pairs(args.p1_baseline_jsonl, args.p2_baseline_jsonl, args.max_pairs)
    if not pairs:
        print("No common p1_ids with non-empty outputs in both files.")
        return

    print(
        f"Pairs to evaluate (common ids): {len(pairs)} "
        f"(6 judge calls per pair; sleep {args.sleep_seconds}s after each; "
        f"429 backoff budget {args.rate_limit_budget_seconds}s per call)"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "compare_results.jsonl"
    checkpoint_path = args.output_dir / "compare_checkpoint.jsonl"
    lock_path = args.output_dir / "compare_write.lock"
    lock_path.touch(exist_ok=True)
    summary_path = args.output_dir / "compare_summary.json"

    processed: Set[str] = load_checkpoint(checkpoint_path)

    for idx, row in enumerate(pairs, 1):
        pid = row["p1_id"]
        if str(pid) in processed:
            logger.info("[%s/%s] skip p1_id=%s (checkpoint)", idx, len(pairs), pid)
            continue

        result = eval_all_three(
            tmpl_h,
            tmpl_hless,
            tmpl_r,
            row["p1"],
            row["p2"],
            row["o_p1_baseline"],
            row["o_p2_baseline"],
            args.temperature,
            args.max_tokens,
            sleep_after=args.sleep_seconds,
            rate_limit_budget_seconds=args.rate_limit_budget_seconds,
        )
        out = {**row, **result}
        append_jsonl(str(results_path), [out], lock_path)
        append_jsonl(str(checkpoint_path), [{"p1_id": pid}], lock_path)
        processed.add(str(pid))

        h = (result.get("helpfulness") or {}).get("compare") or {}
        f = (result.get("harmlessness") or {}).get("compare") or {}
        rel = (result.get("relevance") or {}).get("compare") or {}
        logger.info(
            "[%s/%s] p1_id=%s helpfulness=%s harmlessness=%s relevance=%s",
            idx,
            len(pairs),
            pid,
            h.get("winner"),
            f.get("winner"),
            rel.get("winner"),
        )

    all_rows = load_jsonl(str(results_path))
    summary = compute_summary(all_rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {results_path}\n{summary_path}")


if __name__ == "__main__":
    main()
