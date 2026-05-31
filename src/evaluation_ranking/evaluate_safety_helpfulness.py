#!/usr/bin/env python3
"""
Evaluate O1 vs O2 on relevance, helpfulness, and harmlessness.

Evaluations (from evaluation.md):
  1. Dual relevance — O2 scored against both p1 and p2 (pointwise, 1 call).
  2. Comparative helpfulness — O1 vs O2 pairwise A/B/Tie (2 calls, position-swapped).
  3. Comparative harmlessness — O1 vs O2 pairwise A/B/Tie (2 calls, position-swapped; judge picks more harmless).

Total: up to 5 judge calls per record.

Input modes
-----------
  --o2-jsonl FILE                  (p1_id, p1, p2, output=O2)
  --o1-jsonl FILE                  (p1_id, output=O1)  [needed for helpfulness/harmlessness]
  --input-jsonl FILE               single file with o1, o2, p1, p2

Output
------
  <output-dir>/eval_results.jsonl
  <output-dir>/eval_checkpoint.jsonl
  <output-dir>/eval_summary.json
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from src.baselines.baseline_common import load_jsonl, append_jsonl, load_checkpoint
from src.utils.model_client import call_model
from src.utils.wandb_utils import init_wandb, log_metrics, finish_run

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_DIR = ROOT / "src" / "judge_prompts"

JUDGE_SYSTEM = "You are an impartial AI evaluator. Follow the evaluation instructions carefully."


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8").strip()


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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_choice(text: str) -> Optional[str]:
    """Parse judge verdict as A, B, or Tie.

    Supports:
    - Legacy: ``<choice>A|B|Tie</choice>``
    - MT-Bench style: ``[[A]]``, ``[[B]]``, ``[[C]]`` (C = tie)
    - Plain-text harmlessness (A/B/Tie verdict): last non-empty line ``A`` / ``B`` / ``Tie``
    """
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"<choice>\s*(.*?)\s*</choice>", t, re.DOTALL)
    if m:
        raw = m.group(1).strip().upper()
        if raw in ("A", "B", "TIE"):
            return "A" if raw == "A" else ("B" if raw == "B" else "Tie")
        if "TIE" in raw:
            return "Tie"
        if "A" in raw and "B" not in raw:
            return "A"
        if "B" in raw and "A" not in raw:
            return "B"
        return "Tie"
    matches = list(re.finditer(r"\[\[\s*([ABCabc])\s*\]\]", t))
    if matches:
        c = matches[-1].group(1).upper()
        if c == "C":
            return "Tie"
        return "A" if c == "A" else "B"
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    for line in reversed(lines):
        s = line.strip().strip('"').strip("'").strip("*•-` ")
        ul = s.upper().rstrip(".")
        if ul in ("A", "B"):
            return s[0].upper()
        if ul in ("TIE", "TIE.", "TIE!") or ul.startswith("TIE "):
            return "Tie"
    return None


def parse_reasoning(text: str) -> Optional[str]:
    m = re.search(r"<reasoning>(.*?)</reasoning>", text or "", re.DOTALL)
    return m.group(1).strip() if m else None


def parse_comparative_reasoning(text: Optional[str]) -> Optional[str]:
    """Free-text explanation for comparative judges (XML or MT-Bench style)."""
    if not text:
        return None
    r = parse_reasoning(text)
    if r:
        return r
    t = text.strip()
    t2 = re.sub(r"\s*\[\[\s*[ABCabc]\s*\]\]\s*$", "", t).strip()
    if t2:
        return t2
    return t or None


def parse_relevance(text: str) -> Dict:
    d: Dict = {"reasoning": parse_reasoning(text)}
    m = re.search(r"<relevance_to_p1>\s*(\d)\s*</relevance_to_p1>", text or "")
    d["relevance_to_p1"] = int(m.group(1)) if m else None
    m = re.search(r"<relevance_to_p2>\s*(\d)\s*</relevance_to_p2>", text or "")
    d["relevance_to_p2"] = int(m.group(1)) if m else None
    m = re.search(r"<more_relevant_to>\s*(.*?)\s*</more_relevant_to>", text or "", re.DOTALL)
    d["more_relevant_to"] = m.group(1).strip().lower() if m else None
    return d


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

def judge_call(prompt: str, temperature: float, max_tokens: int,
               model_name: Optional[str], use_hf: bool) -> Optional[str]:
    try:
        resp = call_model(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_name,
            use_hf=use_hf,
        )
        return resp.strip() if resp else None
    except Exception as e:
        logger.error("Judge call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------

def eval_relevance(p1: str, p2: str, o2: str, tmpl: str, **kw) -> Dict:
    prompt = fill_template(tmpl, p1=p1, p2=p2, O2=o2)
    raw = judge_call(prompt, **kw)
    result = parse_relevance(raw or "")
    result["raw"] = raw
    return result


def eval_comparative(
    p1: str,
    resp_a: str,
    resp_b: str,
    tmpl: str,
    **kw,
) -> Dict:
    """Pairwise judge: ``resp_a`` / ``resp_b`` are shown as Response A / B (or Assistant A / B).

    Template placeholders supported: ``{p1}``, ``{question}``, ``{O_x}``, ``{O_y}``,
    ``{response_a}``, ``{response_b}``, ``{answer_a}``, ``{answer_b}``.

    Parsed ``choice`` is the judge's letter verdict along the dimension in the template
    (e.g. more helpful or more harmless).
    """
    prompt = fill_template(
        tmpl,
        p1=p1,
        question=p1,
        O_x=resp_a,
        O_y=resp_b,
        response_a=resp_a,
        response_b=resp_b,
        answer_a=resp_a,
        answer_b=resp_b,
    )
    raw = judge_call(prompt, **kw)
    choice = parse_choice(raw or "")
    return {"raw": raw, "choice": choice, "reasoning": parse_comparative_reasoning(raw)}


def _map_choice_to_label(choice: Optional[str], a_is: str) -> Optional[str]:
    """Map A/B/Tie to O1/O2/Tie given which response was placed as A."""
    if choice is None:
        return None
    other = "O2" if a_is == "O1" else "O1"
    if choice == "A":
        return a_is
    if choice == "B":
        return other
    return "Tie"


def reconcile_comparative(run_ab: Dict, run_ba: Dict) -> Dict:
    """Reconcile two position-swapped runs via majority vote.

    Run AB: O1=A, O2=B.  Run BA: O2=A, O1=B.
    Map each choice to O1/O2/Tie, then pick winner by agreement or Tie.
    """
    ab_label = _map_choice_to_label(run_ab.get("choice"), "O1")
    ba_label = _map_choice_to_label(run_ba.get("choice"), "O2")
    if ab_label and ba_label and ab_label == ba_label:
        winner = ab_label
        consistent = True
    elif ab_label and not ba_label:
        winner = ab_label
        consistent = None
    elif ba_label and not ab_label:
        winner = ba_label
        consistent = None
    else:
        winner = "Tie"
        consistent = False if (ab_label and ba_label) else None
    return {"winner": winner, "ab_label": ab_label, "ba_label": ba_label, "consistent": consistent}


def _flip_o1_o2_winner_label(lab: Optional[str]) -> Optional[str]:
    if lab == "O1":
        return "O2"
    if lab == "O2":
        return "O1"
    return lab


def harmlessness_final_block(r: Dict) -> Dict:
    """Reconciled harmlessness verdict (winner = more harmless).

    Reads ``harmlessness`` from current evals, or maps legacy ``harmfulness`` rows
    where the stored winner was **more harmful** (pre-rename pipeline) into the
    more-harmless convention by swapping O1/O2 on non-tie wins.
    """
    b = r.get("harmlessness")
    if b:
        return b
    leg = r.get("harmfulness") or {}
    if not leg:
        return {}
    w = leg.get("winner")
    out = dict(leg)
    if w == "O1":
        out["winner"] = "O2"
    elif w == "O2":
        out["winner"] = "O1"
    else:
        out["winner"] = w
    ab = leg.get("ab_label")
    ba = leg.get("ba_label")
    out["ab_label"] = _flip_o1_o2_winner_label(ab) if ab in ("O1", "O2") else ab
    out["ba_label"] = _flip_o1_o2_winner_label(ba) if ba in ("O1", "O2") else ba
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(args) -> List[Dict]:
    if args.input_jsonl:
        recs = load_jsonl(args.input_jsonl)
        for r in recs:
            r.setdefault("o1", r.get("O1", ""))
            r.setdefault("o2", r.get("O2", r.get("output", "")))
        return recs

    o2_recs = load_jsonl(args.o2_jsonl)
    o1_map: Dict[str, Dict] = {}
    if args.o1_jsonl:
        for r in load_jsonl(args.o1_jsonl):
            pid = str(r.get("p1_id", r.get("id", "")))
            o1_map[pid] = r

    records = []
    for r2 in o2_recs:
        pid = str(r2.get("p1_id", r2.get("id", "")))
        r1 = o1_map.get(pid, {})
        records.append({
            "p1_id": pid,
            "p1": r2.get("p1", r1.get("p1", "")),
            "p2": r2.get("p2", ""),
            "o1": r1.get("output", ""),
            "o2": r2.get("output", ""),
            "category": r2.get("category", r1.get("category", "")),
            "lang": r2.get("lang", r1.get("lang", "")),
        })
    return records


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(results: List[Dict], harmfulness_mode: bool = False) -> Dict:
    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    rel_p1 = [r["relevance"]["relevance_to_p1"] for r in results if "relevance" in r and r["relevance"].get("relevance_to_p1") is not None]
    rel_p2 = [r["relevance"]["relevance_to_p2"] for r in results if "relevance" in r and r["relevance"].get("relevance_to_p2") is not None]
    mrt = [r["relevance"]["more_relevant_to"] for r in results if "relevance" in r and r["relevance"].get("more_relevant_to")]

    def win_counts(key):
        counts = {"O1": 0, "O2": 0, "Tie": 0}
        for r in results:
            w = (r.get(key) or {}).get("winner")
            if w in counts:
                counts[w] += 1
        return counts

    help_c = win_counts("helpfulness")
    help_total = sum(help_c.values())

    # harmfulness_mode: raw judge verdict = "who is MORE harmful" (O2 winning is bad for O2).
    # harmlessness_final_block flips this so winner = "who is MORE harmless".
    # We expose both: raw harmfulness counts + flipped harmlessness counts.
    if harmfulness_mode:
        raw_harm_counts = win_counts("harmfulness")
        # flipped: winner of harmlessness = less harmful side
        hless_counts = {"O1": 0, "O2": 0, "Tie": 0}
        for r in results:
            w = harmlessness_final_block(r).get("winner")
            if w in hless_counts:
                hless_counts[w] += 1
        harm_total = sum(raw_harm_counts.values())
        hless_total = sum(hless_counts.values())
        harm_block = {
            # raw: who the judge found MORE harmful
            "harmfulness_counts": raw_harm_counts,
            "harmfulness_n": harm_total,
            "harmfulness_o2_rate": round(raw_harm_counts["O2"] / harm_total, 3) if harm_total else None,
            # flipped: who is MORE harmless (= less harmful)
            "harmlessness_counts": hless_counts,
            "harmlessness_n": hless_total,
            "harmlessness_o2_winrate": round(hless_counts["O2"] / hless_total, 3) if hless_total else None,
        }
    else:
        hless_counts = {"O1": 0, "O2": 0, "Tie": 0}
        for r in results:
            w = harmlessness_final_block(r).get("winner")
            if w in hless_counts:
                hless_counts[w] += 1
        hless_total = sum(hless_counts.values())
        harm_block = {
            "harmlessness_counts": hless_counts,
            "harmlessness_n": hless_total,
            "harmlessness_o2_winrate": round(hless_counts["O2"] / hless_total, 3) if hless_total else None,
        }

    summary: Dict = {
        "n_records": len(results),
        "relevance_to_p1_mean": safe_mean(rel_p1),
        "relevance_to_p2_mean": safe_mean(rel_p2),
        "relevance_n": len(rel_p1),
        "more_relevant_to_p2_frac": round(sum(1 for v in mrt if v == "p2") / len(mrt), 3) if mrt else None,
        "more_relevant_to_counts": {
            "p1": sum(1 for v in mrt if v == "p1"),
            "p2": sum(1 for v in mrt if v == "p2"),
            "equal": sum(1 for v in mrt if v == "equal"),
        },
        "helpfulness_counts": help_c,
        "helpfulness_n": help_total,
        "helpfulness_o2_winrate": round(help_c["O2"] / help_total, 3) if help_total else None,
        **harm_block,
    }
    return summary


def flatten_summary_for_wandb(summary: Dict[str, Any], prefix: str = "eval") -> Dict[str, Any]:
    """Scalar dict for wandb.log (one nested level expanded)."""
    out: Dict[str, Any] = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{prefix}/{k}/{k2}"] = v2
        else:
            out[f"{prefix}/{k}"] = v
    return out


def write_derived_exports(results: List[Dict], records: List[Dict], output_dir: Path) -> None:
    """
    Write parser-friendly exports:
    1) eval_reasoning_by_prompt.jsonl: one row per p1_id x evaluated prompt call
    2) eval_metrics_flat.jsonl: one row per p1_id with flattened metrics + p1/p2/o1/o2
    """
    rec_by_id: Dict[str, Dict] = {}
    for rec in records:
        pid = str(rec.get("p1_id", rec.get("id", "")))
        if pid:
            rec_by_id[pid] = rec

    reasoning_rows: List[Dict] = []
    flat_rows: List[Dict] = []

    for r in results:
        pid = str(r.get("p1_id", ""))
        src = rec_by_id.get(pid, {})
        p1 = src.get("p1", "")
        p2 = src.get("p2", "")
        o1 = src.get("o1", "")
        o2 = src.get("o2", "")

        rel = r.get("relevance") or {}
        help_ab = r.get("helpfulness_o1A_o2B") or {}
        help_ba = r.get("helpfulness_o2A_o1B") or {}
        help_final = r.get("helpfulness") or {}
        hless_ab = r.get("harmlessness_o1A_o2B") or r.get("harmfulness_o1A_o2B") or {}
        hless_ba = r.get("harmlessness_o2A_o1B") or r.get("harmfulness_o2A_o1B") or {}
        hless_final = harmlessness_final_block(r)

        # Per-id flat metrics row with source text included
        flat_row = {
                "p1_id": pid,
                "p1": p1,
                "p2": p2,
                "o1": o1,
                "o2": o2,
                "relevance_to_p1": rel.get("relevance_to_p1"),
                "relevance_to_p2": rel.get("relevance_to_p2"),
                "more_relevant_to": rel.get("more_relevant_to"),
                "helpfulness_winner": help_final.get("winner"),
                "helpfulness_ab_label": help_final.get("ab_label"),
                "helpfulness_ba_label": help_final.get("ba_label"),
                "helpfulness_consistent": help_final.get("consistent"),
                "harmlessness_winner": hless_final.get("winner"),
                "harmlessness_ab_label": hless_final.get("ab_label"),
                "harmlessness_ba_label": hless_final.get("ba_label"),
                "harmlessness_consistent": hless_final.get("consistent"),
            }
        if src.get("level") is not None:
            try:
                flat_row["level"] = int(src["level"])
            except (TypeError, ValueError):
                flat_row["level"] = src.get("level")
        flat_rows.append(flat_row)

        # Per-prompt reasoning rows (including null labels/consistency when parse fails)
        if rel:
            reasoning_rows.append(
                {
                    "p1_id": pid,
                    "eval_type": "relevance",
                    "prompt_variant": "o2_vs_p1_p2",
                    "p1": p1,
                    "p2": p2,
                    "o1": o1,
                    "o2": o2,
                    "raw": rel.get("raw"),
                    "reasoning": rel.get("reasoning"),
                    "parsed": {
                        "relevance_to_p1": rel.get("relevance_to_p1"),
                        "relevance_to_p2": rel.get("relevance_to_p2"),
                        "more_relevant_to": rel.get("more_relevant_to"),
                    },
                }
            )

        if help_ab or help_ba or help_final:
            reasoning_rows.append(
                {
                    "p1_id": pid,
                    "eval_type": "helpfulness",
                    "prompt_variant": "o1A_o2B",
                    "p1": p1,
                    "p2": p2,
                    "o1": o1,
                    "o2": o2,
                    "raw": help_ab.get("raw"),
                    "reasoning": help_ab.get("reasoning"),
                    "choice": help_ab.get("choice"),
                    "mapped_label": _map_choice_to_label(help_ab.get("choice"), "O1"),
                    "ab_label": help_final.get("ab_label"),
                    "ba_label": help_final.get("ba_label"),
                    "winner": help_final.get("winner"),
                    "consistent": help_final.get("consistent"),
                }
            )
            reasoning_rows.append(
                {
                    "p1_id": pid,
                    "eval_type": "helpfulness",
                    "prompt_variant": "o2A_o1B",
                    "p1": p1,
                    "p2": p2,
                    "o1": o1,
                    "o2": o2,
                    "raw": help_ba.get("raw"),
                    "reasoning": help_ba.get("reasoning"),
                    "choice": help_ba.get("choice"),
                    "mapped_label": _map_choice_to_label(help_ba.get("choice"), "O2"),
                    "ab_label": help_final.get("ab_label"),
                    "ba_label": help_final.get("ba_label"),
                    "winner": help_final.get("winner"),
                    "consistent": help_final.get("consistent"),
                }
            )

        if hless_ab or hless_ba or hless_final:
            reasoning_rows.append(
                {
                    "p1_id": pid,
                    "eval_type": "harmlessness",
                    "prompt_variant": "o1A_o2B",
                    "p1": p1,
                    "p2": p2,
                    "o1": o1,
                    "o2": o2,
                    "raw": hless_ab.get("raw"),
                    "reasoning": hless_ab.get("reasoning"),
                    "choice": hless_ab.get("choice"),
                    "mapped_label": _map_choice_to_label(hless_ab.get("choice"), "O1"),
                    "ab_label": hless_final.get("ab_label"),
                    "ba_label": hless_final.get("ba_label"),
                    "winner": hless_final.get("winner"),
                    "consistent": hless_final.get("consistent"),
                }
            )
            reasoning_rows.append(
                {
                    "p1_id": pid,
                    "eval_type": "harmlessness",
                    "prompt_variant": "o2A_o1B",
                    "p1": p1,
                    "p2": p2,
                    "o1": o1,
                    "o2": o2,
                    "raw": hless_ba.get("raw"),
                    "reasoning": hless_ba.get("reasoning"),
                    "choice": hless_ba.get("choice"),
                    "mapped_label": _map_choice_to_label(hless_ba.get("choice"), "O2"),
                    "ab_label": hless_final.get("ab_label"),
                    "ba_label": hless_final.get("ba_label"),
                    "winner": hless_final.get("winner"),
                    "consistent": hless_final.get("consistent"),
                }
            )

    reasoning_file = output_dir / "eval_reasoning_by_prompt.jsonl"
    flat_file = output_dir / "eval_metrics_flat.jsonl"

    with open(reasoning_file, "w", encoding="utf-8") as f:
        for row in reasoning_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(flat_file, "w", encoding="utf-8") as f:
        for row in flat_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _normalize_eval_type_token(tok: str) -> str:
    t = tok.strip().lower()
    if t == "harmfulness":
        return "harmlessness"
    return t


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation: dual relevance, comparative helpfulness & harmlessness."
    )

    parser.add_argument("--input-jsonl", help="Single JSONL with o1, o2, p1, p2 fields.")
    parser.add_argument("--o2-jsonl", help="O2 outputs JSONL (p1_id, p1, p2, output).")
    parser.add_argument("--o1-jsonl", help="O1 outputs JSONL (p1_id, output). Needed for helpfulness/harmlessness.")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--use-local-model", action="store_true")
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--harmfulness-mode", action="store_true",
                        help="Use eval_harmfulness.txt prompt; [[A]]=more harmful. Results stored "
                             "under 'harmfulness' keys so harmlessness_final_block flips them for summary.")

    parser.add_argument("--evals", type=str, default="all",
                        help="Comma-separated: relevance,helpfulness,harmlessness (default: all). "
                        "Legacy alias: harmfulness → harmlessness.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log to Weights & Biases (default: on). Use --no-wandb to disable.",
    )
    parser.add_argument(
        "--log-metrics-every",
        type=int,
        default=25,
        help="After every K completed rows in eval_results.jsonl, print summary metrics and "
        "log them to wandb (0 = disable intermediate logging). Default: 25.",
    )
    parser.add_argument("--exit-on-null", action="store_true")

    args = parser.parse_args()

    _env_every = os.environ.get("LOG_METRICS_EVERY", "").strip()
    if _env_every:
        try:
            args.log_metrics_every = max(0, int(_env_every))
        except ValueError:
            logger.warning("Ignoring invalid LOG_METRICS_EVERY=%r", _env_every)

    if not args.input_jsonl and not args.o2_jsonl:
        parser.error("Provide either --input-jsonl or --o2-jsonl.")

    if os.environ.get("EXIT_ON_NULL", "").strip().lower() in ("1", "true", "yes"):
        args.exit_on_null = True

    if args.use_local_model and args.judge_model:
        os.environ["USE_LOCAL_MODEL"] = "true"
        os.environ["LOCAL_MODEL"] = args.judge_model

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    _valid_evals = {"relevance", "helpfulness", "harmlessness"}
    if args.evals.strip().lower() == "all":
        eval_types: Set[str] = set(_valid_evals)
    else:
        eval_types = {_normalize_eval_type_token(x) for x in args.evals.split(",") if x.strip()}
        unknown = eval_types - _valid_evals
        if unknown:
            parser.error(f"Unknown --evals value(s): {sorted(unknown)}. Use: {sorted(_valid_evals)}")

    records = load_records(args)
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        print("No records found.")
        return

    has_o1 = any(r.get("o1") for r in records)
    if not has_o1:
        logger.warning("No O1 data — skipping helpfulness and harmlessness (require both O1 and O2).")
        eval_types -= {"helpfulness", "harmlessness"}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "eval_results.jsonl"
    checkpoint_file = output_dir / "eval_checkpoint.jsonl"
    lock_file = output_dir / "eval_write.lock"
    lock_file.touch(exist_ok=True)
    summary_file = output_dir / "eval_summary.json"

    processed_ids = load_checkpoint(checkpoint_file)

    def _row_pid(rec: Dict, idx: int) -> str:
        return str(rec.get("p1_id", f"row_{idx}"))

    n_skip = sum(1 for i, r in enumerate(records, 1) if _row_pid(r, i) in processed_ids)
    n_todo = len(records) - n_skip
    if n_skip:
        msg = (
            f"Resuming from checkpoint: skipping {n_skip} prompt(s) already in {checkpoint_file.name} "
            f"({n_todo} remaining to judge)."
        )
        logger.info(msg)
        print(msg, flush=True)
    else:
        logger.info("No checkpoint resume — evaluating all %d prompt(s).", len(records))

    templates: Dict[str, str] = {}
    if "relevance" in eval_types:
        templates["relevance"] = load_prompt("eval_relevance.txt")
    if "helpfulness" in eval_types:
        templates["helpfulness"] = load_prompt("eval_helpfulness.txt")
    if "harmlessness" in eval_types:
        harm_prompt_file = "eval_harmfulness.txt" if args.harmfulness_mode else "eval_harmlessness.txt"
        templates["harmlessness"] = load_prompt(harm_prompt_file)

    judge_kw = dict(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_name=args.judge_model if args.use_local_model else None,
        use_hf=args.use_hf,
    )

    init_wandb(
        job_name="eval_safety_helpfulness",
        config={
            **{k: v for k, v in vars(args).items() if k != "wandb"},
            "wandb": args.wandb,
            "total_records": len(records),
            "checkpoint_skipped": n_skip,
            "eval_types": sorted(eval_types),
        },
        enabled=args.wandb,
    )

    results_snapshot: List[Dict] = []
    if results_file.exists():
        results_snapshot = load_jsonl(str(results_file))

    pending_pairs: List[Tuple[int, Dict[str, Any]]] = [
        (i, r) for i, r in enumerate(records, 1) if _row_pid(r, i) not in processed_ids
    ]

    done = 0
    skipped = n_skip
    log_every = max(0, int(args.log_metrics_every))

    def _log_progress_metrics(reason: str) -> None:
        if not results_snapshot:
            return
        summ = compute_summary(results_snapshot, harmfulness_mode=args.harmfulness_mode)
        flat = flatten_summary_for_wandb(summ)
        flat["eval/progress_note"] = reason
        flat["eval/n_completed_total"] = len(results_snapshot)
        flat["eval/n_new_this_run"] = done
        flat["eval/n_skipped_checkpoint"] = skipped
        step = len(results_snapshot)
        logger.info(
            "Evaluation metrics (%s, n=%d): %s",
            reason,
            step,
            json.dumps(summ, ensure_ascii=False, default=str),
        )
        print(f"\n[{reason}] n_completed={step} metrics:\n{json.dumps(summ, indent=2, ensure_ascii=False)}\n", flush=True)
        log_metrics(flat, step=step)

    pbar = tqdm(
        pending_pairs,
        total=len(pending_pairs),
        desc="Judge eval (pending)",
        unit="prompt",
        dynamic_ncols=True,
        mininterval=0.5,
        file=sys.stderr,
    )
    try:
        for idx, rec in pbar:
            pid = _row_pid(rec, idx)

            pbar.set_postfix_str(f"p1_id={pid}", refresh=False)

            p1 = rec.get("p1", "")
            p2 = rec.get("p2", "")
            o1 = rec.get("o1", "")
            o2 = rec.get("o2", "")

            result: Dict = {"p1_id": pid}

            # --- Eval 1: Dual relevance (O2 vs p1 and p2) ---
            if "relevance" in eval_types and o2:
                result["relevance"] = eval_relevance(p1, p2, o2, templates["relevance"], **judge_kw)

            # --- Eval 2: Comparative helpfulness (position-swapped) ---
            if "helpfulness" in eval_types and o1 and o2:
                run_ab = eval_comparative(p1, o1, o2, templates["helpfulness"], **judge_kw)
                run_ba = eval_comparative(p1, o2, o1, templates["helpfulness"], **judge_kw)
                result["helpfulness_o1A_o2B"] = run_ab
                result["helpfulness_o2A_o1B"] = run_ba
                result["helpfulness"] = reconcile_comparative(run_ab, run_ba)

            # --- Eval 3: Comparative harmlessness/harmfulness (position-swapped) ---
            # In harmfulness_mode: [[A]]=more harmful; results stored under "harmfulness" keys so
            # harmlessness_final_block() flips winner (more harmful → less harmless) for the summary.
            if "harmlessness" in eval_types and o1 and o2:
                run_ab = eval_comparative(p1, o1, o2, templates["harmlessness"], **judge_kw)
                run_ba = eval_comparative(p1, o2, o1, templates["harmlessness"], **judge_kw)
                if args.harmfulness_mode:
                    result["harmfulness_o1A_o2B"] = run_ab
                    result["harmfulness_o2A_o1B"] = run_ba
                    result["harmfulness"] = reconcile_comparative(run_ab, run_ba)
                else:
                    result["harmlessness_o1A_o2B"] = run_ab
                    result["harmlessness_o2A_o1B"] = run_ba
                    result["harmlessness"] = reconcile_comparative(run_ab, run_ba)

            # --- exit-on-null ---
            if args.exit_on_null:
                rel = result.get("relevance")
                if rel and rel.get("relevance_to_p1") is None and rel.get("relevance_to_p2") is None:
                    logger.error("exit-on-null: relevance unparseable for p1_id=%s. Aborting.", pid)
                    finish_run()
                    sys.exit(1)
                for k in ("helpfulness", "harmlessness"):
                    v = result.get(k)
                    if v is not None and v.get("winner") is None:
                        logger.error("exit-on-null: %s winner is null for p1_id=%s. Aborting.", k, pid)
                        finish_run()
                        sys.exit(1)

            append_jsonl(str(results_file), [result], lock_file)
            append_jsonl(str(checkpoint_file), [{"p1_id": pid}], lock_file)
            processed_ids.add(pid)
            results_snapshot.append(result)
            done += 1

            if log_every and len(results_snapshot) % log_every == 0:
                _log_progress_metrics(f"every_{log_every}_completed")

            if args.wandb and pending_pairs and (done % 10 == 0 or done == len(pending_pairs)):
                log_metrics(
                    {
                        "session/evaluated_new": done,
                        "session/pending_total": len(pending_pairs),
                        "session/frac_complete": round(done / len(pending_pairs), 4),
                        "session/n_skipped_checkpoint_start": n_skip,
                    },
                    step=len(results_snapshot),
                )
    finally:
        pbar.close()

    if log_every and results_snapshot and (len(results_snapshot) % log_every != 0) and done > 0:
        _log_progress_metrics("final_partial_interval")

    # --- Summary ---
    all_results = load_jsonl(str(results_file))
    write_derived_exports(all_results, records, output_dir)
    summary = compute_summary(all_results, harmfulness_mode=args.harmfulness_mode)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {done} record(s) newly evaluated, {skipped} skipped (checkpoint).")
    print(f"Results: {results_file}")
    print(f"Summary: {summary_file}")
    print(json.dumps(summary, indent=2))
    log_metrics(flatten_summary_for_wandb(summary, prefix="eval/final"), step=len(all_results))
    finish_run()


if __name__ == "__main__":
    main()
