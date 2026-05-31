# evaluate_rewriters.py
import json
import argparse
from evaluate_core import run_evaluation

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to language-specific config JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_evaluation(
        p1_file=cfg["p1_file"],
        pprime_file=cfg["pprime_file"],
        judge_prompt_file=cfg["judge_prompt_file"],
        output_dir=cfg["output_dir"],
        language=cfg["name"],
        sanitizer_terms_file=cfg["sanitizer_terms_file"],
        use_schema_judge=cfg.get("use_schema_judge", True),
        schema_extractor_prompt_file=cfg.get("schema_extractor_prompt_file"),
        weights=cfg.get("weights"),
        weight_profiles=cfg.get("weight_profiles")
    )

if __name__ == "__main__":
    main()
