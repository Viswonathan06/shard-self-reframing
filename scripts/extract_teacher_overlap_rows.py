"""
For each model, find the p1_ids that appear in both the qwen35_122b_teacher
test set and the self_model test set. Filter both inference files to those
overlapping rows and write paired output files for judging.

Output:
  output/SFT/overlap_judge/{model}/teacher_sft.jsonl   (rational_sft filtered)
  output/SFT/overlap_judge/{model}/self_sft.jsonl      (self_model filtered)
"""

import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFT  = os.path.join(ROOT, "output", "SFT")

TINFER_NR = os.path.join(SFT, "qwen35_122b_teacher", "combined_dna_linguasafe", "inference")
TINFER_R  = os.path.join(SFT, "qwen35_122b_teacher", "combined_dna_linguasafe_reasoning", "inference")

# combined teacher test p1_ids (non-reasoning + reasoning)
def _load_ids(path):
    return set(json.loads(l)['p1_id'] for l in open(path))

teacher_ids_nr = _load_ids(os.path.join(SFT, "qwen35_122b_teacher", "combined_dna_linguasafe", "test.jsonl"))
teacher_ids_r  = _load_ids(os.path.join(SFT, "qwen35_122b_teacher", "combined_dna_linguasafe_reasoning", "test.jsonl"))

MODELS = {
    "mistral7b": {
        "sm_infer":      "self_model/mistral7b/inference/mistral7b.jsonl",
        "teacher_infer": f"{TINFER_NR}/mistral7b.jsonl",
        "teacher_ids":   teacher_ids_nr,
    },
    "mistral24b": {
        "sm_infer":      "self_model/mistral24b/inference/mistral24b.jsonl",
        "teacher_infer": f"{TINFER_NR}/mistral24b_rerun.jsonl",
        "teacher_ids":   teacher_ids_nr,
    },
    "qwen35_9b": {
        "sm_infer":      "self_model/qwen35_9b_thinking/inference/qwen35_9b_thinking.jsonl",
        "teacher_infer": f"{TINFER_R}/qwen35_9b.jsonl",
        "teacher_ids":   teacher_ids_r,
    },
    "qwen35_27b": {
        "sm_infer":      "self_model/qwen35_27b_thinking/inference/qwen35_27b_thinking.jsonl",
        "teacher_infer": f"{TINFER_R}/qwen35_27b_thinking.jsonl",
        "teacher_ids":   teacher_ids_r,
    },
}

total_written = 0
for model, paths in MODELS.items():
    sm_path = os.path.join(SFT, paths["sm_infer"])
    t_path  = paths["teacher_infer"]
    t_ids   = paths["teacher_ids"]

    if not os.path.exists(sm_path):
        print(f"[SKIP] {model}: self_model inference not found: {sm_path}")
        continue
    if not os.path.exists(t_path):
        print(f"[SKIP] {model}: teacher inference not found: {t_path}")
        continue

    sm_rows = {json.loads(l)['p1_id']: json.loads(l) for l in open(sm_path)}
    t_rows  = {json.loads(l)['p1_id']: json.loads(l) for l in open(t_path)}

    # common to all three: teacher test set, self_model inference, teacher inference
    common_ids = sorted(t_ids & set(sm_rows) & set(t_rows))

    if not common_ids:
        print(f"[SKIP] {model}: no common p1_ids")
        continue

    out_dir = os.path.join(SFT, "overlap_judge", model)
    os.makedirs(out_dir, exist_ok=True)

    sm_out = os.path.join(out_dir, "self_sft.jsonl")
    t_out  = os.path.join(out_dir, "teacher_sft.jsonl")

    with open(t_out, "w") as f:
        for pid in common_ids:
            f.write(json.dumps(t_rows[pid]) + "\n")
    with open(sm_out, "w") as f:
        for pid in common_ids:
            f.write(json.dumps(sm_rows[pid]) + "\n")

    print(f"{model:12s}: {len(common_ids):4d} overlapping rows -> {out_dir}")
    total_written += 1

print(f"\nDone. {total_written} model pairs ready for judging.")
