#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --job-name=p1_p2_outputs_zh_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@institution.edu

# ===============================
# 1. Environment setup
# ===============================
ROOT_DIR="/nas/longleaf$HOME"
cd "$ROOT_DIR"

DATE_TAG=$(date +%Y%m%d)
RUN_TAG="${DATE_TAG}_p1_p2_outputs_chinese_gpu_${SLURM_JOB_ID}"

LOG_DIR="${ROOT_DIR}/logs/${RUN_TAG}"
OUTPUT_DIR="${ROOT_DIR}/outputs/${RUN_TAG}"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

exec > >(tee -a "${LOG_DIR}/job.log")
exec 2> >(tee -a "${LOG_DIR}/job.err" >&2)

echo "Starting Chinese P1->P2->Outputs GPU job on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Run tag: ${RUN_TAG}"
echo "Logs: ${LOG_DIR}"
echo "Outputs: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/users/v/m/vmanoran/.cache/huggingface}"

# ===============================
# 2. Configuration
# ===============================
LANGUAGE=${LANGUAGE:-zh}  # Chinese
MAX_P1=${MAX_P1:-None}  # Process all Chinese prompts (None = no limit)
CSV_PATH=${CSV_PATH:-/nas/longleaf$HOME
PPRIME_FILE=${PPRIME_FILE:-${ROOT_DIR}/appendix/prompts/pprime_by_category.jsonl}
SANITIZER_TERMS=${SANITIZER_TERMS:-${ROOT_DIR}/config/terms_zh.json}
LOCAL_MODEL=${LOCAL_MODEL:-Qwen/Qwen2.5-32B-Instruct}
# Path to previous checkpoint file to exclude already processed prompts (if any)
EXCLUDE_CHECKPOINT=${EXCLUDE_CHECKPOINT:-}

echo "Configuration:"
echo "  Language: ${LANGUAGE} (Chinese)"
echo "  Max P1 prompts: ${MAX_P1}"
echo "  CSV path: ${CSV_PATH}"
echo "  P' file: ${PPRIME_FILE}"
echo "  Sanitizer terms: ${SANITIZER_TERMS}"
echo "  Local model: ${LOCAL_MODEL}"
echo "  Output directory: ${OUTPUT_DIR}"
if [[ -n "${EXCLUDE_CHECKPOINT}" ]]; then
  echo "  Exclude checkpoint: ${EXCLUDE_CHECKPOINT}"
fi

# ===============================
# 3. Activate environment and run
# ===============================
# Set environment variables for local model (GPU)
export USE_LOCAL_MODEL=true
export LOCAL_MODEL="${LOCAL_MODEL}"
export CUDA_VISIBLE_DEVICES=0

# ===============================
# 4. Create queue file
# ===============================
QUEUE_FILE="${OUTPUT_DIR}/queue_${LANGUAGE}.jsonl"
CURSOR_FILE="${LOG_DIR}/cursor_${LANGUAGE}.txt"
LOCK_FILE="${LOG_DIR}/write_lock_${LANGUAGE}.lock"
CHECKPOINT_FILE="${OUTPUT_DIR}/checkpoint.jsonl"

echo "Creating queue file for Chinese language..."
python - <<'PY' "${CSV_PATH}" "${QUEUE_FILE}" "${LANGUAGE}" "${MAX_P1}" "${EXCLUDE_CHECKPOINT}"
import json, sys, pandas as pd
import numpy as np
import os
csv_path, queue_file, language, max_p1, exclude_checkpoint = sys.argv[1:]
max_p1 = int(max_p1) if max_p1 and max_p1 != "None" else None

# Load excluded IDs from checkpoint file if it exists
excluded_ids = set()
if exclude_checkpoint and os.path.exists(exclude_checkpoint):
    print(f"Loading excluded IDs from: {exclude_checkpoint}")
    with open(exclude_checkpoint, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                    p1_id = rec.get('p1_id') or rec.get('id')
                    if p1_id:
                        excluded_ids.add(str(p1_id))
                except:
                    pass
    print(f"Found {len(excluded_ids)} already processed prompt IDs to exclude")
else:
    print("No exclude checkpoint file provided or file doesn't exist")

df = pd.read_csv(csv_path)
df_filtered = df[df['lang'] == language].copy()

print(f"Total Chinese prompts in dataset: {len(df_filtered)}")

# Exclude already processed prompts
if excluded_ids:
    initial_count = len(df_filtered)
    df_filtered = df_filtered[~df_filtered['id'].astype(str).isin(excluded_ids)]
    print(f"Excluded {initial_count - len(df_filtered)} already processed prompts")

if max_p1 and max_p1 > 0:
    # Stratified sampling: balance across categories and levels
    categories = df_filtered['type'].unique()
    levels = sorted(df_filtered['level'].dropna().unique())
    
    # Calculate samples per category (approximately equal)
    samples_per_category = max_p1 // len(categories)
    remainder = max_p1 % len(categories)
    
    selected = []
    for i, cat in enumerate(categories):
        cat_df = df_filtered[df_filtered['type'] == cat].copy()
        # Add one extra sample to first 'remainder' categories to use up total
        n_samples = samples_per_category + (1 if i < remainder else 0)
        
        if len(cat_df) >= n_samples:
            # Try to balance levels within each category
            level_counts = {}
            for level in levels:
                level_df = cat_df[cat_df['level'] == level]
                if len(level_df) > 0:
                    level_counts[level] = len(level_df)
            
            # Distribute samples across levels proportionally
            level_samples = {}
            total_levels = sum(level_counts.values())
            if total_levels > 0:
                for level, count in level_counts.items():
                    level_samples[level] = max(1, int(n_samples * count / total_levels))
                
                # Adjust to match total
                current_total = sum(level_samples.values())
                if current_total < n_samples:
                    # Add remaining to levels with most data
                    remaining = n_samples - current_total
                    sorted_levels = sorted(level_counts.items(), key=lambda x: x[1], reverse=True)
                    for level, _ in sorted_levels[:remaining]:
                        level_samples[level] = level_samples.get(level, 0) + 1
                elif current_total > n_samples:
                    # Remove from levels with most samples
                    remaining = current_total - n_samples
                    sorted_samples = sorted(level_samples.items(), key=lambda x: x[1], reverse=True)
                    for level, _ in sorted_samples[:remaining]:
                        if level_samples[level] > 1:
                            level_samples[level] -= 1
                
                # Sample from each level
                for level, n in level_samples.items():
                    level_df = cat_df[cat_df['level'] == level]
                    if len(level_df) >= n:
                        sampled = level_df.sample(n=n, random_state=42)
                        selected.append(sampled)
                    else:
                        selected.append(level_df)
            else:
                # Fallback: random sample if no level info
                sampled = cat_df.sample(n=min(n_samples, len(cat_df)), random_state=42)
                selected.append(sampled)
        else:
            # If category has fewer samples than needed, take all
            selected.append(cat_df)
    
    df_filtered = pd.concat(selected, ignore_index=True)
    # Shuffle the final selection
    df_filtered = df_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    # Trim to exact max_p1 if needed
    if len(df_filtered) > max_p1:
        df_filtered = df_filtered.head(max_p1)

with open(queue_file, "w", encoding="utf-8") as f:
    for _, row in df_filtered.iterrows():
        rec = {
            "id": str(row.get("id", "")),
            "p1": str(row.get("prompt", "")),
            "lang": str(row.get("lang", language)),
            "level": row.get("level", None),
            "subtype": str(row.get("subtype", "")) if pd.notna(row.get("subtype")) else "",
            "type": str(row.get("type", "")) if pd.notna(row.get("type")) else "",
            "source": str(row.get("source", "")) if pd.notna(row.get("source")) else "",
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Created queue file with {len(df_filtered)} Chinese records")
print(f"Category distribution: {df_filtered['type'].value_counts().to_dict()}")
print(f"Level distribution: {df_filtered['level'].value_counts().to_dict()}")

# Save reference file with selected prompts and metadata
import os
reference_file = os.path.join(os.path.dirname(queue_file), "selected_prompts_reference.jsonl")
with open(reference_file, "w", encoding="utf-8") as f:
    for _, row in df_filtered.iterrows():
        rec = {
            "id": str(row.get("id", "")),
            "p1": str(row.get("prompt", "")),
            "lang": str(row.get("lang", language)),
            "level": int(row.get("level")) if pd.notna(row.get("level")) else None,
            "subtype": str(row.get("subtype", "")) if pd.notna(row.get("subtype")) else "",
            "type": str(row.get("type", "")) if pd.notna(row.get("type")) else "",
            "source": str(row.get("source", "")) if pd.notna(row.get("source")) else "",
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# Save summary metadata
summary_file = os.path.join(os.path.dirname(queue_file), "selection_summary.json")
crosstab = pd.crosstab(df_filtered['type'], df_filtered['level'])
summary = {
    "total_selected": len(df_filtered),
    "language": language,
    "max_p1": max_p1,
    "selection_method": "stratified_sampling",
    "category_distribution": df_filtered['type'].value_counts().to_dict(),
    "level_distribution": df_filtered['level'].value_counts().to_dict(),
    "category_level_crosstab": {str(k): v.to_dict() for k, v in crosstab.iterrows()}
}
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Saved reference file: {reference_file}")
print(f"Saved selection summary: {summary_file}")
PY

if [[ ! -s "${QUEUE_FILE}" ]]; then
  echo "ERROR: Queue file is empty. Exiting."
  exit 1
fi

# Initialize cursor and lock files
echo "0" > "${CURSOR_FILE}"
: > "${LOCK_FILE}"

echo ""
echo "Queue setup complete:"
echo "  Queue file: ${QUEUE_FILE}"
echo "  Cursor file: ${CURSOR_FILE}"
echo "  Lock file: ${LOCK_FILE}"
echo "  Checkpoint file: ${CHECKPOINT_FILE}"
echo ""

# ===============================
# 5. Run with queue-based processing
# ===============================
echo "Running generate_p1_p2_outputs.py for Chinese with queue-based processing on GPU..."
echo ""

# Build command with conditional --max-p1 argument
CMD_ARGS=(
  --csv-path "${CSV_PATH}"
  --language "${LANGUAGE}"
  --pprime-file "${PPRIME_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --sanitizer-terms "${SANITIZER_TERMS}"
  --use-local-model
  --local-model "${LOCAL_MODEL}"
  --queue-file "${QUEUE_FILE}"
  --cursor-file "${CURSOR_FILE}"
  --write-lock-file "${LOCK_FILE}"
  --checkpoint-file "${CHECKPOINT_FILE}"
)

# Only add --max-p1 if it's not "None" or empty
if [[ "${MAX_P1}" != "None" && -n "${MAX_P1}" ]]; then
  CMD_ARGS+=(--max-p1 "${MAX_P1}")
fi

python src/prompt_generation/generate_p1_p2_outputs.py "${CMD_ARGS[@]}"

echo ""
echo "Job finished at: $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Logs in: ${LOG_DIR}"

