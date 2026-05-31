#!/usr/bin/bash
# Rational-SFT pipeline for Phi-4.
# Stage -1: rational inference with --prompt-as-user -> phi4_user_prompt/
# Stage  0: build SFT splits from those outputs
# Stage  1: QLoRA training
# Stage  2: inference
# Stage  3: Gemma-4 judge
#
# Usage:
#   sbatch scripts/slurm/job_rational_sft_phi4.slurm
# Skip flags:
#   SKIP_RATIONAL_INFER=1  SKIP_TRAIN=1  SKIP_INFER=1

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH -p a6000
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:4
#SBATCH --exclude=gpu1,gpu3
#SBATCH --time=24:00:00
#SBATCH --job-name=rational_sft_phi4
#SBATCH --output=$PROJECT_ROOT/logs/rational_sft_phi4_%j.out
#SBATCH --error=$PROJECT_ROOT/logs/rational_sft_phi4_%j.err

set -euo pipefail

ROOT="$PROJECT_ROOT"
cd "$ROOT"
mkdir -p logs

PYTHON="$CONDA_PREFIX/envs/vllm-serve/bin/python"
TORCHRUN="$CONDA_PREFIX/envs/vllm-serve/bin/torchrun"
source ~/.bashrc 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_BUILD_AIO=0
export DS_SKIP_CUDA_CHECK=1
export TMPDIR=/tmp
export VLLM_USE_V1=0
export VLLM_TRUST_REMOTE_CODE=1

BASE_MODEL="$HF_HOME/models--microsoft--phi-4/snapshots/932b33c0ec9ca189badeb22480721a8de9d0e006"
[[ -f "$BASE_MODEL/config.json" ]] || { echo "ERROR: phi4 not found at $BASE_MODEL"; exit 1; }

SELF_MODEL_DIR="$ROOT/output/SFT/self_model/phi4"
DATA_DIR="$ROOT/output/SFT/rational_sft/phi4"

# rational inference outputs (user_prompt variant)
RATIONAL_DNA_INPUT="$ROOT/output/DNA_Per_Model_Experiments/helpful_assistant_multimodel/phi4/p1_baseline_outputs.jsonl"
RATIONAL_LS_INPUT="$ROOT/output/Linguasafe_Experiments/helpful_assistant_multimodel/phi4/p1_baseline_outputs.jsonl"
RATIONAL_DNA="$ROOT/output/rational_baseline/phi4_user_prompt/dna/inference_compliance.jsonl"
RATIONAL_LS="$ROOT/output/rational_baseline/phi4_user_prompt/linguasafe/inference_compliance.jsonl"

BASELINE="$SELF_MODEL_DIR/modelwise_test_baselines/phi4.jsonl"
SFT_ADAPTER="$DATA_DIR/models/phi4"
INFERENCE_OUT="$DATA_DIR/inference/phi4.jsonl"
JUDGE_DIR="$DATA_DIR/judge/phi4"

[[ -f "$RATIONAL_DNA_INPUT" ]] || { echo "ERROR: DNA input not found: $RATIONAL_DNA_INPUT"; exit 1; }
[[ -f "$RATIONAL_LS_INPUT"  ]] || { echo "ERROR: LS input not found: $RATIONAL_LS_INPUT";   exit 1; }
[[ -f "$BASELINE"           ]] || { echo "ERROR: baseline not found: $BASELINE"; exit 1; }

_GEMMA4_BASE="$HF_HOME/models--google--gemma-4-31B-it/snapshots"
GEMMA4_PATH=$(find "$_GEMMA4_BASE" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -r | head -1)
[[ -f "$GEMMA4_PATH/config.json" ]] || { echo "ERROR: Gemma-4 not found"; exit 1; }

mkdir -p "$DATA_DIR" "$SFT_ADAPTER" "$(dirname "$INFERENCE_OUT")" "$JUDGE_DIR" \
         "$(dirname "$RATIONAL_DNA")" "$(dirname "$RATIONAL_LS")"

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
MASTER_PORT=$(( (SLURM_JOB_ID % 10000) + 20000 ))
# phi4 has 10 KV heads — TP must be 1
INFER_TP=1

echo "========================================================"
echo "Rational-SFT pipeline: phi4"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Start:     $(date)"
echo "Base:      $BASE_MODEL"
echo "GPUs:      $NUM_GPUS  (infer TP=$INFER_TP)"
echo "========================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── gpu1 broken-GPU filter ─────────────────────────────────────────────────────
if [[ "$(hostname -s)" == "gpu1" ]]; then
    CUDA_VISIBLE_DEVICES=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -v '^3$' | grep -v '^7$' | paste -sd ',')
    export CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
fi

# ── Stage -1: Rational inference (--prompt-as-user) ───────────────────────────
SKIP_RATIONAL_INFER="${SKIP_RATIONAL_INFER:-0}"
if [[ "$SKIP_RATIONAL_INFER" == "1" ]]; then
    [[ -f "$RATIONAL_DNA" ]] || { echo "ERROR: SKIP_RATIONAL_INFER=1 but missing: $RATIONAL_DNA"; exit 1; }
    [[ -f "$RATIONAL_LS"  ]] || { echo "ERROR: SKIP_RATIONAL_INFER=1 but missing: $RATIONAL_LS";  exit 1; }
    echo ">>> STAGE -1: Skipped (SKIP_RATIONAL_INFER=1)"
else
    echo ">>> STAGE -1: Rational inference (prompt-as-user)"
    for pair in "dna:$RATIONAL_DNA_INPUT:$RATIONAL_DNA" "linguasafe:$RATIONAL_LS_INPUT:$RATIONAL_LS"; do
        dataset="${pair%%:*}"; rest="${pair#*:}"; input="${rest%%:*}"; output="${rest#*:}"
        mkdir -p "$(dirname "$output")"
        echo "  Dataset: $dataset  Input: $(wc -l < "$input") rows"
        "$PYTHON" scripts/rational_baseline/rational_inference_vllm.py \
            --model                  "$BASE_MODEL" \
            --input                  "$input" \
            --output                 "$output" \
            --model-tag              "phi4_user_prompt" \
            --mode                   compliance \
            --prompt-as-user \
            --tensor-parallel-size   "$INFER_TP" \
            --gpu-memory-utilization 0.85 \
            --max-model-len          4096 \
            --max-new-tokens         4096 \
            --temperature            0.3 \
            --batch-size             128
        echo "  Done: $(wc -l < "$output") rows -> $output"
    done
    echo ">>> Stage -1 complete"
fi
echo ""

# ── Stage 0: Build train/val/test split ───────────────────────────────────────
echo ">>> STAGE 0: Building rational SFT splits for phi4"
"$PYTHON" - <<'PYEOF'
import json
from pathlib import Path

ROOT = Path("$PROJECT_ROOT")
SELF_MODEL_DIR = ROOT / "output/SFT/self_model/phi4"
OUT_DIR        = ROOT / "output/SFT/rational_sft/phi4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RATIONAL_FILES = [
    ROOT / "output/rational_baseline/phi4_user_prompt/dna/inference_compliance.jsonl",
    ROOT / "output/rational_baseline/phi4_user_prompt/linguasafe/inference_compliance.jsonl",
]

SYSTEM_PROMPT = (
    "You are a helpful and safe assistant. When a user asks about a sensitive topic, "
    "respond helpfully while avoiding harm. Always address the user's underlying need "
    "even if you cannot address the exact phrasing of their request."
)

def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

train_ids = {str(r["p1_id"]) for r in load_jsonl(SELF_MODEL_DIR / "train.jsonl")}
val_ids   = {str(r["p1_id"]) for r in load_jsonl(SELF_MODEL_DIR / "val.jsonl")}
test_ids  = {str(r["p1_id"]) for r in load_jsonl(SELF_MODEL_DIR / "test.jsonl")}
print(f"Split IDs — train:{len(train_ids)}  val:{len(val_ids)}  test:{len(test_ids)}")

rational = {}
for f in RATIONAL_FILES:
    for r in load_jsonl(f):
        pid = str(r["p1_id"])
        output = (r.get("rational_full_output") or r.get("rational_response") or "").strip()
        if output:
            rational[pid] = {"p1": r["p1"], "output": output,
                             "dataset": r.get("dataset", ""), "level": r.get("level"),
                             "category": r.get("category", "")}

print(f"Rational outputs loaded: {len(rational)}")

splits = {"train": [], "val": [], "test": [], "all": []}
for pid, rec in rational.items():
    row = {
        "p1_id":    pid,
        "dataset":  rec["dataset"],
        "level":    rec["level"],
        "category": rec["category"],
        "p1":       rec["p1"],
        "o2":       rec["output"],
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": rec["p1"]},
            {"role": "assistant", "content": rec["output"]},
        ],
    }
    splits["all"].append(row)
    if pid in train_ids:   splits["train"].append(row)
    elif pid in val_ids:   splits["val"].append(row)
    elif pid in test_ids:  splits["test"].append(row)

for split, rows in splits.items():
    out = OUT_DIR / f"{split}.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {split}: {len(rows)} rows -> {out}")

stats = {"model": "phi4", "method": "rational_sft",
         "note": "same p1_id splits as self_model/phi4; rational via prompt-as-user",
         **{k: len(v) for k, v in splits.items()}}
(OUT_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2))
print("Done building splits.")
PYEOF
echo ">>> Stage 0 complete"
echo ""

TRAIN_JSONL="$DATA_DIR/train.jsonl"
VAL_JSONL="$DATA_DIR/val.jsonl"
TEST_JSONL="$DATA_DIR/test.jsonl"

# ── Stage 1: QLoRA training ───────────────────────────────────────────────────
SKIP_TRAIN="${SKIP_TRAIN:-0}"
if [[ "$SKIP_TRAIN" == "1" ]]; then
    [[ -f "$SFT_ADAPTER/adapter_config.json" ]] || \
        { echo "ERROR: SKIP_TRAIN=1 but adapter not found at $SFT_ADAPTER"; exit 1; }
    echo ">>> STAGE 1: Skipped (SKIP_TRAIN=1)"
else
    echo ">>> STAGE 1: QLoRA training"
    echo "  Train: $(wc -l < "$TRAIN_JSONL") rows  Val: $(wc -l < "$VAL_JSONL") rows"
    "$TORCHRUN" --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" scripts/sft_qlora.py \
        --base-model            "$BASE_MODEL" \
        --train-jsonl           "$TRAIN_JSONL" \
        --val-jsonl             "$VAL_JSONL" \
        --output-dir            "$SFT_ADAPTER" \
        --epochs                3 \
        --per-device-batch-size 1 \
        --grad-accum            $(( 64 / NUM_GPUS )) \
        --learning-rate         1e-4 \
        --warmup-ratio          0.05 \
        --lora-r                16 \
        --lora-alpha            32 \
        --lora-targets          "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
        --max-length            4096 \
        --logging-steps         1 \
        --seed                  42 \
        --wandb-project         linguasafe-sft-dpo \
        --run-name              "qlora_rational_phi4"
    echo ">>> Stage 1 complete — adapter at $SFT_ADAPTER"
fi
echo ""

# ── Stage 2: Inference ────────────────────────────────────────────────────────
SKIP_INFER="${SKIP_INFER:-0}"
if [[ "$SKIP_INFER" == "1" ]]; then
    [[ -f "$INFERENCE_OUT" ]] || { echo "ERROR: SKIP_INFER=1 but output not found: $INFERENCE_OUT"; exit 1; }
    echo ">>> STAGE 2: Skipped (SKIP_INFER=1)"
else
    echo ">>> STAGE 2: Inference (TP=$INFER_TP)"
    "$PYTHON" scripts/inference_sft_qwen_vllm.py \
        --base-model             "$BASE_MODEL" \
        --sft-adapter            "$SFT_ADAPTER" \
        --input-data             "$TEST_JSONL" \
        --output                 "$INFERENCE_OUT" \
        --tensor-parallel-size   "$INFER_TP" \
        --gpu-memory-utilization 0.85 \
        --max-model-len          4096 \
        --max-new-tokens         2048 \
        --temperature            0.3 \
        --lora-rank              16 \
        --system-prompt          "You are a helpful assistant."
    echo ">>> Stage 2 complete — $(wc -l < "$INFERENCE_OUT") rows -> $INFERENCE_OUT"
fi
echo ""

sleep 30

# ── Stage 3: Gemma-4 judge ────────────────────────────────────────────────────
echo ">>> STAGE 3: Judge"
_SLURM_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
source scripts/shell/setup_gemma4_vllm_env.sh 2>/dev/null || true
[[ -n "$_SLURM_CUDA_DEVICES" ]] && export CUDA_VISIBLE_DEVICES="$_SLURM_CUDA_DEVICES"
JUDGE_TP=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
_tp=1; for _p2 in 2 4 8; do (( _p2 <= JUDGE_TP )) && _tp=$_p2; done; JUDGE_TP=$_tp
export VLLM_TENSOR_PARALLEL_SIZE=$JUDGE_TP
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NUM_BATCHED_TOKENS=16384
export VLLM_ENFORCE_EAGER=0

"$PYTHON" scripts/judge_sft_vs_baseline_gemma4.py \
    --sft-responses  "$INFERENCE_OUT" \
    --baseline       "$BASELINE" \
    --output-dir     "$JUDGE_DIR" \
    --judge-model    "$GEMMA4_PATH" \
    --model-tag      "phi4_rational_sft" \
    --batch-size     8 \
    --max-new-tokens 512 \
    --temperature    0.0
echo ">>> Stage 3 complete — $JUDGE_DIR"

echo ""
echo "========================================================"
echo "Rational-SFT phi4 pipeline done at: $(date)"
echo "========================================================"
