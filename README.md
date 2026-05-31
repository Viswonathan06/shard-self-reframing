# SHARD: Safe and Helpful Alignment via Self-Reframing Distillation

> **arXiv:** [Link coming soon]  
> **Authors:** Viswonathan Manoranjan, [co-authors TBD]  
> **Institution:** University of North Carolina at Chapel Hill

---

## Abstract

We present **SHARD**, a pipeline for aligning large language models toward safe, helpful responses without resorting to blunt refusals. Given a potentially harmful prompt (P1), SHARD rewrites it into a safety-aligned alternative (P2) using LLM-generated category-specific guidelines, then fine-tunes models on (P1, P2) preference pairs via supervised fine-tuning (SFT) and direct preference optimization (DPO). Evaluated across seven models (Llama, Qwen, Mistral, Phi-4) on the LinguaSafe benchmark in English, Chinese, and Arabic, SHARD consistently improves helpfulness while maintaining harmlessness, with the self-distillation variant achieving 78–90% judge-preferred win rates over the helpful-assistant baseline on held-out test sets.

---

## Repository Structure

```
├── dataset/                        # LinguaSafe evaluation benchmark
│   ├── linguasafe.csv              # P1 prompts (id, prompt, lang, type, level, source)
│   └── linguasafe_train.jsonl      # Training split
├── src/
│   ├── prompt_generation/          # P1→P2 rewriting pipeline
│   │   ├── generate_category_guidelines.py
│   │   └── run_p1_to_p2_guidelines.py
│   ├── evaluation_ranking/         # Judge + ranking scripts
│   ├── prompts/                    # System prompts and guidelines templates
│   ├── slurm/                      # SLURM job scripts for GPU cluster
│   └── utils/                      # Shared utilities
├── scripts/
│   ├── sft_qlora.py                # QLoRA SFT for 27B–70B models
│   ├── train_sft_linguasafe.py     # Full SFT training (8B–70B, FSDP/QLoRA)
│   ├── train_dpo_linguasafe.py     # DPO fine-tuning
│   ├── inference_sft_vllm.py       # vLLM inference with SFT adapter
│   └── judge_sft_vs_baseline_gemma4.py  # Gemma-4 judge evaluation
├── reports/
│   ├── plot_rational_vs_baseline.py   # Figure: RATIONAL SFT vs baseline
│   ├── plot_rational_vs_selfsft.py    # Figure: RATIONAL vs SHARD (self-model)
│   ├── plot_sft_results.py            # Figure: SHARD SFT helpfulness
│   └── significance_ttest.py          # Statistical significance tests
├── safeshift_rl/                   # Experimental: GRPO reward modeling (not in paper)
├── appendix/                       # Additional prompts, preference data, setup utils
├── config/                         # Language configs (en/zh/ar)
├── data/                           # SFT training data splits
├── notebooks/                      # Analysis notebooks
├── requirements.txt                # Python dependencies
└── LICENSE
```

---

## Setup

### 1. Environment

```bash
git clone https://github.com/<org>/SHARD.git
cd SHARD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# For SFT/DPO training (additional deps):
pip install peft trl bitsandbytes datasets

# For vLLM inference (Qwen3.5 + Gemma-4):
# See appendix/setup_utilities/install_vllm_qwen35_venv.sh
```

### 2. Credentials

Copy and fill in your credentials:
```bash
cp .env.example .env   # then edit .env with your OpenAI key (optional — only needed for API-mode P2 generation)
```

Model weights are loaded from HuggingFace Hub at runtime. Set `HF_TOKEN` if accessing gated models (Llama):
```bash
export HF_TOKEN=hf_your_token_here
```

---

## Reproducing Main Results

### Step 1 — Generate Safety Guidelines

```bash
python src/prompt_generation/generate_category_guidelines.py \
  --linguasafe dataset/linguasafe.csv \
  --output-dir output/guidelines \
  --universal-guidelines src/prompts/guidelines.txt \
  --template src/prompts/category_guidelines.jsonl \
  --use-api   # or --use-local-model with --local-model <path>
```

### Step 2 — P1 → P2 Rewriting

```bash
python src/prompt_generation/run_p1_to_p2_guidelines.py \
  --csv-path dataset/linguasafe.csv \
  --language en \
  --output-dir output/p1_to_p2/guidelines_en \
  --guidelines-dir output/guidelines
```

Repeat with `--language zh` and `--language ar` for multilingual results.

### Step 3 — SFT Training (Self-Distillation)

```bash
# Small models (8B) — single or multi-GPU via torchrun
torchrun --nproc_per_node=4 scripts/train_sft_linguasafe.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --train-jsonl data/sft_dpo/train.jsonl \
  --val-jsonl   data/sft_dpo/val.jsonl \
  --output-dir  output/SFT/self_model/llama8b/model \
  --epochs 3 --lr 2e-5 --lora-r 16

# Large models (27B–70B) — QLoRA
torchrun --nproc_per_node=4 scripts/sft_qlora.py \
  --base-model Qwen/Qwen3.5-27B \
  --train-jsonl data/sft_dpo/train.jsonl \
  --val-jsonl   data/sft_dpo/val.jsonl \
  --output-dir  output/SFT/self_model/qwen35_27b/model
```

### Step 4 — Inference

```bash
python scripts/inference_sft_vllm.py \
  --base-model  <hf-model-path> \
  --adapter-dir output/SFT/self_model/llama8b/model \
  --test-jsonl  output/SFT/self_model/llama8b/test.jsonl \
  --output-jsonl output/SFT/self_model/llama8b/inference/llama8b.jsonl
```

### Step 5 — Gemma-4 Judge Evaluation

```bash
python scripts/judge_sft_vs_baseline_gemma4.py \
  --sft-responses  output/SFT/self_model/llama8b/inference/llama8b.jsonl \
  --baseline       <baseline-responses.jsonl> \
  --judge-model    google/gemma-4-31B-it \
  --output-dir     output/SFT/self_model/llama8b/inference/vs_baseline_judge
```

### Step 6 — Reproduce Figures

```bash
cd reports/
python plot_sft_results.py          # Figure: SHARD SFT vs baseline + vs self-model
python plot_rational_vs_baseline.py # Figure: RATIONAL SFT vs baseline
python plot_rational_vs_selfsft.py  # Figure: RATIONAL vs SHARD head-to-head
python significance_ttest.py        # Statistical significance (3-split paired t-test)
```

---

## SLURM (GPU Cluster)

All GPU jobs are in `scripts/slurm/`. Adjust `--nodelist`/`--exclude` to match your cluster:

```bash
# Self-model SFT + inference + judge (e.g. Qwen3.5-9B)
sbatch scripts/slurm/job_rational_sft_qwen9b.slurm

# 122B teacher SFT pipeline
sbatch scripts/slurm/job_rational_sft_122b_llama8b.slurm

# Cross-model comparison
MODEL=qwen35_9b sbatch scripts/slurm/job_comparison_pipeline.slurm
```

---

## Model Checkpoints

Fine-tuned LoRA adapters will be released on HuggingFace Hub at:  
`https://huggingface.co/<org>/SHARD-adapters` *(coming soon)*

In the meantime, reproduce adapters using the training commands above.

---

## Citation

```bibtex
@article{manoranjan2025shard,
  title     = {{SHARD}: Safe and Helpful Alignment via Self-Reframing Distillation},
  author    = {Manoranjan, Viswonathan and others},
  journal   = {arXiv preprint},
  year      = {2025},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).

The LinguaSafe dataset (`dataset/linguasafe.csv`) retains its original license — please cite the LinguaSafe paper if you use it.
