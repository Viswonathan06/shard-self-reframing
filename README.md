# SHARD: Safe and Helpful Alignment via Self-Reframing Distillation

> **arXiv:** [Link coming soon]  
> **Authors:** Viswonathan Manoranjan\*, Amogh Gupta\*, Anvesh Rao Vijjini, Thomas Hofweber, Snigdha Chaturvedi  
> **Institution:** University of North Carolina at Chapel Hill  
> \* Equal contribution

---

## Abstract

We present **SHARD**, a pipeline for aligning large language models toward safe, helpful responses without resorting to blunt refusals. Given a potentially harmful prompt (P1), SHARD rewrites it into a safety-aligned alternative (P2) using LLM-generated category-specific guidelines, then fine-tunes models on (P1, P2) preference pairs via supervised fine-tuning (SFT) and direct preference optimization (DPO). Evaluated across seven models (Llama, Qwen, Mistral, Phi-4) on the LinguaSafe and DoNotAnswer benchmarks, SHARD consistently improves helpfulness while maintaining harmlessness, with the self-distillation variant achieving 78–90% judge-preferred win rates over the helpful-assistant baseline on held-out test sets.

---

## Repository Structure

```
├── dataset/                        # Evaluation benchmarks
│   ├── linguasafe.csv              # LinguaSafe prompts (id, prompt, lang, type, level, source)
│   ├── linguasafe_train.jsonl      # LinguaSafe training split
│   └── donotanswer_no_outputs.jsonl  # DoNotAnswer prompts
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
│   ├── train_sft_linguasafe.py     # Full SFT training (8B, FSDP/QLoRA)
│   ├── train_dpo_linguasafe.py     # DPO fine-tuning
│   ├── inference_sft_vllm.py       # vLLM inference with SFT adapter
│   ├── judge_sft_vs_baseline_gemma4.py  # Gemma-4 judge evaluation
│   ├── shell/                      # Helper shell scripts
│   └── slurm/                      # End-to-end SLURM pipelines (SFT+infer+judge)
├── config/                         # Language config (en)
├── env/                            # Frozen cluster environment files
├── notebooks/                      # Analysis notebooks
├── requirements.txt                # Python dependencies
├── CITATION.cff
└── LICENSE
```

---

## Setup

### 1. Environment

```bash
git clone https://github.com/Viswonathan06/shard-self-reframing.git
cd shard-self-reframing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# For SFT/DPO training:
pip install peft trl bitsandbytes datasets
```

Model weights are loaded from HuggingFace Hub at runtime. Set `HF_TOKEN` for gated models (Llama):
```bash
export HF_TOKEN=hf_your_token_here
export HF_HOME=/path/to/your/hf_cache
```

---

## Reproducing Main Results

### Step 1 — Generate Safety Guidelines

```bash
python src/prompt_generation/generate_category_guidelines.py \
  --csv-path dataset/linguasafe.csv \
  --language en \
  --output-dir output/guidelines \
  --universal-guidelines src/prompts/guidelines.txt \
  --template src/prompts/category_guidelines.jsonl \
  --use-local-model \
  --local-model <path-to-model>
```

### Step 2 — P1 → P2 Rewriting

```bash
python src/prompt_generation/run_p1_to_p2_guidelines.py \
  --csv-path dataset/linguasafe.csv \
  --language en \
  --output-dir output/p1_to_p2/guidelines_en \
  --guidelines-dir output/guidelines \
  --use-local-model \
  --local-model <path-to-model>
```

### Step 3 — SFT Training

```bash
# Small models (8B) — multi-GPU via torchrun
torchrun --nproc_per_node=4 --master_port=29500 scripts/train_sft_linguasafe.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --train-jsonl data/sft_dpo/train.jsonl \
  --val-jsonl   data/sft_dpo/val.jsonl \
  --output-dir  output/SFT/self_model/llama8b/model \
  --epochs 3 --lr 2e-5 --batch-size 2 --grad-accum 8

# Large models (27B–70B) — QLoRA via torchrun
torchrun --nproc_per_node=4 --master_port=29500 scripts/sft_qlora.py \
  --base-model <path-to-model> \
  --train-jsonl data/sft_dpo/train.jsonl \
  --val-jsonl   data/sft_dpo/val.jsonl \
  --output-dir  output/SFT/self_model/qwen35_27b/model \
  --epochs 3 --learning-rate 1e-4 --per-device-batch-size 1 --grad-accum 16
```

### Step 4 — Inference

```bash
python scripts/inference_sft_vllm.py \
  --base-model  <path-to-base-model> \
  --sft-adapter output/SFT/self_model/llama8b/model \
  --input-data  output/SFT/self_model/llama8b/test.jsonl \
  --output      output/SFT/self_model/llama8b/inference/llama8b.jsonl \
  --tensor-parallel-size 4
```

### Step 5 — Gemma-4 Judge Evaluation

```bash
python scripts/judge_sft_vs_baseline_gemma4.py \
  --sft-responses output/SFT/self_model/llama8b/inference/llama8b.jsonl \
  --baseline      <baseline-responses.jsonl> \
  --judge-model   <path-to-gemma4-31B-it> \
  --output-dir    output/SFT/self_model/llama8b/inference/vs_baseline_judge
```

---

## SLURM (GPU Cluster)

End-to-end jobs (train → infer → judge in a single script) are in `scripts/slurm/`. Adjust `--nodelist`/`--exclude` to match your cluster, and set `HF_HOME` to your model cache:

```bash
# Self-model SFT + inference + judge — Llama-3.1-8B
sbatch scripts/slurm/job_rational_sft_llama8b.slurm

# Self-model SFT + inference + judge — Qwen3.5-9B
sbatch scripts/slurm/job_rational_sft_qwen9b.slurm

# 122B teacher SFT pipeline — Llama-3.1-8B student
sbatch scripts/slurm/job_rational_sft_122b_llama8b.slurm

# Cross-model comparison (set MODEL= to target model)
MODEL=qwen35_9b sbatch scripts/slurm/job_comparison_pipeline.slurm
```

---

## Model Checkpoints

Fine-tuned LoRA adapters will be released on HuggingFace Hub at:  
`https://huggingface.co/Viswonathan06/SHARD-adapters` *(coming soon)*

---

## Citation

```bibtex
@article{manoranjan2025shard,
  title     = {{SHARD}: Safe and Helpful Alignment via Self-Reframing Distillation},
  author    = {Manoranjan, Viswonathan and Gupta, Amogh and Vijjini, Anvesh Rao and Hofweber, Thomas and Chaturvedi, Snigdha},
  journal   = {arXiv preprint},
  year      = {2025},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## License

This repository is released under the [MIT License](LICENSE).

The LinguaSafe dataset (`dataset/linguasafe.csv`) and DoNotAnswer dataset (`dataset/donotanswer_no_outputs.jsonl`) retain their respective original licenses — please cite both papers if you use them.
