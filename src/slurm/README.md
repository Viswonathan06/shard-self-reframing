# SLURM Job Scripts

Job scripts are organized by pipeline stage.

## Directory structure

| Directory | Description |
|-----------|-------------|
| **baselines/** | P1 baseline runs (model responses directly from P1 prompts) |
| **p2/** | P2 runs (model responses from filtered safe rewrite prompts) |
| **guidelines/** | Category guidelines generation, P1→P2 conversion |
| **pipeline/** | P1-P2 outputs (pprime-based), judge, rank, resume, baseline queue |
| **training/** | Classifier training |

## Usage

### Baselines (P1 prompts)
```bash
sbatch src/slurm/baselines/job_llama8b_gpu.slurm
sbatch src/slurm/baselines/job_mistral7b_gpu.slurm
sbatch src/slurm/baselines/job_qwen7b_gpu.slurm
# Or all three:
sbatch src/slurm/baselines/job_multi_model_gpu.slurm
```

### P2 runs (safe rewrite prompts)
```bash
sbatch src/slurm/p2/job_llama8b_gpu.slurm
sbatch src/slurm/p2/job_mistral7b_gpu.slurm
sbatch src/slurm/p2/job_qwen7b_gpu.slurm
```

**Prerequisite:** Run guidelines pipeline and filter refusals first:
1. Generate guidelines → P1→P2 conversion → `output/p1_to_p2/p1_p2_guidelines/p1_p2_outputs.jsonl`
2. Filter refusals: `python scripts/filter_p2_refusals.py` → `output/p1_to_p2/p1_p2_guidelines/p1_p2_outputs_filtered.jsonl`

### Guidelines
```bash
sbatch src/slurm/guidelines/job_generate_category_guidelines_gpu.slurm
sbatch src/slurm/guidelines/job_p1_to_p2_guidelines_gpu.slurm
```

**Note:** Update `ROOT_DIR` in each script to match your project path.
