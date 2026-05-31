#!/usr/bin/env python3
"""
Check DNA experiment data completeness across all models
"""

import json
import os
from pathlib import Path

def check_experiment_data():
    """Check which models have complete data for evaluation"""
    
    root = Path("$PROJECT_ROOT")
    experiments_dir = root / "output" / "DNA_New_Experiments"
    
    models = ["llama8b", "llama33_70b", "mistral_7b", "mistral_24b", "qwen35_9b", "qwen35_27b"]
    experiments = {
        "baseline": "dna_baseline_multimodel",
        "benign_intent": "dna_benign_intent_multimodel", 
        "benign_intent_guidelines": "dna_benign_intent_guidelines_multimodel"
    }
    
    print("=== DNA Experiment Data Check ===")
    print()
    
    # Check each model
    model_status = {}
    
    for model in models:
        print(f"📊 Model: {model}")
        model_data = {}
        
        for exp_name, exp_dir in experiments.items():
            file_path = experiments_dir / exp_dir / model / "p1_baseline_outputs.jsonl"
            
            if file_path.exists():
                try:
                    # Count records
                    with open(file_path, 'r') as f:
                        records = [json.loads(line) for line in f if line.strip()]
                    
                    record_count = len(records)
                    file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                    
                    model_data[exp_name] = {
                        "exists": True,
                        "records": record_count,
                        "size_mb": round(file_size, 1),
                        "path": str(file_path)
                    }
                    
                    print(f"  ✅ {exp_name}: {record_count} records ({file_size:.1f} MB)")
                    
                except Exception as e:
                    model_data[exp_name] = {"exists": False, "error": str(e)}
                    print(f"  ❌ {exp_name}: Error reading file - {e}")
            else:
                model_data[exp_name] = {"exists": False, "error": "File not found"}
                print(f"  ❌ {exp_name}: File not found")
        
        model_status[model] = model_data
        
        # Check if model is ready for evaluation
        all_exist = all(data.get("exists", False) for data in model_data.values())
        record_counts = [data.get("records", 0) for data in model_data.values()]
        consistent_records = len(set(record_counts)) <= 1 if record_counts else False
        
        if all_exist and consistent_records:
            print(f"  🚀 Ready for evaluation!")
        elif all_exist:
            print(f"  ⚠️  Files exist but record counts inconsistent: {record_counts}")
        else:
            print(f"  ❌ Not ready - missing files")
        
        print()
    
    # Summary
    ready_models = []
    partial_models = []
    missing_models = []
    
    for model, data in model_status.items():
        all_exist = all(exp_data.get("exists", False) for exp_data in data.values())
        if all_exist:
            ready_models.append(model)
        elif any(exp_data.get("exists", False) for exp_data in data.values()):
            partial_models.append(model)
        else:
            missing_models.append(model)
    
    print("=== Summary ===")
    print(f"✅ Ready for evaluation: {len(ready_models)} models")
    if ready_models:
        print(f"   {', '.join(ready_models)}")
    
    print(f"⚠️  Partially complete: {len(partial_models)} models")
    if partial_models:
        print(f"   {', '.join(partial_models)}")
    
    print(f"❌ Missing data: {len(missing_models)} models")
    if missing_models:
        print(f"   {', '.join(missing_models)}")
    
    print()
    
    # Evaluation commands
    if ready_models:
        print("=== Ready to Evaluate ===")
        print("Option 1 - All models at once:")
        print("  bash scripts/shell/submit_all_model_evaluations.sh")
        print()
        print("Option 2 - Individual models:")
        for model in ready_models:
            print(f"  MODEL={model} sbatch src/slurm/evaluation/job_eval_baseline_vs_benign_intent_gemma4.slurm")
            print(f"  MODEL={model} sbatch src/slurm/evaluation/job_eval_baseline_vs_benign_intent_guidelines_gemma4.slurm")
        print()
    
    return model_status

if __name__ == "__main__":
    check_experiment_data()