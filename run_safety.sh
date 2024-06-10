#!/bin/bash
#SBATCH --job-name=safety
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --output=safety_%j.out
#SBATCH --error=safety_%j.err
python scripts/evaluate_safety.py --input-path <'sst_baseline_predictions.jsonl'> --model-name-or-path /home/shared/Meta-Llama-3-70B-Instruct --num-gpus 2 --output-path <'safety_baseline.jsonl'>