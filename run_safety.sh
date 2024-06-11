#!/bin/bash
#SBATCH --job-name=safety_sst
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --output=safety_sst_%j.out
#SBATCH --error=safety_sst_%j.err
python scripts/evaluate_safety.py --input-path 'sst_sft_predictions.jsonl' --model-name-or-path /home/shared/Meta-Llama-3-70B-Instruct --num-gpus 2 --output-path 'safety_sst.jsonl'