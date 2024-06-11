#!/bin/bash
#SBATCH --job-name=alpaca_eval_sft
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --gpus=2
#SBATCH --output=alpaca_eval_sft_%j.out
#SBATCH --error=alpaca_eval_sft_%j.err
alpaca_eval --model_outputs 'alpaca_eval_predictions_sft.json' --annotators_config 'scripts/alpaca_eval_vllm_llama3_70b_fn' --base-dir '.'
