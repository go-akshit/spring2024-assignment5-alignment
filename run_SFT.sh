#!/bin/bash
#SBATCH --job-name=SFT
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --gpus=1
#SBATCH --output=SFT_%j.out
#SBATCH --error=SFT_%j.err
python3 cs336_alignment/instruction_finetuning.py