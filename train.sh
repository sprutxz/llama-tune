#!/bin/bash --login
#SBATCH --job-name=train
#SBATCH -o slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -w ilab3

cd ~/dev/llama-tune
source .venv/bin/activate

torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    main.py \
    --model_name meta-llama/Llama-3.2-11B-Vision-Instruct
