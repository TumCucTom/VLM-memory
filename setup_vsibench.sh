#!/bin/bash
#SBATCH --job-name=vsibench_setup
#SBATCH --output=logs/vsibench_setup_%j.out
#SBATCH --error=logs/vsibench_setup_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --partition=workq

source ~/miniforge3/bin/activate
conda activate vsibench
conda install -y conda-forge::pytorch conda-forge::flash-attn

echo "Installation complete."