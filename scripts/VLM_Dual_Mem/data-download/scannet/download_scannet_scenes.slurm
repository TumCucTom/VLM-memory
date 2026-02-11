#!/bin/bash
#SBATCH --job-name=download_scannet
#SBATCH --output=logs/download_scannet_%j.out
#SBATCH --error=logs/download_scannet_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=workq

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Activate Python virtual environment (needed for pexpect)
source ~/miniforge3/bin/activate
conda activate vlm3r

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the download script
bash scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_scenes.sh

echo "End Time: $(date)"
echo "Job completed"
