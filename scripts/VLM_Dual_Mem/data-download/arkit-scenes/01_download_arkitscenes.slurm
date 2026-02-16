#!/bin/bash
#SBATCH --job-name=arkit_download
#SBATCH --output=logs/download_arkitscenes_%j.out
#SBATCH --error=logs/download_arkitscenes_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=workq

# Download ARKitScenes raw data and metadata to data/raw_arkitscenes/raw.
# Long-running I/O-bound job; run after configuring PROJECT_DIR if needed.

echo "=========================================="
echo "ARKitScenes Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Activate Python virtual environment
source ~/miniforge3/bin/activate
conda activate vlm3r

# Project root (change if not running from workspace root)
PROJECT_DIR="${PROJECT_DIR:-/home/u5fj/trvbale.u5fj/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs

bash scripts/VLM_Dual_Mem/data-download/arkit-scenes/download_arkitscenes.sh

echo ""
echo "End: $(date)"
echo "Job completed"
