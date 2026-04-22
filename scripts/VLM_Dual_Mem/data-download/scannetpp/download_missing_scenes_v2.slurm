#!/bin/bash
#SBATCH --job-name=download_scannetpp_missing
#SBATCH --output=logs/download_scannetpp_missing_%j.out
#SBATCH --error=logs/download_scannetpp_missing_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

source ~/miniforge3/bin/activate
conda activate vlm3r

cd /home/b6ar/trvbale.b6ar/scratch/VLM-memory

CONFIG="scripts/VLM_Dual_Mem/data-download/scannetpp/data_download/download_scannetpp_missing.yml"
WRAPPER="scripts/VLM_Dual_Mem/data-download/scannetpp/download_scannetpp_wrapper.py"

echo "Starting download of 389 missing ScanNet++ scenes..."
python "$WRAPPER" "$CONFIG"

echo "End Time: $(date)"