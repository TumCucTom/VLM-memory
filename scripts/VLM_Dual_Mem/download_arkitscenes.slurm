#!/bin/bash
#SBATCH --job-name=arkitscenes_download
#SBATCH --output=logs/download_arkitscenes_%j.out
#SBATCH --error=logs/download_arkitscenes_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=compute
#SBATCH --account=coms037985

# Download ARKitScenes data using SLURM
# This is a long-running I/O-bound task that should not run on login nodes
# The dataset is several hundred GB and may take hours or days to download

echo "=========================================="
echo "ARKitScenes Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Activate Python virtual environment (needed for pandas and other dependencies)
source ~/venvs/vlm3r/bin/activate

# Verify Python and pandas
echo "Python version: $(python --version)"
echo "Pandas available: $(python -c 'import pandas; print(pandas.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo ""

# Change to project directory
cd ~/VLM-memory

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the download script
bash scripts/VLM_Dual_Mem/download_arkitscenes.sh

echo ""
echo "=========================================="
echo "Download job completed"
echo "=========================================="
echo "End Time: $(date)"
echo ""

