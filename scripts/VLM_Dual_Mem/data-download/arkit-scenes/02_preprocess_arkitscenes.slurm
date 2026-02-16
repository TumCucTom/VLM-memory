#!/bin/bash
#SBATCH --job-name=arkit_preprocess
#SBATCH --output=logs/preprocess_arkitscenes_%j.out
#SBATCH --error=logs/preprocess_arkitscenes_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=workq

# Preprocess ARKitScenes: splits from metadata, scene metadata, mov->mp4, copy to data/vlm_3r_data/arkitscenes/videos/.
# Run after 01_download_arkitscenes.slurm (or after raw data + metadata.csv are in place).

echo "=========================================="
echo "ARKitScenes Preprocess Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

if [ -d "$HOME/miniforge3" ]; then
    source "$HOME/miniforge3/bin/activate"
    conda activate vlm3r 2>/dev/null || true
elif [ -d "$HOME/venvs/vlm3r" ]; then
    source "$HOME/venvs/vlm3r/bin/activate"
fi

PROJECT_DIR="${PROJECT_DIR:-/home/u5fj/trvbale.u5fj/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs

bash scripts/VLM_Dual_Mem/data-download/arkit-scenes/preprocess_arkitscenes.sh

echo ""
echo "End: $(date)"
echo "Job completed"
