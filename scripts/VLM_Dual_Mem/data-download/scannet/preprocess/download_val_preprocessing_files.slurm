#!/bin/bash
#SBATCH --job-name=scannet_download_val_preproc
#SBATCH --output=logs/scannet_download_val_preproc_%j.out
#SBATCH --error=logs/scannet_download_val_preproc_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

source ~/miniforge3/bin/activate
conda activate vlm3r

mkdir -p logs

echo "Downloading preprocessing files (_vh_clean_2.ply, .segs.json, .aggregation.json) for ScanNet VAL scenes..."
bash scripts/VLM_Dual_Mem/data-download/scannet/preprocess/download_val_preprocessing_files.sh

echo "End Time: $(date)"
echo "Job completed"
