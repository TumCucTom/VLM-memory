#!/bin/bash
#SBATCH --job-name=scannetpp_scene_metadata
#SBATCH --output=logs/scannetpp_scene_metadata_%j.out
#SBATCH --error=logs/scannetpp_scene_metadata_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --partition=workq

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Activate Python virtual environment
source ~/miniforge3/bin/activate
conda activate vlm3r

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directories
mkdir -p data/processed_data/ScanNetpp/metadata/train
mkdir -p data/processed_data/ScanNetpp/metadata/val

# Change to the project root directory
cd /home/u5fj/trvbale.u5fj/scratch/VLM-memory

# Run Step 2.1: Generate scene metadata for train and val splits
echo "Starting Step 2.1: Generating scene metadata..."
echo "This processes point clouds to extract scene-wide information..."

cd vlm_3r_data_process

# Process training split
echo "Processing training split..."
python -m src.metadata_generation.ScanNetpp.scannetpp_metadata \
    --input_dir ../data/raw_data/scannetpp/data \
    --scene_list_file splits/scannetpp/nvs_sem_train.txt \
    --save_dir ../data/processed_data/ScanNetpp/metadata/train \
    --output_filename scannetpp_metadata_train.json \
    --split train \
    --dataset_name ScanNetpp \
    --num_workers 64 \
    --overwrite

# Process validation split
echo "Processing validation split..."
python -m src.metadata_generation.ScanNetpp.scannetpp_metadata \
    --input_dir ../data/raw_data/scannetpp/data \
    --scene_list_file splits/scannetpp/nvs_sem_val.txt \
    --save_dir ../data/processed_data/ScanNetpp/metadata/val \
    --output_filename scannetpp_metadata_val.json \
    --split val \
    --dataset_name ScanNetpp \
    --num_workers 64 \
    --overwrite

cd ..

echo "End Time: $(date)"
echo "Step 2.1 completed"
echo ""
echo "Scene metadata saved to:"
echo "  - data/processed_data/ScanNetpp/metadata/train/scannetpp_metadata_train.json"
echo "  - data/processed_data/ScanNetpp/metadata/val/scannetpp_metadata_val.json"
