#!/bin/bash
#SBATCH --job-name=scannetpp_frame_metadata
#SBATCH --output=logs/scannetpp_frame_metadata_%j.out
#SBATCH --error=logs/scannetpp_frame_metadata_%j.err
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

# Run Step 2.2: Generate frame metadata for train and val splits
echo "Starting Step 2.2: Generating frame metadata..."
echo "This processes sampled frames to extract frame-specific information..."

cd vlm_3r_data_process

# Process training split
echo "Processing training split..."
python -m src.metadata_generation.ScanNetpp.scannetpp_frame_metadata \
    --scene_list_file splits/scannetpp/nvs_sem_train.txt \
    --rendered_data_dir ../data/raw_data/scannetpp/data \
    --raw_data_dir ../data/raw_data/scannetpp/data \
    --save_dir ../data/processed_data/ScanNetpp/metadata/train \
    --output_filename scannetpp_frame_metadata_train.json \
    --split train \
    --max_frames 32 \
    --num_workers 64 \
    --overwrite

# Process validation split
echo "Processing validation split..."
python -m src.metadata_generation.ScanNetpp.scannetpp_frame_metadata \
    --scene_list_file splits/scannetpp/nvs_sem_val.txt \
    --rendered_data_dir ../data/raw_data/scannetpp/data \
    --raw_data_dir ../data/raw_data/scannetpp/data \
    --save_dir ../data/processed_data/ScanNetpp/metadata/val \
    --output_filename scannetpp_frame_metadata_val.json \
    --split val \
    --max_frames 32 \
    --num_workers 64 \
    --overwrite

cd ..

echo "End Time: $(date)"
echo "Step 2.2 completed"
echo ""
echo "Frame metadata saved to:"
echo "  - data/processed_data/ScanNetpp/metadata/train/scannetpp_frame_metadata_train.json"
echo "  - data/processed_data/ScanNetpp/metadata/val/scannetpp_frame_metadata_val.json"
