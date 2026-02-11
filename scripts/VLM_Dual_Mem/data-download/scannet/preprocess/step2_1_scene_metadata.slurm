#!/bin/bash
#SBATCH --job-name=scannet_scene_metadata
#SBATCH --output=logs/scannet_scene_metadata_%j.out
#SBATCH --error=logs/scannet_scene_metadata_%j.err
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

# Activate Python virtual environment
source ~/miniforge3/bin/activate
conda activate vlm3r

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directories
mkdir -p data/processed_data/ScanNet/metadata/train
mkdir -p data/processed_data/ScanNet/metadata/val

# Check if label map file exists
if [ ! -f "data/raw_data/scannet/scannetv2-labels.combined.tsv" ]; then
    echo "ERROR: Label map file not found!"
    echo "Expected: data/raw_data/scannet/scannetv2-labels.combined.tsv"
    echo "Please download from ScanNet repository"
    exit 1
fi

# Run Step 2.1: Get scene metadata for train and val splits
echo "Starting Step 2.1: Generating scene metadata..."

cd vlm_3r_data_process

# Process training split
echo "Processing training split..."
python -m src.metadata_generation.ScanNet.scannet_metadata \
    --input_dir ../data/processed_data/ScanNet/point_cloud/train \
    --scene_list_file splits/scannet/scannetv2_train.txt \
    --save_dir ../data/processed_data/ScanNet/metadata/train \
    --output_filename scannet_metadata_train.json \
    --label_mapping_path ../data/raw_data/scannet/scannetv2-labels.combined.tsv \
    --video_dir ScanNet/videos/train \
    --num_workers 64 \
    --overwrite

# Process validation split
echo "Processing validation split..."
python -m src.metadata_generation.ScanNet.scannet_metadata \
    --input_dir ../data/processed_data/ScanNet/point_cloud/val \
    --scene_list_file splits/scannet/scannetv2_val.txt \
    --save_dir ../data/processed_data/ScanNet/metadata/val \
    --output_filename scannet_metadata_val.json \
    --label_mapping_path ../data/raw_data/scannet/scannetv2-labels.combined.tsv \
    --video_dir ScanNet/videos/val \
    --num_workers 64 \
    --overwrite

cd ..

echo "End Time: $(date)"
echo "Step 2.1 completed"
