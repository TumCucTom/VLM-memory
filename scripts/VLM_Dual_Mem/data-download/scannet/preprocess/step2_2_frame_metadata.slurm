#!/bin/bash
#SBATCH --job-name=scannet_frame_metadata
#SBATCH --output=logs/scannet_frame_metadata_%j.out
#SBATCH --error=logs/scannet_frame_metadata_%j.err
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

# Run Step 2.2: Get frame metadata for train and val splits
echo "Starting Step 2.2: Generating frame metadata..."

cd vlm_3r_data_process

# Process training split
echo "Processing training split..."
python -m src.metadata_generation.ScanNet.scannet_frame_metadata \
    --processed_dir "../data/processed_data/ScanNet" \
    --scene_list_file "splits/scannet/scannetv2_train.txt" \
    --save_dir "../data/processed_data/ScanNet/metadata/train" \
    --output_filename "scannet_frame_metadata_train.json" \
    --num_workers 64 \
    --overwrite \
    --random_seed 42

# Process validation split
echo "Processing validation split..."
python -m src.metadata_generation.ScanNet.scannet_frame_metadata \
    --processed_dir "../data/processed_data/ScanNet" \
    --scene_list_file "splits/scannet/scannetv2_val.txt" \
    --save_dir "../data/processed_data/ScanNet/metadata/val" \
    --output_filename "scannet_frame_metadata_val.json" \
    --num_workers 64 \
    --overwrite \
    --random_seed 42

cd ..

echo "End Time: $(date)"
echo "Step 2.2 completed"
