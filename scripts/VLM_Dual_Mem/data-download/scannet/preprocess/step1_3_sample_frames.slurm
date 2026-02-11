#!/bin/bash
#SBATCH --job-name=scannet_sample_frames
#SBATCH --output=logs/scannet_sample_frames_%j.out
#SBATCH --error=logs/scannet_sample_frames_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
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

# Create output directory
mkdir -p data/processed_data/ScanNet

# Run Step 1.3: Sample frame data and camera intrinsics
echo "Starting Step 1.3: Sampling frames and extracting camera intrinsics..."
echo "This will sample 32 frames per scene with camera poses and intrinsics..."

cd vlm_3r_data_process

python -m src.metadata_generation.ScanNet.preprocess.export_sampled_frames \
    --scans_dir ../data/vlm_3r_data/scannet/scans \
    --output_dir ../data/processed_data/ScanNet \
    --train_val_splits_path splits/scannet \
    --num_frames 32 \
    --max_workers 64 \
    --image_size 480 640

cd ..

echo "End Time: $(date)"
echo "Step 1.3 completed"
echo ""
echo "NOTE: After this step completes, you can safely delete .sens files to save ~754 GB"
echo "      All data has been extracted to videos, point clouds, and sampled frames."
