#!/bin/bash
#SBATCH --job-name=scannetpp_export_video
#SBATCH --output=logs/scannetpp_export_video_%j.out
#SBATCH --error=logs/scannetpp_export_video_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
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

# Create output directory
mkdir -p data/processed_data/ScanNetpp/videos

# Change to the project root directory
cd /home/u5fj/trvbale.u5fj/scratch/VLM-memory

# Run Step 1.1: Export video from DSLR images
echo "Starting Step 1.1: Exporting videos from DSLR images..."
echo "This will create .mp4 videos from DSLR resized undistorted images..."

cd vlm_3r_data_process

# Process training split
echo "Processing training split..."
python -m src.metadata_generation.ScanNetpp.preprocess.export_video \
    --raw_data_dir ../data/raw_data/scannetpp/data \
    --rendered_data_dir ../data/raw_data/scannetpp/data \
    --output_dir ../data/processed_data/ScanNetpp/videos \
    --split_path ../vlm_3r_data_process/splits/scannetpp/nvs_sem_train.txt \
    --height 389 \
    --width 584 \
    --fps 3 \
    --num_workers 32

# Process validation split
echo "Processing validation split..."
python -m src.metadata_generation.ScanNetpp.preprocess.export_video \
    --raw_data_dir ../data/raw_data/scannetpp/data \
    --rendered_data_dir ../data/raw_data/scannetpp/data \
    --output_dir ../data/processed_data/ScanNetpp/videos \
    --split_path ../vlm_3r_data_process/splits/scannetpp/nvs_sem_val.txt \
    --height 389 \
    --width 584 \
    --fps 3 \
    --num_workers 32

cd ..

echo "End Time: $(date)"
echo "Step 1.1 completed"
echo ""
echo "Videos saved to: data/processed_data/ScanNetpp/videos/"
