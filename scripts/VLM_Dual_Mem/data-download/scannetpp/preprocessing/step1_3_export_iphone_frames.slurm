#!/bin/bash
#SBATCH --job-name=scannetpp_iphone_frames
#SBATCH --output=logs/scannetpp_iphone_frames_%j.out
#SBATCH --error=logs/scannetpp_iphone_frames_%j.err
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
echo "=========================================="

# Activate Python virtual environment
source ~/miniforge3/bin/activate
conda activate vlm3r

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directory
mkdir -p data/processed_data/ScanNetpp

# Change to the project root directory
cd /home/u5fj/trvbale.u5fj/scratch/VLM-memory

# Run Step 1.3: Export iPhone frames (optional - only if using iPhone data)
echo "Starting Step 1.3: Extracting iPhone frames..."
echo "This will extract RGB and depth frames from iPhone video/depth data using COLMAP poses..."
echo "NOTE: This step is optional and only needed if you want to use iPhone data instead of DSLR data."

cd vlm_3r_data_process

# Process training split
echo "Processing training split..."
python -m src.metadata_generation.ScanNetpp.preprocess.export_iphone_frames \
    --data_root ../data/raw_data/scannetpp/data \
    --scene_list_file splits/scannetpp/nvs_sem_train.txt \
    --output_dir ../data/processed_data/ScanNetpp \
    --device iphone \
    --num_frames 32 \
    --image_size 480 640 \
    --num_processes 64

# Process validation split
echo "Processing validation split..."
python -m src.metadata_generation.ScanNetpp.preprocess.export_iphone_frames \
    --data_root ../data/raw_data/scannetpp/data \
    --scene_list_file splits/scannetpp/nvs_sem_val.txt \
    --output_dir ../data/processed_data/ScanNetpp \
    --device iphone \
    --num_frames 32 \
    --image_size 480 640 \
    --num_processes 64

cd ..

echo "End Time: $(date)"
echo "Step 1.3 completed"
echo ""
echo "iPhone frames saved to: data/processed_data/ScanNetpp/"
echo "  - color/train/ and color/val/ (iPhone RGB)"
echo "  - depth/train/ and depth/val/ (iPhone depth)"
echo "  - pose/train/ and pose/val/ (COLMAP poses)"
echo "  - intrinsic/train/ and intrinsic/val/ (COLMAP intrinsics)"
