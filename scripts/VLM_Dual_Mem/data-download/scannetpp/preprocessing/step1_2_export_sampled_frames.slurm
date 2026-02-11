#!/bin/bash
#SBATCH --job-name=scannetpp_sample_frames
#SBATCH --output=logs/scannetpp_sample_frames_%j.out
#SBATCH --error=logs/scannetpp_sample_frames_%j.err
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

# Run Step 1.2: Sample frame data and camera intrinsics
echo "Starting Step 1.2: Sampling frames and extracting camera intrinsics..."
echo "This will sample 32 frames per scene with camera poses and intrinsics..."

cd vlm_3r_data_process

# Process training split
echo "Processing training split..."
python -m src.metadata_generation.ScanNetpp.preprocess.export_sampled_frames \
    --split_path splits/scannetpp/nvs_sem_train.txt \
    --raw_data_dir ../data/raw_data/scannetpp/data \
    --rendered_data_dir ../data/raw_data/scannetpp/data \
    --output_dir ../data/processed_data/ScanNetpp \
    --num_frames_to_sample 32 \
    --split_name train

# Process validation split
echo "Processing validation split..."
python -m src.metadata_generation.ScanNetpp.preprocess.export_sampled_frames \
    --split_path splits/scannetpp/nvs_sem_val.txt \
    --raw_data_dir ../data/raw_data/scannetpp/data \
    --rendered_data_dir ../data/raw_data/scannetpp/data \
    --output_dir ../data/processed_data/ScanNetpp \
    --num_frames_to_sample 32 \
    --split_name val

cd ..

echo "End Time: $(date)"
echo "Step 1.2 completed"
echo ""
echo "Sampled frames saved to: data/processed_data/ScanNetpp/"
echo "  - color/train/ and color/val/"
echo "  - depth/train/ and depth/val/"
echo "  - pose/train/ and pose/val/"
echo "  - intrinsic/train/ and intrinsic/val/"
