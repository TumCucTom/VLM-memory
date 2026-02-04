#!/bin/bash
#SBATCH --job-name=scannet_export_video
#SBATCH --output=logs/scannet_export_video_%j.out
#SBATCH --error=logs/scannet_export_video_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=mlcnu
#SBATCH --account=coms037985

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Change to project directory
cd ~/VLM-memory

# Activate Python virtual environment
source ~/venvs/vlm3r/bin/activate

# Install required dependencies if missing
python3 -c "import cv2" 2>/dev/null || pip install opencv-python
python3 -c "import png" 2>/dev/null || pip install pypng

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directory
mkdir -p data/processed_data/ScanNet/videos

# Run Step 1.2: Export video from .sens files
echo "Starting Step 1.2: Exporting videos from .sens files..."
echo "This may take several hours for all 1201 scenes..."

cd vlm_3r_data_process

# Run as a module to handle relative imports correctly
python -m src.metadata_generation.ScanNet.preprocess.export_video \
    --scans_dir ../data/vlm_3r_data/scannet/scans \
    --output_dir ../data/processed_data/ScanNet/videos \
    --train_val_splits_path splits/scannet \
    --width 640 \
    --height 480 \
    --fps 24 \
    --frame_skip 1 \
    --max_workers 32

cd ..

echo "End Time: $(date)"
echo "Step 1.2 completed"
