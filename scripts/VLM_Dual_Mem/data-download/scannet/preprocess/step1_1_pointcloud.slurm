#!/bin/bash
#SBATCH --job-name=scannet_pointcloud
#SBATCH --output=logs/scannet_pointcloud_%j.out
#SBATCH --error=logs/scannet_pointcloud_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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

# Create logs directory if it doesn't exist
mkdir -p logs

# Note: This step requires the ScanNet BenchmarkS
# Check if benchmark scripts exist
if [ ! -d "datasets/ScanNet/BenchmarkScripts/ScanNet200" ]; then
    echo "ERROR: ScanNet BenchmarkScripts not found!"
    echo "Expected location: datasets/ScanNet/BenchmarkScripts/ScanNet200/"
    exit 1
fi

# Check if label map file exists
if [ ! -f "data/raw_data/scannet/scannetv2-labels.combined.tsv" ]; then
    echo "ERROR: Label map file not found!"
    echo "Expected: data/raw_data/scannet/scannetv2-labels.combined.tsv"
    echo "Please download from ScanNet repository"
    exit 1
fi

# Create output directory
mkdir -p data/processed_data/ScanNet/point_cloud

# Run Step 1.1: Process ScanNet200 to get .ply files with labels and instance IDs
echo "Starting Step 1.1: Processing ScanNet200 point clouds..."

# Store project root for absolute paths (needed because BenchmarkScripts is a symlink)
PROJECT_ROOT=$(pwd)
cd datasets/ScanNet/BenchmarkScripts/ScanNet200/

python preprocess_scannet200.py \
    --dataset_root "${PROJECT_ROOT}/data/vlm_3r_data/scannet/scans" \
    --output_root "${PROJECT_ROOT}/data/processed_data/ScanNet/point_cloud" \
    --label_map_file "${PROJECT_ROOT}/data/raw_data/scannet/scannetv2-labels.combined.tsv" \
    --num_workers 8

cd "${PROJECT_ROOT}"

echo "End Time: $(date)"
echo "Step 1.1 completed"
