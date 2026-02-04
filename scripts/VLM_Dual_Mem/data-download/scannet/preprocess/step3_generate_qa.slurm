#!/bin/bash
#SBATCH --job-name=scannet_generate_qa
#SBATCH --output=logs/scannet_generate_qa_%j.out
#SBATCH --error=logs/scannet_generate_qa_%j.err
#SBATCH --time=48:00:00
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

# Create logs directory if it doesn't exist
mkdir -p logs

# Create output directory
mkdir -p data/scannet_vsibench

# Run Step 3: Generate QA pairs
echo "Starting Step 3: Generating QA pairs..."
echo "This will generate question-answer pairs based on the processed data..."

cd vlm_3r_data_process

# Generate QA for training split
# Note: Adjust --num_subsample based on your needs (10000 is a subset)
# The script will create a subdirectory based on split_type (train/val)
TAG=$(date +%m_%d_%Y)
OUTPUT_DIR="../data/scannet_vsibench/${TAG}"

python -m src.tasks.all_generate \
    --split_path splits/scannet/scannetv2_train.txt \
    --split_type train \
    --processed_data_path ../data/processed_data/ScanNet \
    --output_dir ${OUTPUT_DIR} \
    --dataset scannet \
    --num_subsample 10000 \
    --num_workers 32

cd ..

echo "End Time: $(date)"
echo "Step 3 completed"
