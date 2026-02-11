#!/bin/bash
#SBATCH --job-name=download_qa_hf
#SBATCH --output=logs/download_qa_hf_%j.out
#SBATCH --error=logs/download_qa_hf_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
mkdir -p data/scannet_vsibench

# Install huggingface_hub if not already installed
pip install -q huggingface_hub datasets || echo "Warning: Failed to install huggingface_hub"

# Download QA pairs from Hugging Face
echo "Downloading QA pairs from Hugging Face..."
echo "Dataset: Journey9ni/VLM-3R-DATA"
echo "This includes training data for VSiBench and VSTiBench"

cd vlm_3r_data_process

# Use Python to download from Hugging Face
python << 'EOF'
from huggingface_hub import snapshot_download
import os
import shutil

# Hugging Face dataset repository
repo_id = "Journey9ni/VLM-3R-DATA"
output_dir = "../data/scannet_vsibench"

print(f"Downloading dataset from {repo_id}...")
print(f"Output directory: {os.path.abspath(output_dir)}")

try:
    # Download the entire dataset
    # Note: This will download all files. You may want to filter for ScanNet-specific files
    local_dir = snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        repo_type="dataset",
        local_dir_use_symlinks=False
    )
    
    print(f"\n✓ Successfully downloaded dataset to: {local_dir}")
    print("\nNote: The dataset includes QA pairs for multiple datasets (ScanNet, ScanNet++, ARKitScenes).")
    print("You may need to filter for ScanNet-specific files based on your needs.")
    
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("\nAlternative: You can use the datasets library to load specific splits:")
    print("  from datasets import load_dataset")
    print("  dataset = load_dataset('Journey9ni/VLM-3R-DATA')")
    exit(1)
EOF

cd ..

echo ""
echo "End Time: $(date)"
echo "Download completed"
echo ""
echo "Next steps:"
echo "1. Check the downloaded data in: data/scannet_vsibench/"
echo "2. Filter for ScanNet-specific QA pairs if needed"
echo "3. The data should be ready to use for training"
