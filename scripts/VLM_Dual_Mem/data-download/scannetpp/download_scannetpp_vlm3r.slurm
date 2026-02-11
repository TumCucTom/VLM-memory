#!/bin/bash
#SBATCH --job-name=download_scannetpp_vlm3r
#SBATCH --output=logs/download_scannetpp_vlm3r_%j.out
#SBATCH --error=logs/download_scannetpp_vlm3r_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# Change to the project root directory
cd /home/u5fj/trvbale.u5fj/scratch/VLM-memory

# Install required packages if not already installed
echo "Checking dependencies..."
pip install -q munch tqdm pyyaml || echo "Warning: Some dependencies may not be installed"

# Set the config file path
CONFIG_FILE="scripts/VLM_Dual_Mem/data-download/scannetpp/download_scannetpp_vlm3r.yml"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannetpp/download_scannetpp_wrapper.py"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if wrapper script exists
if [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "Error: Wrapper script not found: $WRAPPER_SCRIPT"
    exit 1
fi

# Create data directory if it doesn't exist
DATA_DIR="data/raw_data/scannetpp"
mkdir -p "$DATA_DIR"

echo "=========================================="
echo "Downloading ScanNet++ data for VLM_3R"
echo "Config file: $CONFIG_FILE"
echo "Data directory: $DATA_DIR"
echo "=========================================="
echo ""
echo "Starting download..."
echo "Note: This will download training and validation splits (nvs_sem_train, nvs_sem_val)"
echo "The download may require significant disk space (~1.5TB for full dataset)"
echo ""

# Run the wrapper script which handles non-interactive execution
python "$WRAPPER_SCRIPT" "$CONFIG_FILE"

DOWNLOAD_EXIT_CODE=$?

if [ $DOWNLOAD_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Download completed successfully!"
    echo "Data location: $DATA_DIR"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Download failed or was interrupted"
    echo "Exit code: $DOWNLOAD_EXIT_CODE"
    echo "Check the error log for details"
    echo "=========================================="
fi

echo "End Time: $(date)"
echo "Job completed"
