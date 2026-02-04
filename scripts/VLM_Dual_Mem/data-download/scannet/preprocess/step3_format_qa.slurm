#!/bin/bash
#SBATCH --job-name=scannet_format_qa
#SBATCH --output=logs/scannet_format_qa_%j.out
#SBATCH --error=logs/scannet_format_qa_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
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

# Run Step 3 (format): Format QA data
echo "Starting Step 3 (format): Formatting QA data..."
echo "This formats the generated QA data into the required structure..."

cd vlm_3r_data_process

# Format QA data
# Note: Update --tag to match the tag used in step3_generate_qa.slurm
# The tag should match the date when step3_generate_qa was run
TAG=$(date +%m_%d_%Y)
INPUT_DIR="../data/scannet_vsibench/${TAG}"

# Check if input directory exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: Input directory not found: ${INPUT_DIR}"
    echo "Please ensure step3_generate_qa.slurm has completed successfully"
    exit 1
fi

python -m src.format_qa \
    --input_dir ${INPUT_DIR} \
    --output_dir ../data/scannet_vsibench \
    --tag ${TAG}

cd ..

echo "End Time: $(date)"
echo "Step 3 (format) completed"
