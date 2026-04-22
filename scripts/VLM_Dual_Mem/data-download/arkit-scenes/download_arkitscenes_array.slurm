#!/bin/bash
#SBATCH --job-name=arkitscenes_download
#SBATCH --output=logs/download_arkitscenes_%A_%a.out
#SBATCH --error=logs/download_arkitscenes_%A_%a.err
#SBATCH --time=7:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=workq
#SBATCH --array=0-27

# Download ARKitScenes data using metadata chunks in parallel
# Each array task processes one chunk (metadata_chunk_${SLURM_ARRAY_TASK_ID}.csv)

echo "=========================================="
echo "ARKitScenes Download Job (Array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID})"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo ""

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate vlm3r

# Verify Python and pandas
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

PROJECT_DIR="${PROJECT_DIR:-/home/b6ar/trvbale.b6ar/scratch/VLM-memory}"
cd "$PROJECT_DIR"

# Metadata chunks directory
METADATA_CHUNKS_DIR="${PROJECT_DIR}/data/metadata_chunks"
mkdir -p logs

# Get the chunk file for this array task
CHUNK_FILE="${METADATA_CHUNKS_DIR}/metadata_chunk_${SLURM_ARRAY_TASK_ID}.csv"

if [ ! -f "$CHUNK_FILE" ]; then
    echo "ERROR: Chunk file not found: $CHUNK_FILE"
    exit 1
fi

echo "Processing chunk: $CHUNK_FILE"
echo "Number of entries: $(wc -l < "$CHUNK_FILE")"

# Output directory for this chunk's downloads
CHUNK_OUTPUT_DIR="${PROJECT_DIR}/data/raw_arkitscenes/raw"
mkdir -p "$CHUNK_OUTPUT_DIR"

# Clone ARKitScenes repo if needed
REPO_DIR="${ARKITSCENES_REPO_DIR:-${PROJECT_DIR}/data/raw_arkitscenes/ARKitScenes_repo}"
if [ ! -f "${REPO_DIR}/download_data.py" ]; then
    echo "Cloning ARKitScenes repository..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone --depth 1 https://github.com/apple/ARKitScenes.git "$REPO_DIR"
fi

# Assets needed for VLM-3R
RAW_DATASET_ASSETS=(
    mov
    annotation
    mesh
    confidence
    lowres_depth
    "lowres_wide.traj"
    lowres_wide
    lowres_wide_intrinsics
)

# Run Apple's download script for this chunk
DOWNLOAD_PARENT="$(realpath "$(dirname "$CHUNK_OUTPUT_DIR")")"
ASSETS_STRING="${RAW_DATASET_ASSETS[*]}"

echo "Starting download for chunk ${SLURM_ARRAY_TASK_ID}..."
python "${REPO_DIR}/download_data.py" raw \
    --video_id_csv "$CHUNK_FILE" \
    --download_dir "$DOWNLOAD_PARENT" \
    --raw_dataset_assets $ASSETS_STRING

echo ""
echo "=========================================="
echo "Chunk ${SLURM_ARRAY_TASK_ID} download completed"
echo "=========================================="
echo "End Time: $(date)"
