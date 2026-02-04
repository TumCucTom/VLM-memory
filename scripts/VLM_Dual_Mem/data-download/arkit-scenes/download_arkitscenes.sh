#!/bin/bash

# Download ARKitScenes data for VLM-3R training
# Based on ARKitScenes DATA.md and VLM-memory requirements
# Downloads to scratch space to avoid home directory quota issues

set -e  # Exit on error

# Configuration
ARKITSCENES_REPO_DIR="/user/home/hf23482/ARKitScenes"
# Use work space for large data downloads (as per VLM-memory README)
WORK_DIR="/user/work/$USER"
SCRATCH_DIR="${WORK_DIR}/ARKitScenes"
DOWNLOAD_DIR="${SCRATCH_DIR}/raw"
CSV_FILE="${ARKITSCENES_REPO_DIR}/raw/raw_train_val_splits.csv"

# Assets needed for VLM-3R processing (based on vlm_3r_data_process requirements):
# - mov: Video files
# - annotation: Object annotations (3D bounding boxes)
# - mesh: 3D mesh for point cloud extraction
# - lowres_wide: RGB images
# - lowres_depth: Depth maps
# - lowres_wide.traj: Camera poses
# - lowres_wide_intrinsics: Camera intrinsics
# - confidence: Depth confidence maps (optional but useful)
# - highres_depth: High resolution depth (only for upsampling subset, optional)
RAW_DATASET_ASSETS=(
    "mov"
    "annotation"
    "mesh"
    "confidence"
    "lowres_depth"
    "lowres_wide.traj"
    "lowres_wide"
    "lowres_wide_intrinsics"
    "ultrawide"
    "ultrawide_intrinsics"
    "vga_wide"
    "vga_wide_intrinsics"
    "highres_depth"  # Only available for upsampling subset, will be skipped if not available
)

echo "=========================================="
echo "ARKitScenes Data Download Script"
echo "=========================================="
echo "Source CSV: ${CSV_FILE}"
echo "Download directory: ${DOWNLOAD_DIR}"
echo "Assets to download: ${RAW_DATASET_ASSETS[*]}"
echo ""

# Check if ARKitScenes repo exists
if [ ! -d "${ARKITSCENES_REPO_DIR}" ]; then
    echo "ERROR: ARKitScenes repository not found at ${ARKITSCENES_REPO_DIR}"
    echo "Please clone it first: git clone https://github.com/apple/ARKitScenes.git"
    exit 1
fi

# Check if CSV file exists
if [ ! -f "${CSV_FILE}" ]; then
    echo "ERROR: CSV file not found at ${CSV_FILE}"
    exit 1
fi

# Create download directory
mkdir -p "${DOWNLOAD_DIR}"
echo "Created download directory: ${DOWNLOAD_DIR}"
echo ""

# Change to ARKitScenes directory
cd "${ARKITSCENES_REPO_DIR}"

# Check if download_data.py exists
if [ ! -f "download_data.py" ]; then
    echo "ERROR: download_data.py not found in ${ARKITSCENES_REPO_DIR}"
    exit 1
fi

# Convert assets array to space-separated string for the script
ASSETS_STRING=$(IFS=' '; echo "${RAW_DATASET_ASSETS[*]}")

echo "Starting download..."
echo "This may take a long time depending on your connection speed."
echo "The dataset is large (several hundred GB)."
echo ""

# Run the download script
# Using CSV file to download all videos in the dataset
python3 download_data.py raw \
    --video_id_csv "${CSV_FILE}" \
    --download_dir "${DOWNLOAD_DIR}" \
    --raw_dataset_assets ${ASSETS_STRING}

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo "Data location: ${DOWNLOAD_DIR}"
echo ""
echo "Next steps:"
echo "1. The data will be organized in Training/ and Validation/ subdirectories"
echo "2. For VLM-3R processing, you may need to:"
echo "   - Convert .mov videos to .mp4 (see vlm_3r_data_process/src/metadata_generation/ARkitScenes/copy_video.py)"
echo "   - Process the data using vlm_3r_data_process scripts"
echo ""

