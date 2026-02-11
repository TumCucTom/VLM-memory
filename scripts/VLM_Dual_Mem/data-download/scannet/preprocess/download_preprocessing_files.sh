#!/bin/bash
# Download required ScanNet files for preprocessing step 1.1 (point cloud processing)
# This script downloads the PLY mesh files, segments JSON files, and aggregation JSON files
# that are required by the preprocessing script.

# Use python from PATH (should be from activated venv if running from SLURM)
PYTHON_CMD="${PYTHON_CMD:-$(command -v python || command -v python3)}"

SCENES_FILE="data/vlm_3r_data/vsibench/scannet_scenes.txt"
DOWNLOAD_DIR="data/vlm_3r_data/scannet/scans"
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"

# Required file types for preprocessing
REQUIRED_TYPES=(
    "_vh_clean_2.ply"
    "_vh_clean_2.0.010000.segs.json"
    ".aggregation.json"
)

# Check if scenes file exists
if [ ! -f "$SCENES_FILE" ]; then
    echo "Error: Scenes file not found: $SCENES_FILE"
    exit 1
fi

# Check if download script exists
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "Error: ScanNet download script not found: $DOWNLOAD_SCRIPT"
    exit 1
fi

# Check if wrapper script exists
if [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "Error: Wrapper script not found: $WRAPPER_SCRIPT"
    exit 1
fi

# Create directories
mkdir -p "$DOWNLOAD_DIR"

# Count total scenes
TOTAL=$(wc -l < "$SCENES_FILE")
echo "=========================================="
echo "Downloading preprocessing files for $TOTAL ScanNet scenes"
echo "Output directory: $DOWNLOAD_DIR"
echo "Required file types: ${REQUIRED_TYPES[*]}"
echo "=========================================="
echo ""

# Download each file type for each scene
for file_type in "${REQUIRED_TYPES[@]}"; do
    echo "=========================================="
    echo "Downloading file type: $file_type"
    echo "=========================================="
    echo ""
    
    COUNTER=0
    SUCCESS=0
    FAILED=0
    
    while IFS= read -r scene_id; do
        if [ -z "$scene_id" ]; then
            continue
        fi
        
        COUNTER=$((COUNTER + 1))
        echo "[$COUNTER/$TOTAL] Downloading $file_type for $scene_id..."
        
        # Check if file already exists
        scene_file="${DOWNLOAD_DIR}/${scene_id}/${scene_id}${file_type}"
        if [ -f "$scene_file" ]; then
            echo "  ⊙ File already exists, skipping..."
            SUCCESS=$((SUCCESS + 1))
            continue
        fi
        
        # Download the file using wrapper script
        $PYTHON_CMD "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$scene_id" --type "$file_type" --skip_existing
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Successfully downloaded $file_type for $scene_id"
            SUCCESS=$((SUCCESS + 1))
        else
            echo "  ✗ Failed to download $file_type for $scene_id"
            FAILED=$((FAILED + 1))
        fi
        
        # Progress update every 50 scenes
        if [ $((COUNTER % 50)) -eq 0 ]; then
            echo ""
            echo "Progress: $COUNTER/$TOTAL scenes processed (Success: $SUCCESS, Failed: $FAILED)"
            echo ""
        fi
        
        # Add a small delay to avoid overwhelming the server
        sleep 1
    done < "$SCENES_FILE"
    
    echo ""
    echo "Summary for $file_type:"
    echo "  Total scenes: $TOTAL"
    echo "  Successful: $SUCCESS"
    echo "  Failed: $FAILED"
    echo ""
done

echo "=========================================="
echo "Download Summary:"
echo "  All required file types downloaded"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Verify that all required files are present"
echo "2. Run the preprocessing script:"
echo "   sbatch scripts/VLM_Dual_Mem/data-download/scannet/preprocess/step1_1_pointcloud.slurm"
