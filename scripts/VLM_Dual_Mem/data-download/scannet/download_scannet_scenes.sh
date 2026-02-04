#!/bin/bash
# Download specific ScanNet scenes required for training

# Use python from PATH (should be from activated venv if running from SLURM)
# If running directly, activate venv first: source ~/venvs/vlm3r/bin/activate
# Prefer 'python' (from venv) over 'python3' (system)
PYTHON_CMD="${PYTHON_CMD:-$(command -v python || command -v python3)}"

SCENES_FILE="data/vlm_3r_data/vsibench/scannet_scenes.txt"
DOWNLOAD_DIR="data/vlm_3r_data/scannet/scans"
VIDEO_OUTPUT_DIR="data/vlm_3r_data/scannet/videos"
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"

# Check if scenes file exists
if [ ! -f "$SCENES_FILE" ]; then
    echo "Error: Scenes file not found: $SCENES_FILE"
    exit 1
fi

# Check if download script exists
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "Error: ScanNet download script not found: $DOWNLOAD_SCRIPT"
    echo "Please ensure the download-scannet.py script is in scripts/VLM_Dual_Mem/data-download/scannet/"
    exit 1
fi

# Check if wrapper script exists
if [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "Error: Wrapper script not found: $WRAPPER_SCRIPT"
    exit 1
fi

# Create directories
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$VIDEO_OUTPUT_DIR"

# Count total scenes
TOTAL=$(wc -l < "$SCENES_FILE")
echo "=========================================="
echo "Downloading $TOTAL ScanNet scenes"
echo "Output directory: $DOWNLOAD_DIR"
echo "Video output directory: $VIDEO_OUTPUT_DIR"
echo "=========================================="
echo ""

# Download each scene
COUNTER=0
SUCCESS=0
FAILED=0

while IFS= read -r scene_id; do
    if [ -z "$scene_id" ]; then
        continue
    fi
    
    COUNTER=$((COUNTER + 1))
    echo "[$COUNTER/$TOTAL] Downloading $scene_id..."
    
    # Download the scene (.sens files contain the video data)
    # Use wrapper script to handle interactive prompts non-interactively
    # Use python from activated environment (or system python3)
    $PYTHON_CMD "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$scene_id" --skip_existing
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully downloaded $scene_id"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ Failed to download $scene_id"
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
echo "=========================================="
echo "Download Summary:"
echo "  Total scenes: $TOTAL"
echo "  Successful: $SUCCESS"
echo "  Failed: $FAILED"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Extract videos from .sens files using:"
echo "   python vlm_3r_data_process/src/metadata_generation/ScanNet/preprocess/export_video.py \\"
echo "       --scans_dir $DOWNLOAD_DIR \\"
echo "       --output_dir $VIDEO_OUTPUT_DIR \\"
echo "       --scene_list_file $SCENES_FILE \\"
echo "       --width 640 --height 480 --fps 24"
