#!/bin/bash
# Download .sens files for ScanNet validation split only.
# Scene list: vlm_3r_data_process/splits/scannet/scannetv2_val.txt

PYTHON_CMD="${PYTHON_CMD:-$(command -v python || command -v python3)}"

# Val split scene list (relative to project root)
SCENES_FILE="vlm_3r_data_process/splits/scannet/scannetv2_val.txt"
DOWNLOAD_DIR="data/vlm_3r_data/scannet/scans"
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"

if [ ! -f "$SCENES_FILE" ]; then
    echo "Error: Val scene list not found: $SCENES_FILE"
    exit 1
fi

if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "Error: Download script not found: $DOWNLOAD_SCRIPT"
    exit 1
fi

if [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "Error: Wrapper script not found: $WRAPPER_SCRIPT"
    exit 1
fi

mkdir -p "$DOWNLOAD_DIR"

TOTAL=$(wc -l < "$SCENES_FILE")
echo "=========================================="
echo "Downloading .sens for $TOTAL ScanNet VAL scenes"
echo "Output directory: $DOWNLOAD_DIR"
echo "=========================================="
echo ""

COUNTER=0
SUCCESS=0
FAILED=0

while IFS= read -r scene_id; do
    scene_id=$(echo "$scene_id" | tr -d '\r\n' | xargs)
    [ -z "$scene_id" ] && continue

    COUNTER=$((COUNTER + 1))
    echo "[$COUNTER/$TOTAL] Downloading .sens for $scene_id..."

    $PYTHON_CMD "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$scene_id" --type .sens --skip_existing

    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully downloaded $scene_id"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ Failed to download $scene_id"
        FAILED=$((FAILED + 1))
    fi

    if [ $((COUNTER % 50)) -eq 0 ]; then
        echo ""
        echo "Progress: $COUNTER/$TOTAL (Success: $SUCCESS, Failed: $FAILED)"
        echo ""
    fi

    sleep 1
done < "$SCENES_FILE"

echo ""
echo "=========================================="
echo "Download Summary (.sens for val):"
echo "  Total: $TOTAL  Success: $SUCCESS  Failed: $FAILED"
echo "=========================================="
