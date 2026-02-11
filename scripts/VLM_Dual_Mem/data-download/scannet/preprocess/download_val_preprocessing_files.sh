#!/bin/bash
# Download preprocessing files for ScanNet VAL split only (for step 1.1 point cloud).
# Downloads: _vh_clean_2.ply, _vh_clean_2.0.010000.segs.json, .aggregation.json

PYTHON_CMD="${PYTHON_CMD:-$(command -v python || command -v python3)}"

SCENES_FILE="vlm_3r_data_process/splits/scannet/scannetv2_val.txt"
DOWNLOAD_DIR="data/vlm_3r_data/scannet/scans"
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"

REQUIRED_TYPES=(
    "_vh_clean_2.ply"
    "_vh_clean_2.0.010000.segs.json"
    ".aggregation.json"
)

if [ ! -f "$SCENES_FILE" ]; then
    echo "Error: Val scene list not found: $SCENES_FILE"
    exit 1
fi
if [ ! -f "$DOWNLOAD_SCRIPT" ] || [ ! -f "$WRAPPER_SCRIPT" ]; then
    echo "Error: Download or wrapper script not found"
    exit 1
fi

mkdir -p "$DOWNLOAD_DIR"

TOTAL=$(wc -l < "$SCENES_FILE")
echo "=========================================="
echo "Downloading preprocessing files for $TOTAL ScanNet VAL scenes"
echo "Output directory: $DOWNLOAD_DIR"
echo "Required file types: ${REQUIRED_TYPES[*]}"
echo "=========================================="
echo ""

for file_type in "${REQUIRED_TYPES[@]}"; do
    echo "=========================================="
    echo "Downloading file type: $file_type"
    echo "=========================================="
    echo ""

    COUNTER=0
    SUCCESS=0
    FAILED=0

    while IFS= read -r scene_id; do
        scene_id=$(echo "$scene_id" | tr -d '\r\n' | xargs)
        [ -z "$scene_id" ] && continue

        COUNTER=$((COUNTER + 1))
        scene_file="${DOWNLOAD_DIR}/${scene_id}/${scene_id}${file_type}"
        nested_file="${DOWNLOAD_DIR}/scans/${scene_id}/${scene_id}${file_type}"

        if [ -f "$scene_file" ]; then
            echo "  [$COUNTER/$TOTAL] $scene_id: exists, skipping"
            SUCCESS=$((SUCCESS + 1))
            continue
        fi
        if [ -f "$nested_file" ]; then
            mkdir -p "${DOWNLOAD_DIR}/${scene_id}"
            mv "$nested_file" "$scene_file"
            echo "  [$COUNTER/$TOTAL] $scene_id: moved from nested, ok"
            SUCCESS=$((SUCCESS + 1))
            continue
        fi

        echo "  [$COUNTER/$TOTAL] Downloading $file_type for $scene_id..."
        $PYTHON_CMD "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$scene_id" --type "$file_type" --skip_existing

        if [ $? -eq 0 ]; then
            if [ -f "$nested_file" ]; then
                mkdir -p "${DOWNLOAD_DIR}/${scene_id}"
                mv "$nested_file" "$scene_file"
            fi
            if [ -f "$scene_file" ]; then
                echo "    ✓ Success"
                SUCCESS=$((SUCCESS + 1))
            else
                echo "    ✗ File not found after download"
                FAILED=$((FAILED + 1))
            fi
        else
            echo "    ✗ Failed"
            FAILED=$((FAILED + 1))
        fi

        [ $((COUNTER % 50)) -eq 0 ] && echo "  Progress: $COUNTER/$TOTAL (Success: $SUCCESS, Failed: $FAILED)"
        sleep 1
    done < "$SCENES_FILE"

    echo ""
    echo "Summary for $file_type: Total=$TOTAL Success=$SUCCESS Failed=$FAILED"
    echo ""
done

echo "=========================================="
echo "Val preprocessing files download complete"
echo "=========================================="
