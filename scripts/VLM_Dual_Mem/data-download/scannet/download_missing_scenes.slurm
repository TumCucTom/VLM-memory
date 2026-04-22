#!/bin/bash
#SBATCH --job-name=download_missing_scannet
#SBATCH --output=logs/download_missing_scannet_%j.out
#SBATCH --error=logs/download_missing_scannet_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

source ~/miniforge3/bin/activate
conda activate vlm3r

PYTHON_CMD="${PYTHON_CMD:-$(command -v python || command -v python3)}"

MISSING_FILE="scripts/VLM_Dual_Mem/data-download/scannet/missing_scenes.txt"
DOWNLOAD_DIR="data/vlm_3r_data/scannet/scans"
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"

TOTAL=$(wc -l < "$MISSING_FILE")
echo "Downloading $TOTAL missing ScanNet scenes"
echo ""

COUNTER=0
SUCCESS=0
FAILED=0

while IFS= read -r scene_id; do
    if [ -z "$scene_id" ]; then
        continue
    fi

    COUNTER=$((COUNTER + 1))
    echo "[$COUNTER/$TOTAL] Downloading $scene_id..."

    $PYTHON_CMD "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$scene_id"

    if [ $? -eq 0 ]; then
        echo "  ✓ Successfully downloaded $scene_id"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ Failed to download $scene_id"
        FAILED=$((FAILED + 1))
    fi

    sleep 1
done < "$MISSING_FILE"

echo ""
echo "=========================================="
echo "Download Summary:"
echo "  Total scenes: $TOTAL"
echo "  Successful: $SUCCESS"
echo "  Failed: $FAILED"
echo "=========================================="
echo ""
echo "End Time: $(date)"
