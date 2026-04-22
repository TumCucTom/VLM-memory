#!/bin/bash
#SBATCH --job-name=redownload_scannet
#SBATCH --output=logs/redownload_scannet_%j.out
#SBATCH --error=logs/redownload_scannet_%j.err
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

SCENES_FILE="vlm_3r_data_process/splits/scannet/scannetv2_train.txt"
DOWNLOAD_DIR="data/vlm_3r_data/scannet/scans"
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"

FILE_TYPES=(
    ".txt"
    "_vh_clean_2.ply"
    "_vh_clean_2.0.010000.segs.json"
    ".aggregation.json"
)

for file_type in "${FILE_TYPES[@]}"; do
    echo "=========================================="
    echo "Downloading: $file_type"
    echo "=========================================="

    COUNTER=0
    SUCCESS=0
    FAILED=0

    while IFS= read -r scene_id; do
        if [ -z "$scene_id" ]; then
            continue
        fi

        COUNTER=$((COUNTER + 1))
        echo "[$COUNTER] $scene_id..."

        $PYTHON_CMD "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$scene_id" --type "$file_type" --skip_existing

        if [ $? -eq 0 ]; then
            SUCCESS=$((SUCCESS + 1))
        else
            FAILED=$((FAILED + 1))
        fi

        if [ $((COUNTER % 50)) -eq 0 ]; then
            echo "Progress: $COUNTER (OK: $SUCCESS, FAIL: $FAILED)"
        fi

        sleep 1
    done < "$SCENES_FILE"

    echo "Done $file_type: OK=$SUCCESS FAIL=$FAILED"
    echo ""
done

echo "=========================================="
echo "All files re-downloaded"
echo "=========================================="
echo "End Time: $(date)"