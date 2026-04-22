#!/bin/bash
#SBATCH --job-name=retry_scannet_scene
#SBATCH --output=logs/retry_scannet_%j.out
#SBATCH --error=logs/retry_scannet_%j.err
#SBATCH --time=04:00:00
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

MISSING_FILE="scripts/VLM_Dual_Mem/data-download/scannet/retry_scene.txt"
DOWNLOAD_DIR="data/vlm_3r_data/scannet/scans"
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"

TOTAL=$(wc -l < "$MISSING_FILE")
echo "Retrying $TOTAL missing scene(s)"
echo ""

while IFS= read -r scene_id; do
    if [ -z "$scene_id" ]; then
        continue
    fi
    echo "Downloading $scene_id..."
    $PYTHON_CMD "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$scene_id"
done < "$MISSING_FILE"

echo ""
echo "Verification:"
FOUND=$(find data/vlm_3r_data/scannet/scans/scans -name "*.sens" | wc -l)
echo "Total .sens files found: $FOUND"
MISSING=$(comm -23 vlm_3r_data_process/splits/scannet/scannetv2_train.txt <(find data/vlm_3r_data/scannet/scans/scans -name "*.sens" -printf "%f\n" | sed 's/.sens$//' | sort) | wc -l)
echo "Missing from expected list: $MISSING"

echo "End Time: $(date)"