#!/bin/bash
#SBATCH --job-name=download_val_all
#SBATCH --output=logs/download_val_all_%j.out
#SBATCH --error=logs/download_val_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"

source ~/miniforge3/bin/activate
conda activate vlm3r

PYTHON_CMD="${PYTHON_CMD:-$(command -v python || command -v python3)}"

# NOTE: out_dir is data/vlm_3r_data/scannet (NOT data/vlm_3r_data/scannet/scans)
# The download script appends 'scans' to out_dir, so files go to data/vlm_3r_data/scannet/scans/sceneXXXX/
# which is correct since sens files are at data/vlm_3r_data/scannet/scans/sceneXXXX/
SCENES_FILE="vlm_3r_data_process/splits/scannet/scannetv2_val.txt"
DOWNLOAD_DIR="data/vlm_3r_data/scannet"  # NOT /scans
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"

FILE_TYPES=(
    ".txt"
    "_vh_clean_2.ply"
    "_vh_clean_2.0.010000.segs.json"
    ".aggregation.json"
)

N=$SLURM_NTASKS
TOTAL=$(wc -l < "$SCENES_FILE")
echo "Val scenes: $TOTAL, Workers: $N"
echo ""

mkdir -p /tmp/scannet_val_splits
split -n l/$(($N)) -d -a 3 --additional-suffix=.txt "$SCENES_FILE" /tmp/scannet_val_splits/split_

for file_type in "${FILE_TYPES[@]}"; do
    echo "Downloading: $file_type"

    PIDS=()
    for i in $(seq 0 $((N-1))); do
        split_file=$(printf "/tmp/scannet_val_splits/split_%03d.txt" $i)
        if [ -s "$split_file" ]; then
            (
                while IFS= read -r scene_id; do
                    [ -z "$scene_id" ] && continue
                    $PYTHON_CMD "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$scene_id" --type "$file_type" --skip_existing
                done < "$split_file"
            ) &
            PIDS+=($!)
        fi
    done

    for pid in "${PIDS[@]}"; do wait $pid; done
    echo "  Done: $file_type"
done

rm -rf /tmp/scannet_val_splits
echo "End Time: $(date)"