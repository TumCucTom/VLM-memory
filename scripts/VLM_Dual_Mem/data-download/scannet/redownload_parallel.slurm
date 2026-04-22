#!/bin/bash
#SBATCH --job-name=redownload_scannet
#SBATCH --output=logs/redownload_scannet_%j.out
#SBATCH --error=logs/redownload_scannet_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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

# Split scenes into N chunks (N = ntasks)
N=$SLURM_NTASKS
TOTAL=$(wc -l < "$SCENES_FILE")
SPLIT_SIZE=$(( (TOTAL + N - 1) / N ))

echo "Total scenes: $TOTAL"
echo "Parallel workers: $N"
echo "Scenes per worker: ~$SPLIT_SIZE"
echo ""

for file_type in "${FILE_TYPES[@]}"; do
    echo "=========================================="
    echo "Downloading: $file_type"
    echo "=========================================="

    # Create temp dir for split files
    mkdir -p /tmp/scannet_splits

    # Split scenes into N files
    split -n l/$(($N)) -d -a 3 --additional-suffix=.txt \
        "$SCENES_FILE" \
        /tmp/scannet_splits/split_

    # Launch N parallel downloaders
    PIDS=()
    for i in $(seq 0 $((N-1))); do
        split_file=$(printf "/tmp/scannet_splits/split_%03d.txt" $i)
        if [ -s "$split_file" ]; then
            (
                COUNTER=0
                SUCCESS=0
                FAILED=0
                while IFS= read -r scene_id; do
                    if [ -z "$scene_id" ]; then
                        continue
                    fi
                    COUNTER=$((COUNTER + 1))
                    $PYTHON_CMD "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$DOWNLOAD_DIR" "$scene_id" --type "$file_type" --skip_existing
                    if [ $? -eq 0 ]; then
                        SUCCESS=$((SUCCESS + 1))
                    else
                        FAILED=$((FAILED + 1))
                    fi
                done < "$split_file"
                echo "Worker $i done: OK=$SUCCESS FAIL=$FAILED"
            ) &
            PIDS+=($!)
        fi
    done

    # Wait for all workers to finish
    for pid in "${PIDS[@]}"; do
        wait $pid
    done

    echo "All workers done for $file_type"
    echo ""
done

rm -rf /tmp/scannet_splits

echo "=========================================="
echo "All downloads complete"
echo "End Time: $(date)"