#!/bin/bash
#SBATCH --job-name=scannet_download_val_metadata
#SBATCH --output=logs/scannet_download_val_metadata_%j.out
#SBATCH --error=logs/scannet_download_val_metadata_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=workq

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

source ~/miniforge3/bin/activate
conda activate vlm3r

mkdir -p logs

# Download .txt metadata for ScanNet VAL split only
echo "Downloading .txt metadata files for ScanNet validation scenes..."

SCENES_FILE="vlm_3r_data_process/splits/scannet/scannetv2_val.txt"
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
SCANS_DIR="data/vlm_3r_data/scannet/scans"

if [ ! -f "$SCENES_FILE" ]; then
    echo "ERROR: Val scene list not found: $SCENES_FILE"
    exit 1
fi

TOTAL_SCENES=$(wc -l < "$SCENES_FILE")
echo "Found $TOTAL_SCENES val scenes to process"

# Move any existing .txt from nested scans/scans/ to correct location
if [ -d "${SCANS_DIR}/scans" ]; then
    for nested_txt in "${SCANS_DIR}"/scans/scene*/scene*.txt; do
        if [ -f "$nested_txt" ]; then
            scene_dir=$(dirname "$nested_txt")
            scene_id=$(basename "$scene_dir")
            target_dir="${SCANS_DIR}/${scene_id}"
            target_file="${target_dir}/${scene_id}.txt"
            if [ ! -f "$target_file" ]; then
                mkdir -p "$target_dir"
                mv "$nested_txt" "$target_file"
            fi
        fi
    done
fi

DOWNLOADED=0
SKIPPED=0
FAILED=0

while IFS= read -r scene_id; do
    scene_id=$(echo "$scene_id" | tr -d '\r\n' | xargs)
    [ -z "$scene_id" ] && continue

    if [ -f "${SCANS_DIR}/${scene_id}/${scene_id}.txt" ]; then
        echo "Skipping ${scene_id}: .txt already exists"
        ((SKIPPED++))
        continue
    fi
    if [ -f "${SCANS_DIR}/scans/${scene_id}/${scene_id}.txt" ]; then
        mkdir -p "${SCANS_DIR}/${scene_id}"
        mv "${SCANS_DIR}/scans/${scene_id}/${scene_id}.txt" "${SCANS_DIR}/${scene_id}/${scene_id}.txt"
        echo "✓ Moved ${scene_id}.txt from nested location"
        ((DOWNLOADED++))
        continue
    fi

    # Ensure scene dir exists (download script may create nested dir)
    mkdir -p "${SCANS_DIR}/${scene_id}"

    echo "Downloading .txt for ${scene_id}..."
    if printf 'y\ny\n' | python3 "$DOWNLOAD_SCRIPT" -o "$SCANS_DIR" --id "$scene_id" --type .txt --skip_existing > /tmp/download_val_${scene_id}.log 2>&1; then
        if [ -f "${SCANS_DIR}/scans/${scene_id}/${scene_id}.txt" ]; then
            mkdir -p "${SCANS_DIR}/${scene_id}"
            mv "${SCANS_DIR}/scans/${scene_id}/${scene_id}.txt" "${SCANS_DIR}/${scene_id}/${scene_id}.txt"
            echo "✓ Downloaded and moved ${scene_id}.txt"
            ((DOWNLOADED++))
        elif [ -f "${SCANS_DIR}/${scene_id}/${scene_id}.txt" ]; then
            echo "✓ Downloaded ${scene_id}.txt"
            ((DOWNLOADED++))
        else
            echo "✗ Download failed: ${scene_id}.txt not found after download"
            ((FAILED++))
        fi
        rm -f /tmp/download_val_${scene_id}.log
    else
        echo "✗ Download failed for ${scene_id}"
        ((FAILED++))
    fi

    if [ $(( (DOWNLOADED + SKIPPED + FAILED) % 50 )) -eq 0 ]; then
        echo "Progress: Downloaded=$DOWNLOADED, Skipped=$SKIPPED, Failed=$FAILED"
    fi
done < "$SCENES_FILE"

echo ""
echo "=========================================="
echo "Download Summary (val .txt metadata):"
echo "  Downloaded: $DOWNLOADED  Skipped: $SKIPPED  Failed: $FAILED  Total: $TOTAL_SCENES"
echo "=========================================="

echo "End Time: $(date)"
echo "Download of val metadata completed"
