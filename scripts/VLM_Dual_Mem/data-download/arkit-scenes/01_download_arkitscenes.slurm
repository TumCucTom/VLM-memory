#!/bin/bash
#SBATCH --job-name=arkit_download
#SBATCH --output=logs/download_arkitscenes_%j.out
#SBATCH --error=logs/download_arkitscenes_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=workq

# Download ARKitScenes raw data and metadata to data/raw_arkitscenes/raw.
# Long-running I/O-bound job; run after configuring PROJECT_DIR if needed.

echo "=========================================="
echo "ARKitScenes Download Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# Activate Python virtual environment
source ~/miniforge3/bin/activate
conda activate vlm3r

# Project root (change if not running from workspace root)
PROJECT_DIR="${PROJECT_DIR:-/home/u5fj/trvbale.u5fj/scratch/VLM-memory}"
cd "$PROJECT_DIR"
mkdir -p logs

RAW_DIR="${PROJECT_DIR}/data/raw_arkitscenes/raw"
METADATA_CSV="${RAW_DIR}/metadata.csv"

# Run download in background so we can log progress
bash scripts/VLM_Dual_Mem/data-download/arkit-scenes/download_arkitscenes.sh &
DOWNLOAD_PID=$!

# Wait for metadata to exist (created at start of download script), then get total video count
echo "Waiting for metadata to determine total videos..."
for _ in $(seq 1 60); do
    if [ -f "$METADATA_CSV" ]; then
        TOTAL_VIDEOS=$(($(wc -l < "$METADATA_CSV") - 1))  # subtract header
        echo "Total videos to download: $TOTAL_VIDEOS"
        break
    fi
    sleep 5
done

# Log progress every 2 minutes while download is running
while kill -0 "$DOWNLOAD_PID" 2>/dev/null; do
    sleep 120
    if ! kill -0 "$DOWNLOAD_PID" 2>/dev/null; then
        break
    fi
    if [ -d "${RAW_DIR}/Training" ] || [ -d "${RAW_DIR}/Validation" ]; then
        DOWNLOADED=$(find "${RAW_DIR}/Training" "${RAW_DIR}/Validation" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        if [ -n "${TOTAL_VIDEOS:-}" ] && [ "$TOTAL_VIDEOS" -gt 0 ]; then
            PCT=$((DOWNLOADED * 100 / TOTAL_VIDEOS))
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Progress: $DOWNLOADED / $TOTAL_VIDEOS videos ($PCT%)"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Progress: $DOWNLOADED videos downloaded"
        fi
    fi
done

wait "$DOWNLOAD_PID"
DOWNLOAD_EXIT=$?

# Final progress line
if [ -d "${RAW_DIR}/Training" ] || [ -d "${RAW_DIR}/Validation" ]; then
    DOWNLOADED=$(find "${RAW_DIR}/Training" "${RAW_DIR}/Validation" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ -n "${TOTAL_VIDEOS:-}" ] && [ "$TOTAL_VIDEOS" -gt 0 ]; then
        PCT=$((DOWNLOADED * 100 / TOTAL_VIDEOS))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Final: $DOWNLOADED / $TOTAL_VIDEOS videos ($PCT%)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Final: $DOWNLOADED videos downloaded"
    fi
fi

echo ""
echo "End: $(date)"
if [ "$DOWNLOAD_EXIT" -eq 0 ]; then
    echo "Job completed"
else
    echo "Job finished with exit code $DOWNLOAD_EXIT"
    exit "$DOWNLOAD_EXIT"
fi
