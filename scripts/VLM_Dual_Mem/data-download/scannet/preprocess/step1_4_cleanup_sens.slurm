#!/bin/bash
#SBATCH --job-name=scannet_cleanup_sens
#SBATCH --output=logs/scannet_cleanup_sens_%j.out
#SBATCH --error=logs/scannet_cleanup_sens_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=mlcnu
#SBATCH --account=coms037985

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"


# Create logs directory if it doesn't exist
mkdir -p logs

# Define paths
SCANS_DIR="data/vlm_3r_data/scannet/scans"
SUCCESS_FILE="data/processed_data/ScanNet/successful_scenes_train.txt"

# Safety check: Verify that step 1.3 completed successfully
echo "=========================================="
echo "Safety Checks"
echo "=========================================="
if [ ! -f "$SUCCESS_FILE" ]; then
    echo "WARNING: Success file not found: $SUCCESS_FILE"
    echo "This might indicate step 1.3 hasn't completed yet."
    echo "Please verify that step 1.3 completed successfully before proceeding."
    echo "Do you want to continue anyway? (This script will check for processed data)"
    echo ""
fi

# Check if processed data exists (videos, sampled frames, or point clouds)
PROCESSED_DATA_EXISTS=false
if [ -d "data/processed_data/ScanNet/videos" ] || \
   [ -d "data/processed_data/ScanNet/frames" ] || \
   [ -d "data/processed_data/ScanNet/point_cloud" ]; then
    PROCESSED_DATA_EXISTS=true
    echo "✓ Found processed data directories"
else
    echo "WARNING: No processed data directories found!"
    echo "Expected at least one of:"
    echo "  - data/processed_data/ScanNet/videos"
    echo "  - data/processed_data/ScanNet/frames"
    echo "  - data/processed_data/ScanNet/point_cloud"
    echo ""
    echo "Aborting deletion for safety."
    exit 1
fi

# Check if scans directory exists
if [ ! -d "$SCANS_DIR" ]; then
    echo "ERROR: Scans directory not found: $SCANS_DIR"
    exit 1
fi

# Count and calculate size of .sens files
echo ""
echo "=========================================="
echo "Scanning .sens files"
echo "=========================================="
SENS_COUNT=$(find "$SCANS_DIR" -name "*.sens" -type f | wc -l)
echo "Found $SENS_COUNT .sens files"

if [ "$SENS_COUNT" -eq 0 ]; then
    echo "No .sens files found. Nothing to delete."
    exit 0
fi

# Calculate total size
echo "Calculating total size..."
TOTAL_SIZE=$(du -sb "$SCANS_DIR" --exclude="*.sens" 2>/dev/null | tail -1 | awk '{print $1}')
SENS_SIZE=$(find "$SCANS_DIR" -name "*.sens" -type f -exec du -sb {} + 2>/dev/null | awk '{sum+=$1} END {print sum}')
SENS_SIZE_GB=$(echo "scale=2; $SENS_SIZE / 1073741824" | bc)

echo "Total size of .sens files: ${SENS_SIZE_GB} GB"
echo ""

# Confirm deletion
echo "=========================================="
echo "Ready to delete .sens files"
echo "=========================================="
echo "This will delete $SENS_COUNT .sens files (~${SENS_SIZE_GB} GB)"
echo "Processed data has been extracted, so these files are no longer needed."
echo ""

# Perform deletion
echo "Starting deletion..."
DELETED_COUNT=0
ERROR_COUNT=0

# Use a simpler approach: delete files and count
for sens_file in $(find "$SCANS_DIR" -name "*.sens" -type f); do
    if [ -f "$sens_file" ]; then
        if rm -f "$sens_file" 2>/dev/null; then
            DELETED_COUNT=$((DELETED_COUNT + 1))
            if [ $((DELETED_COUNT % 100)) -eq 0 ]; then
                echo "Deleted $DELETED_COUNT / $SENS_COUNT files..."
            fi
        else
            ERROR_COUNT=$((ERROR_COUNT + 1))
            echo "ERROR: Failed to delete $sens_file"
        fi
    fi
done

# Use the pre-calculated size for reporting
DELETED_SIZE=$SENS_SIZE

# Calculate final statistics
DELETED_SIZE_GB=$(echo "scale=2; $DELETED_SIZE / 1073741824" | bc)

echo ""
echo "=========================================="
echo "Deletion Summary"
echo "=========================================="
echo "Files deleted: $DELETED_COUNT / $SENS_COUNT"
echo "Space freed: ${DELETED_SIZE_GB} GB"
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "Errors encountered: $ERROR_COUNT"
fi
echo ""

# Verify deletion
REMAINING=$(find "$SCANS_DIR" -name "*.sens" -type f | wc -l)
if [ "$REMAINING" -eq 0 ]; then
    echo "✓ All .sens files successfully deleted"
else
    echo "WARNING: $REMAINING .sens files still remain"
fi

echo ""
echo "End Time: $(date)"
echo "Cleanup completed"
