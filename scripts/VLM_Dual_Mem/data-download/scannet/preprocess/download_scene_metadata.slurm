#!/bin/bash
#SBATCH --job-name=scannet_download_txt
#SBATCH --output=logs/scannet_download_txt_%j.out
#SBATCH --error=logs/scannet_download_txt_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=mlcnu
#SBATCH --account=coms037985

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Change to project directory
cd ~/VLM-memory

# Activate Python virtual environment
source ~/venvs/vlm3r/bin/activate

# Ensure pexpect is available (needed for wrapper script)
python3 -c "import pexpect" 2>/dev/null || pip install pexpect

# Create logs directory if it doesn't exist
mkdir -p logs

# Download .txt metadata files for all ScanNet scenes
# This is required before running step1_1_pointcloud.slurm
# We download per scene to ensure .txt files are added to existing directories
echo "Downloading .txt metadata files for all ScanNet scenes..."
echo "This will download .txt files for each scene individually to avoid overwriting existing data..."

# Get list of scenes
SCENES_FILE="data/vlm_3r_data/vsibench/scannet_scenes.txt"
if [ ! -f "$SCENES_FILE" ]; then
    echo "ERROR: Scene list file not found: $SCENES_FILE"
    exit 1
fi

TOTAL_SCENES=$(wc -l < "$SCENES_FILE")
echo "Found $TOTAL_SCENES scenes to process"

# First, move any existing .txt files from nested scans/scans/ to correct location
echo "Checking for .txt files in nested location and moving them..."
MOVED=0
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
                ((MOVED++))
            fi
        fi
    done
    if [ $MOVED -gt 0 ]; then
        echo "Moved $MOVED .txt files from nested location to correct location"
    fi
fi

# Download .txt files for each scene
# Use the wrapper script to handle interactive prompts
DOWNLOAD_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"
SCANS_DIR="data/vlm_3r_data/scannet/scans"

DOWNLOADED=0
SKIPPED=0
FAILED=0

while IFS= read -r scene_id; do
    scene_id=$(echo "$scene_id" | tr -d '\r\n' | xargs)  # Clean whitespace
    if [ -z "$scene_id" ]; then
        continue
    fi
    
    # Check if .txt file already exists (check both possible locations)
    if [ -f "${SCANS_DIR}/${scene_id}/${scene_id}.txt" ] || [ -f "${SCANS_DIR}/scans/${scene_id}/${scene_id}.txt" ]; then
        # If it's in the nested location, move it to the correct location
        if [ -f "${SCANS_DIR}/scans/${scene_id}/${scene_id}.txt" ] && [ ! -f "${SCANS_DIR}/${scene_id}/${scene_id}.txt" ]; then
            echo "Moving ${scene_id}.txt from nested location..."
            mkdir -p "${SCANS_DIR}/${scene_id}"
            mv "${SCANS_DIR}/scans/${scene_id}/${scene_id}.txt" "${SCANS_DIR}/${scene_id}/${scene_id}.txt"
            echo "✓ Moved ${scene_id}.txt to correct location"
        else
            echo "Skipping ${scene_id}: .txt file already exists"
        fi
        ((SKIPPED++))
        continue
    fi
    
    # Check if scene directory exists (should exist with .sens file)
    if [ ! -d "${SCANS_DIR}/${scene_id}" ]; then
        echo "WARNING: Scene directory not found: ${SCANS_DIR}/${scene_id}"
        ((FAILED++))
        continue
    fi
    
    # Download .txt file for this scene
    # Use printf to send two 'y' responses (TOS agreement + file type confirmation)
    # Note: The download script creates a nested scans/ directory, so we'll handle that
    echo "Downloading .txt for ${scene_id}..."
    if printf 'y\ny\n' | python3 "$DOWNLOAD_SCRIPT" -o "$SCANS_DIR" --id "$scene_id" --type .txt --skip_existing > /tmp/download_${scene_id}.log 2>&1; then
        # Check both possible locations (nested and correct)
        if [ -f "${SCANS_DIR}/scans/${scene_id}/${scene_id}.txt" ]; then
            # Move from nested location to correct location
            mkdir -p "${SCANS_DIR}/${scene_id}"
            mv "${SCANS_DIR}/scans/${scene_id}/${scene_id}.txt" "${SCANS_DIR}/${scene_id}/${scene_id}.txt"
            echo "✓ Downloaded and moved ${scene_id}.txt to correct location"
            ((DOWNLOADED++))
            rm -f /tmp/download_${scene_id}.log
        elif [ -f "${SCANS_DIR}/${scene_id}/${scene_id}.txt" ]; then
            echo "✓ Downloaded ${scene_id}.txt"
            ((DOWNLOADED++))
            rm -f /tmp/download_${scene_id}.log
        else
            echo "✗ Download failed: ${scene_id}.txt not found after download"
            echo "  Check /tmp/download_${scene_id}.log for details"
            ((FAILED++))
        fi
    else
        echo "✗ Download failed for ${scene_id}"
        echo "  Check /tmp/download_${scene_id}.log for details"
        ((FAILED++))
    fi
    
    # Progress update every 100 scenes
    if [ $(( (DOWNLOADED + SKIPPED + FAILED) % 100 )) -eq 0 ]; then
        echo "Progress: Downloaded=$DOWNLOADED, Skipped=$SKIPPED, Failed=$FAILED"
    fi
done < "$SCENES_FILE"

echo ""
echo "=========================================="
echo "Download Summary:"
echo "  Downloaded: $DOWNLOADED"
echo "  Skipped (already exists): $SKIPPED"
echo "  Failed: $FAILED"
echo "  Total: $TOTAL_SCENES"
echo "=========================================="

echo "End Time: $(date)"
echo "Download of .txt metadata files completed"
