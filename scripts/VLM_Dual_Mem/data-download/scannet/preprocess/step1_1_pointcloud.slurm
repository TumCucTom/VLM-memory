#!/bin/bash
#SBATCH --job-name=scannet_pointcloud
#SBATCH --output=logs/scannet_pointcloud_%j.out
#SBATCH --error=logs/scannet_pointcloud_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=workq

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"


# Activate Python virtual environment
source ~/miniforge3/bin/activate
conda activate vlm3r

# Create logs directory if it doesn't exist
mkdir -p logs

# Note: This step requires the ScanNet BenchmarkS
# Check if benchmark scripts exist
if [ ! -d "datasets/ScanNet/BenchmarkScripts/ScanNet200" ]; then
    echo "ERROR: ScanNet BenchmarkScripts not found!"
    echo "Expected location: datasets/ScanNet/BenchmarkScripts/ScanNet200/"
    exit 1
fi

# Check if label map file exists
if [ ! -f "data/raw_data/scannet/scannetv2-labels.combined.tsv" ]; then
    echo "ERROR: Label map file not found!"
    echo "Expected: data/raw_data/scannet/scannetv2-labels.combined.tsv"
    echo "Please download from ScanNet repository"
    exit 1
fi

# Check if scene metadata .txt files exist (required for preprocessing)
# Sample a few scenes to check
SAMPLE_SCENES=("scene0000_00" "scene0001_00" "scene0002_00")
MISSING_TXT=0
for scene in "${SAMPLE_SCENES[@]}"; do
    if [ ! -f "data/vlm_3r_data/scannet/scans/${scene}/${scene}.txt" ]; then
        MISSING_TXT=1
        break
    fi
done

if [ $MISSING_TXT -eq 1 ]; then
    echo "ERROR: Scene metadata .txt files are missing!"
    echo "The preprocessing script requires .txt files in each scene directory."
    echo "These contain axis alignment and camera intrinsics information."
    echo ""
    echo "You need to download .txt files for all scenes first."
    echo "Create a script to download them using:"
    echo "  python scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py \\"
    echo "    -o data/vlm_3r_data/scannet/scans \\"
    echo "    --type .txt"
    echo ""
    echo "Or use the download script with --type .txt for each scene."
    exit 1
fi

# Check for and handle any nested 'scans' directory that might cause issues
# The preprocessing script uses glob.glob() which will pick up nested directories
SCANS_DIR="data/vlm_3r_data/scannet/scans"
if [ -d "${SCANS_DIR}/scans" ]; then
    echo "WARNING: Found nested 'scans' directory. Moving scene directories to correct location..."
    
    # Move any scene directories from nested location to correct location
    MOVED=0
    for nested_scene in "${SCANS_DIR}"/scans/scene*; do
        if [ -d "$nested_scene" ]; then
            scene_id=$(basename "$nested_scene")
            target_dir="${SCANS_DIR}/${scene_id}"
            
            if [ ! -d "$target_dir" ]; then
                # Target doesn't exist, move the entire directory
                mv "$nested_scene" "$target_dir"
                ((MOVED++))
            else
                # Target exists, merge files (move files that don't exist in target)
                for file in "$nested_scene"/*; do
                    if [ -f "$file" ]; then
                        filename=$(basename "$file")
                        if [ ! -f "${target_dir}/${filename}" ]; then
                            mv "$file" "${target_dir}/${filename}"
                            ((MOVED++))
                        fi
                    fi
                done
                # Remove the now-empty nested scene directory
                rmdir "$nested_scene" 2>/dev/null || true
            fi
        fi
    done
    
    # Remove the nested scans directory if it's now empty
    if [ -d "${SCANS_DIR}/scans" ]; then
        if [ -z "$(ls -A "${SCANS_DIR}/scans" 2>/dev/null)" ]; then
            rmdir "${SCANS_DIR}/scans"
        else
            echo "  Note: Some files/directories remain in nested scans directory"
        fi
    fi
    
    if [ $MOVED -gt 0 ]; then
        echo "✓ Moved $MOVED scene directories/files from nested location"
    else
        echo "✓ Cleaned up nested scans directory"
    fi
fi

# Create output directory
mkdir -p data/processed_data/ScanNet/point_cloud

# Run Step 1.1: Process ScanNet200 to get .ply files with labels and instance IDs
echo "Starting Step 1.1: Processing ScanNet200 point clouds..."

# Store project root for absolute paths (needed because BenchmarkScripts is a symlink)
PROJECT_ROOT=$(pwd)
cd datasets/ScanNet/BenchmarkScripts/ScanNet200/

python preprocess_scannet200.py \
    --dataset_root "${PROJECT_ROOT}/data/vlm_3r_data/scannet/scans" \
    --output_root "${PROJECT_ROOT}/data/processed_data/ScanNet/point_cloud" \
    --label_map_file "${PROJECT_ROOT}/data/raw_data/scannet/scannetv2-labels.combined.tsv" \
    --train_val_splits_path "${PROJECT_ROOT}/vlm_3r_data_process/splits/scannet" \
    --num_workers 8

cd "${PROJECT_ROOT}"

echo "End Time: $(date)"
echo "Step 1.1 completed"