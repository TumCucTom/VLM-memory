#!/bin/bash
# Preprocess ARKitScenes: generate splits, scene metadata, mov->mp4, and copy to data/vlm_3r_data/arkitscenes/videos/.

set -e

PROJECT_DIR="${PROJECT_DIR:-$(cd -P "$(dirname "$0")/../../../../.." && pwd)}"
cd "$PROJECT_DIR"

RAW_DIR="${RAW_DIR:-data/raw_arkitscenes/raw}"
METADATA_CSV="${RAW_DIR}/metadata.csv"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed_data/ARkitScenes}"
SPLITS_DIR="${PROCESSED_DIR}/splits"
VLM3R_VIDEOS="${VLM3R_VIDEOS:-data/vlm_3r_data/arkitscenes/videos}"

echo "=========================================="
echo "ARKitScenes preprocess"
echo "=========================================="
echo "Project: $PROJECT_DIR"
echo "Raw: $RAW_DIR"
echo "Processed: $PROCESSED_DIR"
echo "VLM-3R videos: $VLM3R_VIDEOS"
echo ""

if [ ! -f "$METADATA_CSV" ]; then
    echo "ERROR: Metadata CSV not found: $METADATA_CSV. Run 01_download first."
    exit 1
fi

# 1. Generate train.txt and val.txt from metadata.csv
echo "Step 1: Generating train/val split files..."
mkdir -p "$SPLITS_DIR"
python3 scripts/VLM_Dual_Mem/data-download/arkit-scenes/generate_splits_from_metadata.py \
    "$METADATA_CSV" "$SPLITS_DIR"

# 2. Scene metadata (Training and Validation)
echo "Step 2: Generating scene metadata (Training + Validation)..."
cd vlm_3r_data_process
for SPLIT in Training Validation; do
    LOW=$(echo "$SPLIT" | tr '[:upper:]' '[:lower:]')
    OUT_NAME="arkitscenes_${LOW}_metadata_seed0.json"
    SAVE_DIR="../${PROCESSED_DIR}/metadata/${LOW}"
    mkdir -p "$SAVE_DIR"
    python -m src.metadata_generation.ARkitScenes.arkitscenes_metadata \
        --root_dir "../${RAW_DIR}" \
        --metadata_csv "../${METADATA_CSV}" \
        --annotations_dir "../${RAW_DIR}" \
        --split "$SPLIT" \
        --save_dir "$SAVE_DIR" \
        --output_filename "$OUT_NAME" \
        --num_workers 4 \
        --overwrite
done
cd ..

# 3. Convert .mov to .mp4 (copy_video) for train and val
echo "Step 3: Converting .mov to .mp4..."
cd vlm_3r_data_process
python3 -c '
from pathlib import Path
from src.metadata_generation.ARkitScenes.copy_video import convert_and_copy_videos
proj = Path("..").resolve()
raw = proj / "data/raw_arkitscenes/raw"
out = proj / "data/processed_data/ARkitScenes"
splits = proj / "data/processed_data/ARkitScenes/splits"
convert_and_copy_videos(str(splits / "train.txt"), str(raw / "Training"), str(out), "train")
convert_and_copy_videos(str(splits / "val.txt"), str(raw / "Validation"), str(out), "val")
'
cd ..

# 4. Copy (or symlink) to training-ready location: data/vlm_3r_data/arkitscenes/videos/<id>.mp4
echo "Step 4: Copying videos to $VLM3R_VIDEOS..."
mkdir -p "$VLM3R_VIDEOS"
for sub in train val; do
    SRC="${PROCESSED_DIR}/videos/${sub}"
    if [ -d "$SRC" ]; then
        for f in "$SRC"/*.mp4; do
            [ -f "$f" ] || continue
            name=$(basename "$f")
            if [ ! -f "${VLM3R_VIDEOS}/${name}" ]; then
                cp -n "$f" "${VLM3R_VIDEOS}/${name}"
            fi
        done
        echo "  Copied from $SRC to $VLM3R_VIDEOS"
    fi
done
echo "Done. Training expects: $VLM3R_VIDEOS/<video_id>.mp4 (with VIDEO_FOLDER=data/vlm_3r_data)."
echo ""
