#!/bin/bash
# Fetch the missing ScanNet scene scene0706_00 and export it to the training video location.
# Run from repo root. ScanNet download requires accepting TOS (interactive or pexpect).
set -e
SCENE_ID="scene0706_00"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"
DOWNLOAD_SCRIPT="$REPO_ROOT/scripts/VLM_Dual_Mem/data-download/scannet/download-scannet.py"
WRAPPER_SCRIPT="$REPO_ROOT/scripts/VLM_Dual_Mem/data-download/scannet/download_scannet_wrapper.py"
SCANS_DIR="$REPO_ROOT/data/vlm_3r_data/scannet/scans"
VIDEO_DIR="$REPO_ROOT/data/vlm_3r_data/scannet/videos"
mkdir -p "$SCANS_DIR" "$VIDEO_DIR"

echo "=========================================="
echo "Fetch and export $SCENE_ID for route-plan training"
echo "=========================================="

# Download script expects parent of "scans" (it appends /scans and then /scene_id)
SCANS_PARENT="$REPO_ROOT/data/vlm_3r_data/scannet"

# 1. Download .sens for scene0706_00 (may prompt for ScanNet TOS)
if [ -f "$SCANS_DIR/$SCENE_ID/$SCENE_ID.sens" ]; then
  echo "Already have $SCENE_ID.sens, skipping download."
else
  echo "Downloading $SCENE_ID (ScanNet TOS may be prompted)..."
  PYTHON_CMD="${PYTHON_CMD:-$(command -v python || command -v python3)}"
  "$PYTHON_CMD" "$WRAPPER_SCRIPT" "$DOWNLOAD_SCRIPT" "$SCANS_PARENT" "$SCENE_ID" --skip_existing
  # Script writes to $SCANS_PARENT/scans/$SCENE_ID/; move if it landed in scans/scans/
  if [ -f "$SCANS_DIR/scans/$SCENE_ID/$SCENE_ID.sens" ]; then
    mkdir -p "$SCANS_DIR/$SCENE_ID"
    mv "$SCANS_DIR/scans/$SCENE_ID/$SCENE_ID.sens" "$SCANS_DIR/$SCENE_ID/"
    rmdir "$SCANS_DIR/scans/$SCENE_ID" 2>/dev/null || true
    rmdir "$SCANS_DIR/scans" 2>/dev/null || true
  fi
  if [ ! -f "$SCANS_DIR/$SCENE_ID/$SCENE_ID.sens" ]; then
    echo "Error: Download did not produce $SCANS_DIR/$SCENE_ID/$SCENE_ID.sens"
    exit 1
  fi
fi

# 2. Export video to training location (matches other ScanNet videos: 640x480, 24fps)
if [ -f "$VIDEO_DIR/$SCENE_ID.mp4" ]; then
  echo "Already have $VIDEO_DIR/$SCENE_ID.mp4, skipping export."
else
  echo "Exporting video to $VIDEO_DIR/$SCENE_ID.mp4 ..."
  cd "$REPO_ROOT/vlm_3r_data_process"
  PYTHON_EXPORT="${PYTHON_EXPORT:-$(command -v python || command -v python3)}"
  "$PYTHON_EXPORT" -m src.metadata_generation.ScanNet.preprocess.export_video \
    --scans_dir "$SCANS_DIR" \
    --output_dir "$VIDEO_DIR" \
    --width 640 \
    --height 480 \
    --fps 24 \
    --frame_skip 1
  cd ..
  if [ ! -f "$VIDEO_DIR/$SCENE_ID.mp4" ]; then
    echo "Error: Export did not produce $VIDEO_DIR/$SCENE_ID.mp4"
    exit 1
  fi
fi

echo "Done: $VIDEO_DIR/$SCENE_ID.mp4"
ls -la "$VIDEO_DIR/$SCENE_ID.mp4"
