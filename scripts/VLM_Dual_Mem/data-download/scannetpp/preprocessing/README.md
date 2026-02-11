# ScanNet++ Preprocessing Workflow

This directory contains SLURM scripts for preprocessing ScanNet++ data after downloading.

## Workflow Overview

The preprocessing pipeline consists of the following steps:

### Step 1: Data Extraction

1. **step1_1_export_video.slurm** - Export videos from DSLR images
   - Creates `.mp4` videos from DSLR resized undistorted images
   - Processes both train and val splits
   - Output: `data/processed_data/ScanNetpp/videos/`

2. **step1_2_prep_render_and_camera.slurm** - **Run this first** to create data needed for 1.2
   - Clones ScanNet++ toolbox to `~/scratch/scannetpp`, installs renderpy, renders DSLR depth from mesh into `data/raw_data/scannetpp/data/<scene_id>/dslr/render_depth`, and exports COLMAP to `dslr/camera/*.npz`.
   - After this completes, run **step1_2_export_sampled_frames.slurm**.

3. **step1_2_export_sampled_frames.slurm** - Sample frames with poses and intrinsics
   - Samples 32 frames per scene from DSLR data (requires `camera/` and `render_depth/` from step above)
   - Extracts color, depth, poses, and camera intrinsics
   - Output: `data/processed_data/ScanNetpp/` with subdirectories:
     - `color/train/` and `color/val/`
     - `depth/train/` and `depth/val/`
     - `pose/train/` and `pose/val/`
     - `intrinsic/train/` and `intrinsic/val/`

4. **step1_3_export_iphone_frames.slurm** - Extract iPhone frames (OPTIONAL)
   - Only needed if you want to use iPhone data instead of DSLR data
   - Extracts RGB and depth frames from iPhone video/depth data
   - Uses COLMAP poses and intrinsics
   - Output: Same structure as step1_2 but from iPhone data

### Step 2: Metadata Generation

5. **step2_1_scene_metadata.slurm** - Generate scene-level metadata
   - Processes point clouds to extract scene-wide information
   - Extracts: room size, room center, object counts, 3D bounding boxes
   - Output: 
     - `data/processed_data/ScanNetpp/metadata/train/scannetpp_metadata_train.json`
     - `data/processed_data/ScanNetpp/metadata/val/scannetpp_metadata_val.json`

6. **step2_2_frame_metadata.slurm** - Generate frame-level metadata
   - Processes sampled frames to extract frame-specific information
   - Extracts: camera poses, 2D bounding boxes, camera intrinsics
   - Output:
     - `data/processed_data/ScanNetpp/metadata/train/scannetpp_frame_metadata_train.json`
     - `data/processed_data/ScanNetpp/metadata/val/scannetpp_frame_metadata_val.json`

## Usage

### Running Individual Steps

```bash
# Step 1.1: Export videos
sbatch step1_1_export_video.slurm

# Step 1.2 prep: Render depth + export camera .npz (run once)
sbatch step1_2_prep_render_and_camera.slurm
# Step 1.2: Sample frames (DSLR) - after prep completes
sbatch step1_2_export_sampled_frames.slurm

# Step 1.3: Extract iPhone frames (optional)
sbatch step1_3_export_iphone_frames.slurm

# Step 2.1: Generate scene metadata
sbatch step2_1_scene_metadata.slurm

# Step 2.2: Generate frame metadata
sbatch step2_2_frame_metadata.slurm
```

### Running in Sequence

Steps should be run in order, as later steps depend on outputs from earlier steps:

1. Run step 1.1 and 1.2 (or 1.3 if using iPhone data)
2. Run step 2.1 (depends on raw data)
3. Run step 2.2 (depends on step 1.2/1.3 outputs)

### Dependencies

- **Step 1.1** requires: Downloaded ScanNet++ data in `data/raw_data/scannetpp/data/`
- **Step 1.2** requires: Downloaded ScanNet++ data in `data/raw_data/scannetpp/data/`
- **Step 1.3** requires: Downloaded ScanNet++ data with iPhone data
- **Step 2.1** requires: Downloaded ScanNet++ data (point clouds)
- **Step 2.2** requires: 
  - Downloaded ScanNet++ data
  - Output from step 1.2 or 1.3 (sampled frames)

## Output Structure

After running all steps, you should have:

```
data/processed_data/ScanNetpp/
├── videos/
│   ├── <scene_id>.mp4 (for each scene)
├── color/
│   ├── train/<scene_id>/<frame_id>.jpg
│   └── val/<scene_id>/<frame_id>.jpg
├── depth/
│   ├── train/<scene_id>/<frame_id>.png
│   └── val/<scene_id>/<frame_id>.png
├── pose/
│   ├── train/<scene_id>/<frame_id>.txt
│   └── val/<scene_id>/<frame_id>.txt
├── intrinsic/
│   ├── train/intrinsics_<scene_id>.txt
│   └── val/intrinsics_<scene_id>.txt
└── metadata/
    ├── train/
    │   ├── scannetpp_metadata_train.json
    │   └── scannetpp_frame_metadata_train.json
    └── val/
        ├── scannetpp_metadata_val.json
        └── scannetpp_frame_metadata_val.json
```

## Next Steps

After completing preprocessing, you can:
1. Generate QA pairs using scripts in `vlm_3r_data_process/src/tasks/`
2. Use the metadata files for training or evaluation

## Notes

- All scripts assume the project root is `/home/u5fj/trvbale.u5fj/scratch/VLM-memory`
- Scripts use the `vlm3r` conda environment
- Logs are saved to `logs/` directory in the project root
- Step 1.3 (iPhone frames) is optional - only run if you need iPhone data instead of DSLR data
