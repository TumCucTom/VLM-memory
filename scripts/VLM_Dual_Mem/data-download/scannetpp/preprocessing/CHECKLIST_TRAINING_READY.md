# ScanNet++ Preprocessing vs Training Readiness

## What exists (done)

| Step | Status | Output |
|------|--------|--------|
| **1.1** Export video | Done | `data/processed_data/ScanNetpp/videos/` (737 .mp4) |
| **2.1** Scene metadata | Done | `data/processed_data/ScanNetpp/metadata/train|val/*.json` |
| **1.2** Sampled frames | Not run | Needs `camera/`, `render_depth/` in raw data (not in standard download) |
| **1.3** iPhone frames | Optional | Only if using iPhone data |
| **2.2** Frame metadata | Not run | Same raw-data requirements as 1.2 |

## What training expects

- **Video path:** `VIDEO_FOLDER` + `video` from QA JSON  
  Example: `data/vlm_3r_data` + `scannetpp/videos/<scene_id>.mp4`  
  → `data/vlm_3r_data/scannetpp/videos/<scene_id>.mp4`
- **QA JSON:** `data/vlm_3r_data/vsibench/merged_qa_scannetpp_train.json`  
  (from `scripts/VLM_3R/vsibench_data.yaml`)

Videos are currently under `data/processed_data/ScanNetpp/videos/`.  
The existing QA file uses paths like `scannetpp/videos/<id>.mp4` (lowercase `scannetpp`).

## Make ScanNet++ ready for training

Run from repo root (`/home/u5fj/trvbale.u5fj/scratch/VLM-memory`):

```bash
# 1. Make ScanNet++ videos visible under data/vlm_3r_data (lowercase for QA paths)
mkdir -p data/vlm_3r_data/scannetpp
ln -sfn "$(pwd)/data/processed_data/ScanNetpp/videos" data/vlm_3r_data/scannetpp/videos

# 2. Make merged QA visible where the training yaml points
mkdir -p data/vlm_3r_data/vsibench
ln -sfn "$(pwd)/data/scannet_vsibench/vsibench_train/merged_qa_scannetpp_train.json" data/vlm_3r_data/vsibench/merged_qa_scannetpp_train.json
```

After that, training (e.g. `scripts/VLM_3R/train_vsibench.sh` with `VIDEO_FOLDER="data/vlm_3r_data"`) will find:
- Videos at `data/vlm_3r_data/scannetpp/videos/<scene_id>.mp4`
- QA at `data/vlm_3r_data/vsibench/merged_qa_scannetpp_train.json`

## Summary

- **Preprocessing:** 1.1 and 2.1 are done; 1.2 and 2.2 are not (and need extra raw data).
- **Training-ready:** Yes for the existing ScanNet++ QA and videos, **after** creating the two symlinks above.
